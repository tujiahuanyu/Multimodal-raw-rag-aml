import base64
from dataclasses import replace
import io
import json

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import pytest

from evaluation.src.aml.app import create_app
from evaluation.src.aml.config import Settings, endpoint
from evaluation.src.aml.providers import Providers, ServiceError
from evaluation.src.aml.store import Store


class FakeProviders:
    def __init__(self):
        self.embedding_inputs = []
        self.caption_inputs = []
        self.detail_inputs = []
        self.rerank_inputs = []
        self.rerank_limits = []
        self.fail_embedding = False
        self.fail_caption = False
        self.fail_details = False

    async def close(self):
        pass

    async def caption(self, url, context):
        self.caption_inputs.append((url, context))
        if self.fail_caption:
            raise ServiceError(502, 'vision_http_403')
        return 'A red backpack.'

    async def embed(self, texts):
        self.embedding_inputs.extend(texts)
        if self.fail_embedding:
            raise ServiceError(502, 'embedding_http_403')
        matrix = np.asarray(
            [
                [
                    1,
                    text.lower().count('red'),
                    text.lower().count('blue'),
                    text.lower().count('paris'),
                ]
                for text in texts
            ],
            dtype=np.float32,
        )
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    async def image_details(self, url):
        self.detail_inputs.append(url)
        if self.fail_details:
            raise ServiceError(502, 'vision_http_403')
        return ['A red backpack.', 'Blue straps cross the front of the backpack.']

    async def rerank(self, query, candidates, limit):
        self.rerank_inputs.append((query, candidates))
        self.rerank_limits.append(limit)
        return [
            {**row, 'score': 1 - position / max(1, len(candidates))}
            for position, row in enumerate(candidates[:limit])
        ]


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / 'store',
        api_token='test-only-token-' * 3,
        vision_url='https://vision.example/v1/chat/completions',
        vision_key='vision-key',
        vision_model='gpt-4o-mini',
        embedding_url='https://embedding.example/v1/embeddings',
        embedding_key='embedding-key',
        embedding_model='text-embedding-v4',
        rerank_url='https://rerank.example/v2/rerank',
        rerank_key='rerank-key',
        rerank_model='rerank-v3.5',
    )


@pytest.fixture
def backend():
    return FakeProviders()


@pytest.fixture
def client(settings, backend):
    with TestClient(
        create_app(settings, backend), raise_server_exceptions=False
    ) as instance:
        instance.headers['Authorization'] = 'Token ' + settings.api_token
        yield instance


def add_payload(
    user='user-1',
    request='request-1',
    session='session-1',
    content='I took a red backpack to Paris.',
):
    return {
        'request_id': request,
        'user_id': user,
        'session_id': session,
        'messages': [{'role': 'user', 'content': content, 'timestamp': 1704067200000}],
    }


def search_payload(user='user-1', query='red backpack'):
    return {'user_id': user, 'query': query, 'top_k': 100}


def image_url(color='red', image_format='PNG'):
    buffer = io.BytesIO()
    Image.new('RGB', (12, 12), color).save(buffer, format=image_format)
    mime = {
        'PNG': 'image/png',
        'JPEG': 'image/jpeg',
        'WEBP': 'image/webp',
        'GIF': 'image/gif',
    }[image_format]
    return 'data:' + mime + ';base64,' + base64.b64encode(buffer.getvalue()).decode()


def image_part(url=None):
    return {'type': 'image_url', 'image_url': {'url': url or image_url()}}


def test_health_is_public_but_memory_requires_auth(client):
    client.headers.pop('Authorization')
    assert client.get('/health').status_code == 200
    assert client.post('/v1/memories/add', json=add_payload()).status_code == 401
    assert client.post('/v1/memories/search', json=search_payload()).status_code == 401
    assert client.get('/docs').status_code == 401


@pytest.mark.parametrize('scheme', ['Token', 'Bearer', 'X-Api-Key'])
def test_all_supported_authentication_schemes(client, settings, scheme):
    client.headers.pop('Authorization')
    client.headers['X-Api-Key' if scheme == 'X-Api-Key' else 'Authorization'] = (
        settings.api_token
        if scheme == 'X-Api-Key'
        else scheme + ' ' + settings.api_token
    )
    assert client.post('/v1/memories/add', json=add_payload()).status_code == 200


def test_add_echoes_identifiers_and_search_returns_evidence(client):
    payload = add_payload(user='用户:../1', session='会话/a')
    reply = client.post('/v1/memories/add', json=payload)
    assert reply.status_code == 200
    assert reply.json() == {
        'success': True,
        'request_id': 'request-1',
        'user_id': payload['user_id'],
        'session_id': payload['session_id'],
    }
    result = client.post(
        '/v1/memories/search', json=search_payload(user=payload['user_id'])
    )
    assert result.status_code == 200
    row = result.json()['data'][0]
    assert row['id']
    assert 'I took a red backpack to Paris.' in row['content']
    assert row['created_at'] == '2024-01-01T00:00:00+00:00'


def test_duplicate_request_has_no_extra_api_calls(client, backend):
    payload = add_payload()
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    count = len(backend.embedding_inputs)
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    assert len(backend.embedding_inputs) == count
    payload['messages'][0]['content'] = 'Different body'
    assert client.post('/v1/memories/add', json=payload).status_code == 409


def test_user_id_is_the_only_retrieval_scope(client, backend):
    assert client.post('/v1/memories/add', json=add_payload()).status_code == 200
    count = len(backend.embedding_inputs)
    assert client.post(
        '/v1/memories/search', json=search_payload(user='another-user')
    ).json() == {'data': []}
    assert len(backend.embedding_inputs) == count
    assert (
        client.post(
            '/v1/memories/add', json=add_payload(user='another-user')
        ).status_code
        == 200
    )


def test_search_includes_all_sessions_and_separate_choice_options(client, backend):
    client.post('/v1/memories/add', json=add_payload())
    client.post(
        '/v1/memories/add',
        json=add_payload(
            request='request-2', session='session-2', content='A blue bicycle.'
        ),
    )
    payload = {**search_payload(), 'options': ['A. red', 'B. blue']}
    result = client.post('/v1/memories/search', json=payload)
    assert result.status_code == 200
    assert len(result.json()['data']) == 2
    assert 'A. red' in backend.rerank_inputs[-1][0]


def test_incremental_append_reuses_closed_window_embeddings(client, backend):
    client.post('/v1/memories/add', json=add_payload(content='red ' * 800))
    previous = list(backend.embedding_inputs)
    assert len(previous) > 2
    reply = client.post(
        '/v1/memories/add', json=add_payload(request='request-2', content='blue ' * 100)
    )
    assert reply.status_code == 200
    assert previous[0] not in backend.embedding_inputs[len(previous) :]
    evidence = client.post(
        '/v1/memories/search', json=search_payload(query='blue')
    ).json()['data']
    assert any('blue' in row['content'] for row in evidence)


def test_multimodal_parts_preserve_pixels_order_and_source_binding(client, backend):
    red = image_url('red')
    green = image_url('green')
    content = [
        {'type': 'text', 'text': 'before'},
        image_part(red),
        {'type': 'text', 'text': 'between'},
        image_part(green),
        {'type': 'text', 'text': 'after'},
    ]
    assert (
        client.post('/v1/memories/add', json=add_payload(content=content)).status_code
        == 200
    )
    result = client.post('/v1/memories/search', json=search_payload()).json()['data'][
        0
    ]['content']
    assert result[1:] == content
    assert not backend.caption_inputs
    assert not any('answer' in str(item).lower() for item in result)


def test_query_pixels_never_become_historical_evidence(client, backend):
    history = image_url('red')
    query = image_url('blue')
    client.post('/v1/memories/add', json=add_payload(content=[image_part(history)]))
    payload = search_payload(
        query=[{'type': 'text', 'text': 'Find this'}, image_part(query)]
    )
    result = client.post('/v1/memories/search', json=payload)
    assert result.status_code == 200
    assert history in result.text
    assert query not in result.text
    assert backend.caption_inputs[-1] == (query, '')
    count = len(backend.caption_inputs)
    client.post('/v1/memories/search', json=payload)
    assert len(backend.caption_inputs) == count


@pytest.mark.parametrize(
    'content',
    [
        [image_part('https://example.com/private.jpg')],
        [image_part('file:///etc/passwd')],
        [image_part('data:image/png;base64,!!!')],
        [image_part('data:image/png;base64,YWJj')],
        [image_part(image_url(image_format='GIF'))],
        [image_part(image_url().replace('image/png', 'image/jpeg'))],
        '',
        '   ',
        [],
        [{'type': 'text', 'text': '   '}],
    ],
)
def test_invalid_content_is_rejected_before_upstream_calls(client, backend, content):
    response = client.post('/v1/memories/add', json=add_payload(content=content))
    assert response.status_code == 422
    assert not backend.embedding_inputs
    assert not backend.caption_inputs


def test_invalid_timestamp_or_boolean_top_k_is_rejected(client):
    payload = add_payload()
    payload['messages'][0]['timestamp'] = 10**50
    assert client.post('/v1/memories/add', json=payload).status_code == 422
    assert (
        client.post(
            '/v1/memories/search', json={**search_payload(), 'top_k': True}
        ).status_code
        == 422
    )


def test_validation_errors_do_not_echo_private_payloads(client):
    response = client.post(
        '/v1/memories/add', json={**add_payload(), 'private-data': 'DO_NOT_ECHO_THIS'}
    )
    assert response.status_code == 422
    assert 'DO_NOT_ECHO_THIS' not in response.text


def test_zero_top_k_does_not_call_models(client, backend):
    client.post('/v1/memories/add', json=add_payload())
    count = len(backend.embedding_inputs)
    assert client.post(
        '/v1/memories/search', json={**search_payload(), 'top_k': 0}
    ).json() == {'data': []}
    assert len(backend.embedding_inputs) == count


def test_top_k_limits_response_count(client):
    client.post('/v1/memories/add', json=add_payload(content='red ' * 900))
    response = client.post('/v1/memories/search', json={**search_payload(), 'top_k': 1})
    assert response.status_code == 200
    assert len(response.json()['data']) == 1


def test_failed_write_is_not_acknowledged_or_searchable(client, backend):
    backend.fail_embedding = True
    assert client.post('/v1/memories/add', json=add_payload()).status_code == 502
    assert client.post('/v1/memories/search', json=search_payload()).json() == {
        'data': []
    }
    backend.fail_embedding = False
    assert client.post('/v1/memories/add', json=add_payload()).status_code == 200


def test_history_images_keep_provided_caption_and_add_independent_details(
    client, backend
):
    backend.fail_caption = True
    content = [
        {'type': 'text', 'text': 'Provided caption: a blue bicycle.'},
        image_part(),
    ]
    result = client.post('/v1/memories/add', json=add_payload(content=content))
    assert result.status_code == 200
    assert not backend.caption_inputs
    assert len(backend.detail_inputs) == 1
    assert any(
        'Provided caption: a blue bicycle.' in text for text in backend.embedding_inputs
    )
    assert any(
        'generated description]\nA red backpack.' in text
        for text in backend.embedding_inputs
    )
    source = client.app.state.engine.store.messages('user-1', 'session-1')[0]
    assert 'Provided caption: a blue bicycle.' in source['render']
    assert 'Blue straps' not in source['render']


def test_query_caption_failure_does_not_silently_drop_visual_retrieval(client, backend):
    assert client.post('/v1/memories/add', json=add_payload()).status_code == 200
    backend.fail_caption = True
    before = len(backend.embedding_inputs)
    response = client.post(
        '/v1/memories/search',
        json=search_payload(
            query=[{'type': 'text', 'text': 'Find this'}, image_part()]
        ),
    )
    assert response.status_code == 502
    assert response.json() == {'error': 'vision_http_403'}
    assert len(backend.embedding_inputs) == before
    assert not backend.rerank_inputs


def test_restart_preserves_memory_and_idempotence(settings):
    first = FakeProviders()
    with TestClient(create_app(settings, first)) as client:
        client.headers['Authorization'] = 'Token ' + settings.api_token
        client.post('/v1/memories/add', json=add_payload())
        original = client.post('/v1/memories/search', json=search_payload()).json()[
            'data'
        ]
    second = FakeProviders()
    with TestClient(create_app(settings, second)) as client:
        client.headers['Authorization'] = 'Token ' + settings.api_token
        assert client.post('/v1/memories/add', json=add_payload()).status_code == 200
        assert not second.embedding_inputs
        assert (
            client.post('/v1/memories/search', json=search_payload()).json()['data']
            == original
        )


def test_store_rejects_second_process_or_changed_configuration(settings):
    store = Store(settings.data_dir, settings.identity())
    with pytest.raises(ValueError, match='Another service'):
        Store(settings.data_dir, settings.identity())
    store.close()
    with pytest.raises(ValueError, match='different model'):
        Store(settings.data_dir, replace(settings, chunk_size=512).identity())


def test_local_purge_removes_all_user_artifacts_and_preserves_others(client):
    client.post('/v1/memories/add', json=add_payload(content=[image_part()]))
    client.post('/v1/memories/add', json=add_payload(user='other'))
    client.app.state.engine.store.purge_user('user-1')
    assert client.post('/v1/memories/search', json=search_payload()).json() == {
        'data': []
    }
    assert client.post('/v1/memories/search', json=search_payload(user='other')).json()[
        'data'
    ]


def test_image_and_request_budgets_are_explicit(settings):
    backend = FakeProviders()
    small = replace(settings, max_media_bytes=100)
    with TestClient(create_app(small, backend)) as client:
        client.headers['Authorization'] = 'Token ' + small.api_token
        result = client.post(
            '/v1/memories/add', json=add_payload(content=[image_part(), image_part()])
        )
        assert result.status_code == 413
        assert not backend.caption_inputs


def test_body_limit_and_auth_precedence(settings):
    small = replace(settings, max_body_bytes=100)
    with TestClient(create_app(small, FakeProviders())) as client:
        assert client.post('/v1/memories/add', content='x' * 101).status_code == 401
        client.headers['Authorization'] = 'Token ' + small.api_token
        assert client.post('/v1/memories/add', content='x' * 101).status_code == 413


@pytest.mark.parametrize(
    'field,value',
    [
        ('api_token', 'short'),
        ('vision_model', 'other-model'),
        ('embedding_model', 'other-model'),
        ('vision_url', 'https://secret@example.com/api'),
        ('vision_url', 'https://example.com/api?key=secret'),
    ],
)
def test_unsafe_or_nonacademic_configuration_fails(settings, field, value):
    with pytest.raises(ValueError):
        replace(settings, **{field: value})


def test_endpoint_accepts_base_or_complete_url():
    assert endpoint('https://host/v1', '/embeddings') == 'https://host/v1/embeddings'
    assert (
        endpoint('https://host/v1/embeddings', '/embeddings')
        == 'https://host/v1/embeddings'
    )


@pytest.mark.asyncio
async def test_embedding_provider_orders_and_normalizes_responses(settings):
    import httpx

    provider = Providers(settings)
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    'data': [
                        {'index': 1, 'embedding': [0, 2]},
                        {'index': 0, 'embedding': [3, 0]},
                    ]
                },
            )
        )
    )
    assert np.array_equal(await provider.embed(['first', 'second']), [[1, 0], [0, 1]])
    await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'rows',
    [
        [],
        [{'index': 0, 'embedding': [0, 0]}],
        [{'index': 0, 'embedding': [float('nan')]}],
    ],
)
async def test_embedding_provider_rejects_corrupt_results(settings, rows):
    provider = Providers(settings)

    async def response(*arguments, **kwargs):
        return {'data': rows}

    provider._post = response
    with pytest.raises(ServiceError, match='embedding_invalid_response'):
        await provider.embed(['test'])
    await provider.close()


@pytest.mark.asyncio
async def test_rerank_rejects_duplicate_positions(settings):
    provider = Providers(settings)

    async def response(*arguments, **kwargs):
        return {
            'results': [
                {'index': 0, 'relevance_score': 1},
                {'index': 0, 'relevance_score': 0.5},
            ]
        }

    provider._post = response
    with pytest.raises(ServiceError, match='rerank_invalid_response'):
        await provider.rerank(
            'query',
            [
                {'chunk_id': 'a', 'content': 'first document'},
                {'chunk_id': 'b', 'content': 'second document'},
            ],
            2,
        )
    await provider.close()


@pytest.mark.parametrize(
    'proxy_url',
    [
        'socks5://localhost:7897',
        'http://user:secret@localhost:7897',
        'http://localhost:bad',
        'http://localhost:7897/rerank',
    ],
)
def test_rerank_proxy_rejects_invalid_configuration(settings, proxy_url):
    with pytest.raises(ValueError, match='proxy'):
        replace(settings, rerank_proxy_url=proxy_url)


def test_rerank_proxy_does_not_change_index_identity(settings):
    proxied = replace(settings, rerank_proxy_url='http://127.0.0.1:47897')
    assert proxied.identity() == settings.identity()


@pytest.mark.asyncio
async def test_only_rerank_uses_configured_proxy(settings, monkeypatch):
    import httpx

    clients = []

    class RecordingClient:
        def __init__(self, **kwargs):
            self.configuration = kwargs
            self.requests = []
            self.closed = False
            clients.append(self)

        async def post(self, url, **kwargs):
            self.requests.append((url, kwargs))
            return httpx.Response(200, json={'ok': True})

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(httpx, 'AsyncClient', RecordingClient)
    provider = Providers(replace(settings, rerank_proxy_url='http://127.0.0.1:47897'))
    for name in ['vision', 'embedding', 'rerank']:
        await provider._post(name, 'https://upstream.example', 'test-key', {})
    assert len(clients) == 2
    assert 'proxy' not in clients[0].configuration
    assert clients[1].configuration['proxy'] == 'http://127.0.0.1:47897'
    assert len(clients[0].requests) == 2
    assert len(clients[1].requests) == 1
    await provider.close()
    assert all(client.closed for client in clients)


@pytest.mark.asyncio
async def test_no_explicit_proxy_reuses_default_client(settings):
    provider = Providers(settings)
    assert provider.rerank_client is provider.client
    await provider.close()


def test_official_document_examples_end_to_end(client, settings):
    from evaluation.src.aml.contract_smoke import run

    client.headers.pop('Authorization')
    report = {}
    run(client, settings.api_token, report, pause_seconds=0)
    assert report['passed']
    assert all(row['passed'] for row in report['checks'])


@pytest.mark.parametrize('top_k', [0, 100])
@pytest.mark.parametrize(
    'query', ['   ', [image_part('https://example.com/image.png')]]
)
def test_invalid_search_is_rejected_even_without_results(client, backend, top_k, query):
    response = client.post(
        '/v1/memories/search',
        json={'user_id': 'unknown-user', 'query': query, 'top_k': top_k},
    )
    assert response.status_code == 422
    assert not backend.embedding_inputs
    assert not backend.caption_inputs


@pytest.mark.parametrize('image_format', ['JPEG', 'PNG', 'WEBP'])
def test_supported_image_formats_preserve_original_bytes(client, image_format):
    content = [image_part(image_url(image_format=image_format))]
    assert (
        client.post('/v1/memories/add', json=add_payload(content=content)).status_code
        == 200
    )
    result = client.post('/v1/memories/search', json=search_payload()).json()['data'][
        0
    ]['content']
    assert result[1:] == content


@pytest.mark.parametrize('delta,status', [(0, 200), (-1, 413)])
def test_decoded_image_byte_limit_is_exact(settings, delta, status):
    url = image_url()
    size = len(base64.b64decode(url.split(',', 1)[1]))
    bounded = replace(settings, max_image_bytes=size + delta)
    backend = FakeProviders()
    with TestClient(create_app(bounded, backend)) as client:
        client.headers['Authorization'] = 'Token ' + settings.api_token
        response = client.post(
            '/v1/memories/add', json=add_payload(content=[image_part(url)])
        )
        assert response.status_code == status


def test_search_media_budget_preserves_complete_candidates(settings):
    url = image_url()
    size = len(base64.b64decode(url.split(',', 1)[1]))
    bounded = replace(settings, max_media_bytes=size)
    with TestClient(create_app(bounded, FakeProviders())) as client:
        client.headers['Authorization'] = 'Token ' + settings.api_token
        for session in ['first', 'second']:
            response = client.post(
                '/v1/memories/add',
                json=add_payload(
                    request=session, session=session, content=[image_part(url)]
                ),
            )
            assert response.status_code == 200
        response = client.post('/v1/memories/search', json=search_payload())
        assert response.status_code == 200
        assert len(response.json()['data']) == 1
        assert response.json()['data'][0]['content'][1:] == [image_part(url)]


@pytest.mark.parametrize(
    'source_type', ['email', 'image', 'video_text', 'video_frames']
)
def test_atm_sources_in_aml_envelope_preserve_evidence(client, backend, source_type):
    metadata = {
        'timestamp': '2024-01-01 00:00:00',
        'location_name': 'Paris',
        'short_caption': 'A red backpack at a station.',
        'caption': 'A red backpack beside a departure board.',
        'ocr_text': 'Platform 7, 09:45',
        'tags': ['backpack', 'station'],
    }
    if source_type == 'email':
        metadata = {
            'id': 'email202401010001',
            'timestamp': metadata['timestamp'],
            'short_summary': 'Paris train booking',
            'detail': 'Bring the red backpack. Departure: platform 7 at 09:45.',
        }
        content = json.dumps(metadata)
    else:
        metadata['image_path' if source_type == 'image' else 'video_path'] = (
            '20240101_000000.jpg' if source_type == 'image' else '20240101_000000.mp4'
        )
        content = [{'type': 'text', 'text': json.dumps(metadata)}]
        if source_type == 'image':
            content.append(image_part(image_url('red')))
        elif source_type == 'video_frames':
            content.extend(
                [
                    {'type': 'text', 'text': 'Frame timestamp: 00:00:01.000'},
                    image_part(image_url('red')),
                    {'type': 'text', 'text': 'Frame timestamp: 00:00:02.000'},
                    image_part(image_url('blue')),
                ]
            )
    payload = add_payload(session='atm-' + source_type, content=content)
    response = client.post('/v1/memories/add', json=payload)
    assert response.status_code == 200
    messages = client.app.state.engine.store.messages('user-1', payload['session_id'])
    assert len(messages) == 1
    assert all(
        str(value) in messages[0]['render']
        for value in metadata.values()
        if isinstance(value, str)
    )
    assert all(
        any(value in text for text in backend.embedding_inputs)
        for value in metadata.values()
        if isinstance(value, str)
    )
    response = client.post(
        '/v1/memories/search', json=search_payload(query='Paris station')
    )
    assert response.status_code == 200
    evidence = response.json()['data']
    assert evidence
    for row in evidence:
        assert row['created_at'] == '2024-01-01T00:00:00+00:00'
        if isinstance(content, str):
            assert row['content'].endswith(content)
        else:
            assert row['content'][1:] == content
    assert not backend.caption_inputs


def test_raw_video_is_rejected_without_model_calls(client, backend):
    content = [
        {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,YWJj'}}
    ]
    response = client.post('/v1/memories/add', json=add_payload(content=content))
    assert response.status_code == 422
    assert not backend.caption_inputs
    assert not backend.embedding_inputs
    assert not backend.rerank_inputs


def test_expanded_long_message_is_deduplicated_after_single_rerank(client, backend):
    content = 'The red camera was bought in Paris. ' * 600
    assert (
        client.post('/v1/memories/add', json=add_payload(content=content)).status_code
        == 200
    )
    response = client.post('/v1/memories/search', json=search_payload())
    assert response.status_code == 200
    assert len(backend.rerank_inputs) == 1
    assert len(backend.rerank_inputs[0][1]) > 1
    assert backend.rerank_limits == [len(backend.rerank_inputs[0][1])]
    evidence = response.json()['data']
    assert len(evidence) == 1
    assert evidence[0]['content'].endswith(content)
    message = client.app.state.engine.store.messages('user-1', 'session-1')[0]
    assert evidence[0]['id'] == message['message_id']


def test_partially_overlapping_chunks_expand_to_each_message_once(
    client, backend, monkeypatch
):
    payload = add_payload()
    payload['messages'] = [
        {'role': 'user', 'content': 'First complete source.'},
        {'role': 'assistant', 'content': 'Second complete source.'},
        {'role': 'user', 'content': 'Third complete source.'},
    ]
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    store = client.app.state.engine.store
    messages = store.messages('user-1', 'session-1')
    identifiers = [row['message_id'] for row in messages]
    candidates = [
        {
            'chunk_id': 'first',
            'content': 'First fragment',
            'source_ids': identifiers[:2],
        },
        {
            'chunk_id': 'second',
            'content': 'Second fragment',
            'source_ids': identifiers[1:],
        },
        {
            'chunk_id': 'third',
            'content': 'Third fragment',
            'source_ids': identifiers[:1],
        },
    ]
    monkeypatch.setattr(store, 'candidates', lambda *arguments: candidates)
    response = client.post('/v1/memories/search', json=search_payload())
    assert response.status_code == 200
    assert len(backend.rerank_inputs) == 1
    assert backend.rerank_limits == [len(candidates)]
    evidence = response.json()['data']
    assert [row['id'] for row in evidence] == identifiers
    assert evidence[0]['score'] == evidence[1]['score'] == 1
    assert evidence[2]['score'] == 1 - 1 / len(candidates)
    for original in payload['messages']:
        assert sum(row['content'].count(original['content']) for row in evidence) == 1


@pytest.mark.parametrize('sessions', [['first', 'second'], ['same', 'same']])
def test_equal_text_from_distinct_source_messages_is_not_deduplicated(
    client, backend, sessions
):
    content = 'I took a red backpack to Paris.'
    for position, session in enumerate(sessions):
        payload = add_payload(request=str(position), session=session, content=content)
        payload['messages'][0]['timestamp'] += position * 86400000
        assert client.post('/v1/memories/add', json=payload).status_code == 200
    response = client.post('/v1/memories/search', json=search_payload())
    assert response.status_code == 200
    evidence = response.json()['data']
    assert len(evidence) == 2
    assert len({row['id'] for row in evidence}) == 2
    assert len({row['created_at'] for row in evidence}) == 2
    assert all(row['content'].endswith(content) for row in evidence)
    assert len(backend.rerank_inputs) == 1


def test_source_id_stays_stable_after_session_rechunking(client):
    content = 'red camera in Paris ' * 300
    assert (
        client.post('/v1/memories/add', json=add_payload(content=content)).status_code
        == 200
    )
    first = client.post('/v1/memories/search', json=search_payload()).json()['data']
    assert len(first) == 1
    assert (
        client.post(
            '/v1/memories/add',
            json=add_payload(request='append', content='blue ' * 300),
        ).status_code
        == 200
    )
    second = client.post('/v1/memories/search', json=search_payload()).json()['data']
    original = [row for row in second if row['id'] == first[0]['id']]
    assert len(original) == 1
    assert original[0]['content'] == first[0]['content']


def test_single_chunk_rerank_determines_source_order_and_scores(
    client, backend, monkeypatch
):
    payload = add_payload()
    payload['messages'] = [
        {'role': 'user', 'content': 'First complete source.'},
        {'role': 'assistant', 'content': 'Second complete source.'},
    ]
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    store = client.app.state.engine.store
    messages = store.messages('user-1', 'session-1')
    monkeypatch.setattr(
        store,
        'candidates',
        lambda *arguments: [
            {
                'chunk_id': 'first-chunk',
                'content': 'First fragment.',
                'source_ids': [messages[0]['message_id']],
            },
            {
                'chunk_id': 'second-chunk',
                'content': 'Second fragment.',
                'source_ids': [messages[1]['message_id']],
            },
        ],
    )

    async def rerank(query, candidates, limit):
        backend.rerank_inputs.append((query, candidates))
        assert limit == 2
        assert [row['content'] for row in candidates] == [
            'First fragment.',
            'Second fragment.',
        ]
        return [{**candidates[1], 'score': 0.42}, {**candidates[0], 'score': 0.1}]

    backend.rerank = rerank
    response = client.post('/v1/memories/search', json={**search_payload(), 'top_k': 1})
    assert response.status_code == 200
    evidence = response.json()['data']
    assert len(evidence) == 1
    assert evidence[0]['id'] == messages[1]['message_id']
    assert evidence[0]['content'].endswith('Second complete source.')
    assert evidence[0]['score'] == 0.42
    assert len(backend.rerank_inputs) == 1


@pytest.mark.parametrize('top_k', [1, 3, 15, 100])
def test_message_dedup_backfills_beyond_fifteenth_ranked_chunk(
    client, backend, monkeypatch, top_k
):
    payload = add_payload()
    payload['messages'] = [
        {'role': 'user', 'content': f'Complete source number {position}.'}
        for position in range(15)
    ]
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    store = client.app.state.engine.store
    messages = store.messages('user-1', 'session-1')
    candidates = [
        {
            'chunk_id': f'repeat-{position}',
            'content': 'A fragment of the first source.',
            'source_ids': [messages[0]['message_id']],
        }
        for position in range(20)
    ] + [
        {
            'chunk_id': f'other-{position}',
            'content': 'A fragment of another source.',
            'source_ids': [message['message_id']],
        }
        for position, message in enumerate(messages[1:])
    ]
    monkeypatch.setattr(store, 'candidates', lambda *arguments: candidates)
    response = client.post(
        '/v1/memories/search', json={**search_payload(), 'top_k': top_k}
    )
    assert response.status_code == 200
    assert len(backend.rerank_inputs) == 1
    assert backend.rerank_limits == [34]
    evidence = response.json()['data']
    expected = min(top_k, 15)
    assert len(evidence) == expected
    assert [row['id'] for row in evidence] == [
        row['message_id'] for row in messages[:expected]
    ]
    assert evidence[0]['score'] == 1
    if expected > 1:
        assert evidence[1]['score'] == 1 - 20 / len(candidates)
    assert all(
        row['content'].endswith(original['content'])
        for row, original in zip(evidence, payload['messages'])
    )


def test_message_dedup_backfills_after_over_budget_source(settings, monkeypatch):
    backend = FakeProviders()
    url = image_url()
    size = len(base64.b64decode(url.split(',', 1)[1]))
    with TestClient(create_app(settings, backend)) as client:
        client.headers['Authorization'] = 'Token ' + settings.api_token
        assert (
            client.post(
                '/v1/memories/add',
                json=add_payload(content=[image_part(url), image_part(url)]),
            ).status_code
            == 200
        )
        assert (
            client.post(
                '/v1/memories/add',
                json=add_payload(
                    request='second',
                    session='second',
                    content='A complete text source.',
                ),
            ).status_code
            == 200
        )
        engine = client.app.state.engine
        first = engine.store.messages('user-1', 'session-1')[0]
        second = engine.store.messages('user-1', 'second')[0]
        monkeypatch.setattr(
            engine, 'settings', replace(settings, result_count=1, max_media_bytes=size)
        )
        monkeypatch.setattr(
            engine.store,
            'candidates',
            lambda *arguments: [
                {
                    'chunk_id': 'large',
                    'content': 'Image fragments.',
                    'source_ids': [first['message_id']],
                },
                {
                    'chunk_id': 'small',
                    'content': 'Text fragment.',
                    'source_ids': [second['message_id']],
                },
            ],
        )
        response = client.post('/v1/memories/search', json=search_payload())
        assert response.status_code == 200
        assert len(response.json()['data']) == 1
        assert response.json()['data'][0]['id'] == second['message_id']
        assert response.json()['data'][0]['content'].endswith('A complete text source.')
        assert backend.rerank_limits == [2]


def test_multimodal_message_dedup_preserves_all_original_parts(client, backend):
    parts = [
        {'type': 'text', 'text': 'red backpack in Paris ' * 600},
        image_part(),
        {'type': 'text', 'text': 'The same image appears again within this message.'},
        image_part(),
    ]
    assert (
        client.post('/v1/memories/add', json=add_payload(content=parts)).status_code
        == 200
    )
    response = client.post('/v1/memories/search', json=search_payload())
    assert response.status_code == 200
    assert len(backend.rerank_inputs) == 1
    assert backend.rerank_limits == [len(backend.rerank_inputs[0][1])]
    evidence = response.json()['data']
    assert len(evidence) == 1
    assert evidence[0]['content'][1:] == parts


def test_single_rerank_failure_does_not_return_unranked_or_duplicate_sources(
    client, backend
):
    assert client.post('/v1/memories/add', json=add_payload()).status_code == 200

    async def rerank(query, candidates, limit):
        backend.rerank_inputs.append((query, candidates))
        raise ServiceError(502, 'rerank_http_429')

    backend.rerank = rerank
    response = client.post('/v1/memories/search', json=search_payload())
    assert response.status_code == 502
    assert response.json() == {'error': 'rerank_http_429'}
    assert len(backend.rerank_inputs) == 1


def test_supplied_captions_stay_next_to_their_own_images(client, backend):
    parts = [
        {'type': 'text', 'text': 'Caption one: a red backpack.'},
        image_part(image_url('red')),
        {'type': 'text', 'text': 'Caption two: a blue bicycle.'},
        image_part(image_url('blue')),
    ]
    assert (
        client.post('/v1/memories/add', json=add_payload(content=parts)).status_code
        == 200
    )
    assert not backend.caption_inputs
    engine = client.app.state.engine
    message = engine.store.messages('user-1', 'session-1')[0]
    content = json.loads(message['content'])
    red_marker = f"[Image img_{content[1]['hash'][:16]}]"
    blue_marker = f"[Image img_{content[3]['hash'][:16]}]"
    rendered = message['render']
    assert rendered.index(parts[0]['text']) < rendered.index(red_marker)
    assert rendered.index(red_marker) < rendered.index(parts[2]['text'])
    assert rendered.index(parts[2]['text']) < rendered.index(blue_marker)
    response = client.post(
        '/v1/memories/search', json=search_payload(query='blue bicycle')
    )
    assert response.status_code == 200
    assert response.json()['data'][0]['content'][1:] == parts
    assert not backend.caption_inputs


def test_query_and_image_caption_retrieve_separately_then_rerank_once(
    client, backend, monkeypatch
):
    payload = add_payload()
    payload['messages'] = [
        {'role': 'user', 'content': f'Source {position}.'} for position in range(3)
    ]
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    store = client.app.state.engine.store
    messages = store.messages('user-1', 'session-1')
    candidates = [
        {
            'chunk_id': f'chunk-{position}',
            'score': 0.9 - position / 10,
            'content': message['render'],
            'source_ids': [message['message_id']],
        }
        for position, message in enumerate(messages)
    ]
    vectors = []

    def retrieve(user, vector, count):
        assert user == 'user-1' and count == 128
        vectors.append(vector)
        if vector[2] > vector[1]:
            return [candidates[0], candidates[2]]
        return [candidates[1], candidates[2]]

    monkeypatch.setattr(store, 'candidates', retrieve)
    before = len(backend.embedding_inputs)
    question = 'Where was the blue item bought?'
    response = client.post(
        '/v1/memories/search',
        json=search_payload(query=[{'type': 'text', 'text': question}, image_part()]),
    )
    assert response.status_code == 200
    assert len(vectors) == 2
    assert backend.embedding_inputs[before:] == [question, 'A red backpack.']
    assert len(backend.caption_inputs) == 1
    assert len(backend.rerank_inputs) == 1
    rerank_query, merged = backend.rerank_inputs[0]
    assert question in rerank_query and 'A red backpack.' in rerank_query
    assert [row['chunk_id'] for row in merged] == ['chunk-2', 'chunk-0', 'chunk-1']
    assert backend.rerank_limits == [3]
    assert [row['id'] for row in response.json()['data']] == [
        messages[position]['message_id'] for position in [2, 0, 1]
    ]


def test_image_only_query_uses_caption_without_an_empty_text_branch(client, backend):
    assert client.post('/v1/memories/add', json=add_payload()).status_code == 200
    before = len(backend.embedding_inputs)
    response = client.post(
        '/v1/memories/search', json=search_payload(query=[image_part()])
    )
    assert response.status_code == 200
    assert backend.embedding_inputs[before:] == ['A red backpack.']
    assert len(backend.rerank_inputs) == 1


def test_multiple_query_images_share_one_caption_branch(client, backend, monkeypatch):
    assert client.post('/v1/memories/add', json=add_payload()).status_code == 200
    captions = iter(['A red backpack.', 'A blue bicycle.'])

    async def caption(url, context):
        backend.caption_inputs.append((url, context))
        return next(captions)

    backend.caption = caption
    engine = client.app.state.engine
    retrieve = engine.store.candidates
    vectors = []

    def search(user, vector, count):
        vectors.append(vector)
        return retrieve(user, vector, count)

    monkeypatch.setattr(engine.store, 'candidates', search)
    before = len(backend.embedding_inputs)
    response = client.post(
        '/v1/memories/search',
        json=search_payload(
            query=[
                {'type': 'text', 'text': 'Compare these items'},
                image_part(image_url('red')),
                image_part(image_url('blue')),
            ]
        ),
    )
    assert response.status_code == 200
    assert len(backend.caption_inputs) == 2
    assert len(vectors) == 2
    assert backend.embedding_inputs[before:] == [
        'Compare these items',
        'A red backpack.\nA blue bicycle.',
    ]
    assert len(backend.rerank_inputs) == 1


def test_fusion_deduplicates_chunk_ids_and_caps_the_shared_pool(client):
    engine = client.app.state.engine
    first = [{'chunk_id': f'text-{position}'} for position in range(128)]
    second = [{'chunk_id': f'caption-{position}'} for position in range(128)]
    fused = engine._fuse_candidates([first, second])
    assert len(fused) == 128
    assert len({row['chunk_id'] for row in fused}) == 128
    assert {row['chunk_id'] for row in fused[:2]} == {'text-0', 'caption-0'}
    overlap = engine._fuse_candidates(
        [[{'chunk_id': 'same'}, {'chunk_id': 'same'}], [{'chunk_id': 'other'}]]
    )
    assert [row['chunk_id'] for row in overlap] == ['same', 'other']
    assert overlap[0]['score'] == pytest.approx(1 / 61)


def test_empty_query_caption_fails_without_retrieval(client, backend):
    assert client.post('/v1/memories/add', json=add_payload()).status_code == 200
    original = backend.caption

    async def caption(url, context):
        return '   '

    backend.caption = caption
    before = len(backend.embedding_inputs)
    response = client.post(
        '/v1/memories/search', json=search_payload(query=[image_part()])
    )
    assert response.status_code == 502
    assert response.json() == {'error': 'empty_query_image_caption'}
    assert len(backend.embedding_inputs) == before
    assert not backend.rerank_inputs
    backend.caption = original
    response = client.post(
        '/v1/memories/search', json=search_payload(query=[image_part()])
    )
    assert response.status_code == 200
    assert len(backend.rerank_inputs) == 1


def test_image_views_keep_exact_picture_and_message_binding(client, backend):
    red, blue = image_url('red'), image_url('blue')
    content = [
        {'type': 'text', 'text': 'First provided caption.'},
        image_part(red),
        {'type': 'text', 'text': 'Second provided caption.'},
        image_part(blue),
    ]
    payload = add_payload(content=content)
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    store = client.app.state.engine.store
    source = store.messages('user-1', 'session-1')[0]
    parts = json.loads(source['content'])
    with store.connection() as connection:
        rows = [dict(row) for row in connection.execute('SELECT * FROM image_views')]
    assert len(rows) == 6
    assert {row['message_id'] for row in rows} == {source['message_id']}
    provided = {
        row['image_hash']: row['content'] for row in rows if row['kind'] == 'provided'
    }
    assert 'First provided caption.' in provided[parts[1]['hash']]
    assert 'Second provided caption.' not in provided[parts[1]['hash']]
    assert 'Second provided caption.' in provided[parts[3]['hash']]
    assert 'First provided caption.' not in provided[parts[3]['hash']]
    assert backend.detail_inputs == [red, blue]
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    assert backend.detail_inputs == [red, blue]
    result = client.post('/v1/memories/search', json=search_payload()).json()['data']
    assert len(result) == 1
    assert [
        part['image_url']['url']
        for part in result[0]['content']
        if part['type'] == 'image_url'
    ] == [red, blue]
    assert not any(
        'Blue straps' in part.get('text', '') for part in result[0]['content']
    )


def test_each_image_view_can_retrieve_but_sibling_views_have_one_candidate(
    client, backend
):
    async def details(url):
        return ['Red straps.', 'Blue buckles.', 'Paris is readable on the label.']

    backend.image_details = details
    content = [{'type': 'text', 'text': 'An original travel photograph.'}, image_part()]
    assert (
        client.post('/v1/memories/add', json=add_payload(content=content)).status_code
        == 200
    )
    engine = client.app.state.engine
    for query in ['red', 'blue', 'paris']:
        vector = np.asarray(
            [1, int(query == 'red'), int(query == 'blue'), int(query == 'paris')],
            dtype=np.float32,
        )
        vector /= np.linalg.norm(vector)
        candidates = engine.store.candidates('user-1', vector, 128)
        image_candidates = [row for row in candidates if row.get('image_hash')]
        assert len(image_candidates) == 1
        assert query in image_candidates[0]['content'].lower()
    vector = np.asarray([1, 0, 0, 0], dtype=np.float32)
    candidates = engine.store.candidates('user-1', vector, 128)
    assert (
        'original travel photograph'
        in next(row for row in candidates if row.get('image_hash'))['content']
    )
    assert engine.store.candidates('other-user', vector, 128) == []
    before = len(backend.rerank_inputs)
    reply = client.post(
        '/v1/memories/search', json=search_payload(query=[image_part()])
    )
    assert reply.status_code == 200
    assert len(reply.json()['data']) == 1
    assert len(backend.rerank_inputs) - before == 1


def test_image_fusion_preserves_two_matched_views_without_extra_rrf_votes(client):
    engine = client.app.state.engine
    first = {
        'chunk_id': 'image-group',
        'image_hash': 'pixels',
        'content': 'red straps',
        'source_ids': ['source'],
    }
    second = {**first, 'content': 'blue buckles'}
    fused = engine._fuse_candidates([[first, first], [second]])
    assert len(fused) == 1
    assert fused[0]['score'] == pytest.approx(2 / 61)
    assert 'red straps' in fused[0]['content'] and 'blue buckles' in fused[0]['content']
    assert first['content'] == 'red straps'


def test_history_details_failure_does_not_commit_partial_memory(client, backend):
    backend.fail_details = True
    payload = add_payload(content=[image_part()])
    assert client.post('/v1/memories/add', json=payload).status_code == 502
    store = client.app.state.engine.store
    with store.connection() as connection:
        for table in ['messages', 'requests', 'media', 'chunks', 'image_views']:
            assert (
                connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == 0
            )
    backend.fail_details = False
    backend.fail_embedding = True
    assert client.post('/v1/memories/add', json=payload).status_code == 502
    assert store.messages('user-1', 'session-1') == []
    backend.fail_embedding = False
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    assert len(backend.detail_inputs) == 2


def test_history_details_cache_survives_append_and_restart_and_is_user_scoped(settings):
    first = FakeProviders()
    payload = add_payload(content=[image_part()])
    with TestClient(create_app(settings, first)) as instance:
        instance.headers['Authorization'] = 'Token ' + settings.api_token
        assert instance.post('/v1/memories/add', json=payload).status_code == 200
        assert (
            instance.post(
                '/v1/memories/add', json=add_payload(request='second')
            ).status_code
            == 200
        )
        with instance.app.state.engine.store.connection() as connection:
            assert (
                connection.execute('SELECT count(*) FROM image_views').fetchone()[0]
                == 2
            )
    second = FakeProviders()
    with TestClient(create_app(settings, second)) as instance:
        instance.headers['Authorization'] = 'Token ' + settings.api_token
        assert (
            instance.post(
                '/v1/memories/add', json={**payload, 'request_id': 'third'}
            ).status_code
            == 200
        )
        assert not second.detail_inputs
        assert (
            instance.post(
                '/v1/memories/add', json={**payload, 'user_id': 'other'}
            ).status_code
            == 200
        )
        assert len(second.detail_inputs) == 1
        store = instance.app.state.engine.store
        store.purge_user('user-1')
        with store.connection() as connection:
            rows = connection.execute('SELECT user_id FROM image_views').fetchall()
            assert len(rows) == 2 and {row['user_id'] for row in rows} == {'other'}


@pytest.mark.parametrize(
    'descriptions', [[], ['A red backpack.', ' A  RED backpack. ']]
)
def test_empty_or_duplicate_generated_descriptions_are_bounded(
    client, backend, descriptions
):
    async def details(url):
        return descriptions

    backend.image_details = details
    payload = add_payload(
        content=[{'type': 'text', 'text': 'Supplied caption.'}, image_part()]
    )
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    with client.app.state.engine.store.connection() as connection:
        rows = connection.execute('SELECT kind FROM image_views').fetchall()
    assert len(rows) == (1 if not descriptions else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'value,finish,valid',
    [
        ({'descriptions': ['Red straps.', 'Blue buckles.']}, 'stop', True),
        ({'descriptions': []}, 'stop', True),
        ({'descriptions': ['same', ' SAME ']}, 'stop', True),
        ({'descriptions': ['text']}, 'length', False),
        ({'descriptions': ['text']}, 'content_filter', False),
        ({}, 'stop', False),
        ({'descriptions': 'not a list'}, 'stop', False),
        ({'descriptions': [None]}, 'stop', False),
        ({'descriptions': ['']}, 'stop', False),
        ({'descriptions': ['x' * 1201]}, 'stop', False),
        ({'descriptions': ['one', 'two', 'three', 'four', 'five']}, 'stop', False),
        ('not json', 'stop', False),
    ],
)
async def test_image_details_provider_validates_grounded_multi_view_json(
    settings, value, finish, valid
):
    provider = Providers(settings)
    recorded = []

    async def response(name, url, key, payload):
        recorded.append(payload)
        return {
            'choices': [
                {
                    'finish_reason': finish,
                    'message': {
                        'content': (
                            json.dumps(value) if isinstance(value, dict) else value
                        )
                    },
                }
            ]
        }

    provider._post = response
    try:
        if valid:
            result = await provider.image_details(image_url())
            assert len(result) <= 4
            assert len(result) == len(set(text.casefold() for text in result))
        else:
            with pytest.raises(ServiceError, match='vision_invalid_image_details'):
                await provider.image_details(image_url())
        assert len(recorded) == 1
        assert recorded[0]['model'] == 'gpt-4o-mini'
        assert recorded[0]['response_format'] == {'type': 'json_object'}
        assert recorded[0]['messages'][1]['content'][1]['image_url']['detail'] == 'high'
    finally:
        await provider.close()


def test_new_caption_protocol_rejects_a_legacy_data_directory(settings):
    store = Store(settings.data_dir, 'legacy-v1.1-generated-history-captions')
    store.close()
    with pytest.raises(ValueError, match='different model or configuration'):
        Store(settings.data_dir, settings.identity())


@pytest.mark.parametrize('concurrency,request_count', [(1, 4), (2, 16)])
def test_over_capacity_queues_until_writes_complete(
    settings, concurrency, request_count
):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    backend = FakeProviders()
    original = backend.embed
    active = 0
    peak = 0

    async def delayed(texts):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.1)
            return await original(texts)
        finally:
            active -= 1

    backend.embed = delayed
    barrier = Barrier(request_count)
    application = create_app(replace(settings, concurrency=concurrency), backend)
    with TestClient(application) as client:
        client.headers['Authorization'] = 'Token ' + settings.api_token

        def send(position):
            barrier.wait(timeout=5)
            return client.post('/v1/memories/add', json=add_payload(user=str(position)))

        with ThreadPoolExecutor(max_workers=request_count) as executor:
            responses = list(executor.map(send, range(request_count)))
        assert [response.status_code for response in responses] == [200] * request_count
        assert all(response.json()['success'] for response in responses)
        assert peak == concurrency
        assert active == 0
        with client.app.state.engine.store.connection() as connection:
            assert (
                connection.execute('SELECT count(*) FROM requests').fetchone()[0]
                == request_count
            )
