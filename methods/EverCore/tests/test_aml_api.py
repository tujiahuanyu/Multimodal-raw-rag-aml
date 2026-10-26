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
        self.rerank_inputs = []
        self.fail_embedding = False
        self.fail_caption = False

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

    async def rerank(self, query, candidates, limit):
        self.rerank_inputs.append((query, candidates))
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
    assert (
        'blue'
        in client.post('/v1/memories/search', json=search_payload(query='blue')).json()[
            'data'
        ][0]['content']
    )


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
    assert len(backend.caption_inputs) == 2
    assert backend.caption_inputs[0][1] == 'before\nbetween\nafter'
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


def test_caption_failure_is_not_silently_replaced_by_text_only(client, backend):
    backend.fail_caption = True
    result = client.post('/v1/memories/add', json=add_payload(content=[image_part()]))
    assert result.status_code == 502
    assert not backend.embedding_inputs


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


def test_over_capacity_returns_retry_after(settings):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    backend = FakeProviders()
    original = backend.embed

    async def delayed(texts):
        await asyncio.sleep(0.3)
        return await original(texts)

    backend.embed = delayed
    barrier = Barrier(3)
    with TestClient(create_app(settings, backend)) as client:
        client.headers['Authorization'] = 'Token ' + settings.api_token

        def send(position):
            barrier.wait(timeout=5)
            return client.post('/v1/memories/add', json=add_payload(user=str(position)))

        with ThreadPoolExecutor(max_workers=3) as executor:
            responses = list(executor.map(send, range(3)))
        assert sorted(response.status_code for response in responses) == [200, 200, 429]
        assert (
            next(
                response for response in responses if response.status_code == 429
            ).headers['Retry-After']
            == '1'
        )
