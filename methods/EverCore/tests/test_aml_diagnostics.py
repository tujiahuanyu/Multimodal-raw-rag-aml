import json
from types import SimpleNamespace

import httpx
import pytest

from evaluation.src.aml import diagnostics
from evaluation.src.aml.providers import Providers, ServiceError


def completion(content, reason='stop', refusal=None):
    return {
        'id': 'chatcmpl-diagnostic-test',
        'choices': [
            {'finish_reason': reason, 'message': {'content': content, 'refusal': refusal}}
        ],
        'usage': {'prompt_tokens': 8647, 'completion_tokens': 150, 'total_tokens': 8797},
    }


def records(caplog):
    return [
        json.loads(record.message.removeprefix('AML_DIAGNOSTIC '))
        for record in caplog.records
        if record.message.startswith('AML_DIAGNOSTIC ')
    ]


def provider_settings():
    return SimpleNamespace(
        request_timeout=120,
        rerank_proxy_url='',
        api_token='PRIVATEKEY',
        vision_url='https://vision.example/v1/chat/completions',
        vision_key='PRIVATEPROVIDERKEY',
        vision_model='gpt-4o-mini',
    )


@pytest.mark.parametrize(
    'data,expected',
    [
        (completion('{"descriptions": ["synthetic"]}', 'length'), 'finish_reason_not_stop'),
        (completion(None, 'content_filter'), 'finish_reason_not_stop'),
        (completion(None, refusal='private refusal text'), 'content_not_string'),
        (completion('private non-JSON text'), 'content_json_parse_failed'),
        (completion('[]'), 'content_json_not_object'),
        (completion('{"private field": "private text"}'), 'descriptions_missing'),
        (completion('{"descriptions": "private text"}'), 'descriptions_not_array'),
        (completion(json.dumps({'descriptions': ['private'] * 5})), 'too_many_descriptions'),
        (completion('{"descriptions": [null]}'), 'description_not_string'),
        (completion('{"descriptions": [" "]}'), 'description_empty'),
        (completion(json.dumps({'descriptions': ['x' * 1201]})), 'description_too_long'),
        (completion('{"descriptions": []}'), 'valid'),
    ],
)
def test_diagnostic_reasons_and_privacy(data, expected, caplog):
    diagnostics.image_details_failure(data, 'data:image/png;base64,PRIVATEIMAGE', 'PRIVATEKEY')
    result = records(caplog)[-1]
    assert result['validation_reason'] == expected
    assert result['prompt_tokens'] == 8647
    assert result['completion_tokens'] == 150
    assert len(result['image_fingerprint']) == 24
    for secret in ('private refusal text', 'private non-JSON text', 'private field', 'private text', 'PRIVATEIMAGE', 'PRIVATEKEY', 'x' * 1201):
        assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['stop', 'length', 'content_filter'])
async def test_provider_payload_calls_and_errors_unchanged(reason, caplog):
    settings = provider_settings()
    provider = Providers(settings)
    await provider.client.aclose()
    sent = []
    data = completion('{"descriptions": ["PRIVATECAPTION"]}', reason)

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=data, headers={'x-request-id': 'req-safe-123'})

    provider.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        with diagnostics.add_trace('PRIVATEUSER', 'PRIVATEREQUEST', 'PRIVATEKEY'):
            if reason == 'stop':
                assert await provider.image_details('data:image/png;base64,PRIVATEIMAGE') == ['PRIVATECAPTION']
            else:
                with pytest.raises(ServiceError) as caught:
                    await provider.image_details('data:image/png;base64,PRIVATEIMAGE')
                assert caught.value.status == 502
                assert caught.value.code == 'vision_invalid_image_details'
    finally:
        await provider.close()
    assert len(sent) == 1
    assert provider.calls == {'vision': 1, 'embedding': 0, 'rerank': 0}
    assert sent[0]['model'] == 'gpt-4o-mini'
    assert sent[0]['max_tokens'] == 768
    assert sent[0]['temperature'] == 0
    assert sent[0]['response_format'] == {'type': 'json_object'}
    assert sent[0]['messages'][1]['content'][1]['image_url']['detail'] == 'high'
    event = next(row for row in records(caplog) if row['event'] == 'vision_response')
    assert event['http_status'] == 200
    assert event['finish_reason'] == reason
    assert event['provider_request_id'] == 'req-safe-123'
    for secret in ('PRIVATECAPTION', 'PRIVATEIMAGE', 'PRIVATEKEY', 'PRIVATEPROVIDERKEY', 'PRIVATEUSER', 'PRIVATEREQUEST'):
        assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'mode,expected_code,expected_calls',
    [
        ('http_429', 'vision_http_429', 2),
        ('http_503', 'vision_http_503', 2),
        ('network', 'vision_connection_failed', 2),
        ('bad_json', 'vision_invalid_json', 1),
        ('bad_descriptions', 'vision_invalid_image_details', 1),
    ],
)
async def test_existing_retry_and_validation_rules_are_unchanged(
    mode, expected_code, expected_calls, monkeypatch, caplog
):
    provider = Providers(provider_settings())
    await provider.client.aclose()
    sent = []
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    def respond(request):
        sent.append(request)
        if mode == 'network':
            raise httpx.ConnectError('PRIVATENETWORKERROR', request=request)
        if mode.startswith('http_'):
            return httpx.Response(
                int(mode.removeprefix('http_')),
                json={'error': {'message': 'PRIVATEUPSTREAMERROR'}},
            )
        if mode == 'bad_json':
            return httpx.Response(200, text='PRIVATEBADJSON')
        return httpx.Response(
            200, json=completion(json.dumps({'descriptions': ['PRIVATE', None]}))
        )

    monkeypatch.setattr('evaluation.src.aml.providers.asyncio.sleep', fake_sleep)
    provider.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(ServiceError) as caught:
            await provider.image_details('data:image/png;base64,PRIVATEIMAGE')
    finally:
        await provider.close()
    assert caught.value.status == 502
    assert caught.value.code == expected_code
    assert len(sent) == expected_calls
    assert sleeps == ([1] if expected_calls == 2 else [])
    for secret in ('PRIVATENETWORKERROR', 'PRIVATEUPSTREAMERROR', 'PRIVATEBADJSON', 'PRIVATEIMAGE'):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_extra_descriptions_are_deduplicated_and_capped_without_retry(caplog):
    provider = Providers(provider_settings())
    await provider.client.aclose()
    sent = []
    descriptions = [
        'PRIVATE first', ' PRIVATE  FIRST ', 'PRIVATE second',
        'PRIVATE SECOND', 'PRIVATE third', 'PRIVATE fourth',
        'PRIVATE fifth', 'PRIVATE sixth', 'PRIVATE seventh',
    ]

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, json=completion(json.dumps({'descriptions': descriptions}))
        )

    provider.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        with diagnostics.add_trace('PRIVATEUSER', 'PRIVATEREQUEST', 'PRIVATEKEY'):
            assert await provider.image_details('data:image/png;base64,PRIVATEIMAGE') == [
                'PRIVATE first', 'PRIVATE second', 'PRIVATE third', 'PRIVATE fourth',
            ]
    finally:
        await provider.close()
    assert len(sent) == 1
    assert provider.calls == {'vision': 1, 'embedding': 0, 'rerank': 0}
    assert sent[0]['max_tokens'] == 768
    assert sent[0]['temperature'] == 0
    assert sent[0]['response_format'] == {'type': 'json_object'}
    assert sent[0]['messages'][1]['content'][1]['image_url']['detail'] == 'high'
    events = records(caplog)
    assert events[-1]['event'] == 'add_completed'
    assert not any(row['event'] == 'image_details_invalid' for row in events)
    assert next(row for row in events if row['event'] == 'vision_response')['description_count'] == 9
    for secret in descriptions + ['PRIVATEIMAGE', 'PRIVATEKEY']:
        assert secret not in caplog.text


def test_request_fingerprints_are_stable_isolated_and_context_resets(caplog):
    with diagnostics.add_trace('user-a', 'request-a', 'secret-a'):
        diagnostics.image_details_failure({}, 'synthetic-image', 'secret-a')
    with diagnostics.add_trace('user-a', 'request-a', 'secret-a'):
        pass
    with diagnostics.add_trace('user-b', 'request-a', 'secret-a'):
        pass
    starts = [row['add_fingerprint'] for row in records(caplog) if row['event'] == 'add_started']
    assert starts[0] == starts[1]
    assert starts[0] != starts[2]
    diagnostics.image_details_failure({}, 'synthetic-image', 'secret-a')
    assert 'add_fingerprint' not in records(caplog)[-1]


def test_diagnostics_do_not_replace_errors_when_logging_fails(monkeypatch):
    def broken_logger(*args, **kwargs):
        raise RuntimeError('logger unavailable')

    monkeypatch.setattr(diagnostics._logger, 'warning', broken_logger)
    original = ServiceError(502, 'vision_invalid_image_details')
    with pytest.raises(ServiceError) as caught:
        with diagnostics.add_trace('user-a', 'request-a', 'secret-a'):
            diagnostics.image_details_failure({}, 'synthetic-image', 'secret-a')
            raise original
    assert caught.value is original
