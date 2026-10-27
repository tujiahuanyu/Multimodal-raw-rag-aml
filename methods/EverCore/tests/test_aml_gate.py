import asyncio
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

from evaluation.src.aml.app import Gate


TOKEN = 'test-only-queue-token-' * 2


def make_gate(application, concurrency=1):
    return Gate(
        application,
        SimpleNamespace(
            concurrency=concurrency, api_token=TOKEN, max_body_bytes=1024,
        ),
    )


async def request(
    gate, path='/v1/memories/add', token=TOKEN, reads=None, disconnected=False
):
    messages = []

    async def receive():
        if reads is not None:
            reads.append(path)
        return (
            {'type': 'http.disconnect'}
            if disconnected
            else {'type': 'http.request', 'body': b'{}', 'more_body': False}
        )

    async def send(message):
        messages.append(message)

    await gate(
        {
            'type': 'http', 'path': path,
            'headers': [(b'authorization', ('Token ' + token).encode())],
        },
        receive,
        send,
    )
    return [
        message['status'] for message in messages
        if message['type'] == 'http.response.start'
    ]


@pytest.mark.asyncio
async def test_waiting_add_and_search_share_capacity_without_reading_queued_bodies():
    started = asyncio.Event()
    release = asyncio.Event()
    entered = []
    reads = []

    async def application(scope, receive, send):
        if scope['path'] != '/health':
            entered.append(scope['path'])
            started.set()
            await release.wait()
        await JSONResponse({'ok': True})(scope, receive, send)

    gate = make_gate(application)
    first = asyncio.create_task(request(gate, reads=reads))
    await asyncio.wait_for(started.wait(), 2)
    second = asyncio.create_task(request(gate, '/v1/memories/search', reads=reads))
    try:
        await asyncio.sleep(0.1)
        assert not second.done()
        assert entered == ['/v1/memories/add']
        assert reads == ['/v1/memories/add']
        assert (
            await asyncio.wait_for(request(gate, '/health', token=''), 2) == [200]
        )
        assert await asyncio.wait_for(request(gate, token='invalid'), 2) == [401]
    finally:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(first, second), 2)
    assert results == [[200], [200]]
    assert entered == ['/v1/memories/add', '/v1/memories/search']
    assert reads == entered


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_consume_capacity():
    started = asyncio.Event()
    release = asyncio.Event()

    async def application(scope, receive, send):
        started.set()
        await release.wait()
        await JSONResponse({'ok': True})(scope, receive, send)

    gate = make_gate(application)
    first = asyncio.create_task(request(gate))
    await asyncio.wait_for(started.wait(), 2)
    waiting = asyncio.create_task(request(gate))
    await asyncio.sleep(0.1)
    waiting.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        release.set()
        assert await asyncio.wait_for(first, 2) == [200]
    assert await asyncio.wait_for(request(gate), 2) == [200]


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['exception', 'cancellation'])
async def test_failed_active_request_releases_capacity(failure):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def application(scope, receive, send):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            raise RuntimeError('test failure')
        await JSONResponse({'ok': True})(scope, receive, send)

    gate = make_gate(application)
    first = asyncio.create_task(request(gate))
    await asyncio.wait_for(started.wait(), 2)
    second = asyncio.create_task(request(gate))
    await asyncio.sleep(0.1)
    if failure == 'cancellation':
        first.cancel()
        expected = asyncio.CancelledError
    else:
        release.set()
        expected = RuntimeError
    with pytest.raises(expected):
        await first
    assert await asyncio.wait_for(second, 2) == [200]
    assert await asyncio.wait_for(request(gate), 2) == [200]


@pytest.mark.asyncio
async def test_disconnected_queued_request_skips_application_and_releases_capacity():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def application(scope, receive, send):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        await JSONResponse({'ok': True})(scope, receive, send)

    gate = make_gate(application)
    first = asyncio.create_task(request(gate))
    await asyncio.wait_for(started.wait(), 2)
    disconnected = asyncio.create_task(request(gate, disconnected=True))
    await asyncio.sleep(0.1)
    release.set()
    assert (
        await asyncio.wait_for(asyncio.gather(first, disconnected), 2)
        == [[200], []]
    )
    assert calls == 1
    assert await asyncio.wait_for(request(gate), 2) == [200]
