import argparse
import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evaluation.src.core.data_models import Conversation, Dataset, Message
from evaluation.src.multimodal import runner, full_run
from evaluation.src.multimodal.raw_backend import MultimodalRAGAdapter


@pytest.mark.asyncio
async def test_caption_workers_checkpoint_and_resume(tmp_path, monkeypatch):
    images = [
        {'caption_key': str(position), 'path': 'unused.png'} for position in range(4)
    ]
    dataset = Dataset(
        'memgallery',
        [
            Conversation(
                'source',
                [Message('user', 'User', 'context', metadata={'images': images})],
            )
        ],
        [],
    )
    monkeypatch.setattr(runner, 'load_memgallery', lambda *args, **kwargs: dataset)
    monkeypatch.setattr(runner, '_load_config', lambda path: {})
    client = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(
        runner, '_vision_client', lambda config: (client, {'model': 'test-only'})
    )
    active = 0
    peak = 0
    calls = 0

    async def describe(*args, **kwargs):
        nonlocal active, peak, calls
        active += 1
        calls += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return 'generated description'

    monkeypatch.setattr(runner, 'describe_image', describe)
    arguments = argparse.Namespace(
        data_root=tmp_path,
        scenario=None,
        config=None,
        captions=tmp_path / 'captions.json',
        limit_images=None,
        caption_workers=2,
    )
    await runner.generate_captions(arguments)
    await runner.generate_captions(arguments)
    assert peak == 2
    assert calls == 4
    assert len(json.loads(arguments.captions.read_text())) == 4


def test_multimodal_proxy_is_only_used_for_rerank(monkeypatch):
    adapter = object.__new__(MultimodalRAGAdapter)
    adapter.config = {'rerank': {'proxy_url': 'http://127.0.0.1:47897'}}
    adapter.rerank_api_key = 'test-only'
    requests = []
    proxies = []

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            return io.BytesIO(b'{"results": []}')

    def opener(handler):
        proxies.append(handler.proxies)
        return Opener()

    monkeypatch.setattr(
        'evaluation.src.multimodal.raw_backend.urllib.request.build_opener', opener
    )
    result = adapter._post_json(
        'https://api.cohere.com/v2/rerank', {'model': 'rerank-v3.5'}, 10
    )
    assert result == {'results': []}
    assert proxies == [
        {'http': 'http://127.0.0.1:47897', 'https': 'http://127.0.0.1:47897'}
    ]
    assert requests[0].full_url == 'https://api.cohere.com/v2/rerank'


@pytest.mark.asyncio
async def test_full_evaluation_requires_quota_confirmation(tmp_path):
    arguments = argparse.Namespace(
        env_file=tmp_path / 'missing.env',
        output=tmp_path / 'output',
        stage='evaluate',
        data_root=tmp_path,
        config=None,
        caption_workers=1,
        confirm_full_rerank_quota=False,
    )
    with pytest.raises(ValueError, match='quota'):
        await full_run.execute(arguments)
    status = json.loads((arguments.output / 'evaluation_job_status.json').read_text())
    assert status['state'] == 'failed_or_interrupted'


@pytest.mark.asyncio
async def test_local_qwen_does_not_use_http_or_double_format(monkeypatch):
    adapter = object.__new__(MultimodalRAGAdapter)
    adapter.config = {'rerank': {'provider': 'local_qwen3', 'model_path': '/test-only'}}
    calls = []

    class Local:
        def __init__(self, options):
            calls.append(options)

        def rank(self, query, candidates, limit):
            calls.append((query, candidates, limit))
            return [{**candidates[0], 'rerank_score': 0.9}]

    monkeypatch.setattr(
        'evaluation.src.multimodal.local_reranker.LocalQwenReranker', Local
    )
    candidates = [{'content': 'Paris is in France.'}]
    ranked = await adapter._rerank_candidates(
        'France?', candidates, 1, 'scenario', 'question'
    )
    assert ranked[0]['rerank_score'] == 0.9
    assert calls[1] == ('France?', candidates, 1)


@pytest.mark.asyncio
async def test_local_evaluation_does_not_require_paid_cohere(tmp_path, monkeypatch):
    arguments = argparse.Namespace(
        env_file=tmp_path / 'missing.env',
        output=tmp_path / 'output',
        stage='evaluate',
        data_root=tmp_path,
        config='local.yaml',
        caption_workers=1,
        confirm_full_rerank_quota=False,
    )
    monkeypatch.setattr(
        runner, '_load_config', lambda path: {'rerank': {'provider': 'local_qwen3'}}
    )
    run = AsyncMock()
    monkeypatch.setattr(runner, 'run', run)
    await full_run.execute(arguments)
    run.assert_awaited_once()
