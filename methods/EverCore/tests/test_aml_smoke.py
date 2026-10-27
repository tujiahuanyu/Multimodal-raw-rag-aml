import httpx
import json
from types import SimpleNamespace

from evaluation.src.aml.memgallery_smoke import post, prepare


class Client:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.payloads = []

    def post(self, path, json):
        self.payloads.append(json)
        return httpx.Response(next(self.statuses))


def test_smoke_retries_same_logical_write(monkeypatch):
    monkeypatch.setattr(
        'evaluation.src.aml.memgallery_smoke.time.sleep', lambda delay: None
    )
    client = Client([502, 429, 200])
    payload = {'request_id': 'fixed-logical-write'}
    assert post(client, '/add', payload).status_code == 200
    assert client.payloads == [payload, payload, payload]


def test_smoke_does_not_retry_invalid_payload():
    client = Client([422])
    assert post(client, '/add', {}).status_code == 422
    assert len(client.payloads) == 1


def test_smoke_returns_final_failure(monkeypatch):
    monkeypatch.setattr(
        'evaluation.src.aml.memgallery_smoke.time.sleep', lambda delay: None
    )
    client = Client([502, 502, 502])
    assert post(client, '/add', {}).status_code == 502
    assert len(client.payloads) == 3


def test_public_memgallery_bundle_uses_history_captions_but_not_query_captions(
    tmp_path, monkeypatch
):
    image = tmp_path / 'image.png'
    image.write_bytes(b'local-test-pixels')
    images = [
        {'path': str(image), 'caption': f'Provided caption {number}.'}
        for number in [1, 2]
    ]
    message = SimpleNamespace(
        content='Original round text.',
        sender_id='user',
        timestamp=None,
        metadata={'session': 'D1', 'images': images},
    )
    question = SimpleNamespace(
        question='Find the matching image.',
        question_id='scene:0',
        category='VS',
        evidence=['scoring-only-clue'],
        metadata={
            'question_images': [
                {'path': str(image), 'caption': 'DO_NOT_USE_QUERY_CAPTION'}
            ]
        },
    )
    conversation = SimpleNamespace(conversation_id='scene', messages=[message])
    dataset = SimpleNamespace(conversations=[conversation], qa_pairs=[question])
    calls = []

    def load(data_root, **kwargs):
        calls.append(kwargs)
        return dataset

    monkeypatch.setattr('evaluation.src.multimodal.memgallery.load_memgallery', load)
    monkeypatch.setattr(
        'evaluation.src.aml.memgallery_smoke.image_part',
        lambda source: {
            'type': 'image_url',
            'image_url': {'url': 'data:image/png;base64,AA=='},
        },
    )
    arguments = SimpleNamespace(
        data_root=tmp_path,
        scenario='scene',
        questions=1,
        bundle=tmp_path / 'bundle.json',
    )
    prepare(arguments)
    bundle = json.loads(arguments.bundle.read_text(encoding='utf-8'))
    assert calls == [{'scenario': 'scene', 'use_dataset_captions': True}]
    parts = bundle['adds'][0]['messages'][0]['content']
    assert [part['type'] for part in parts] == [
        'text',
        'text',
        'image_url',
        'text',
        'image_url',
    ]
    assert parts[1]['text'] == 'Provided caption 1.'
    assert parts[3]['text'] == 'Provided caption 2.'
    assert bundle['history_caption_source'] == 'dataset'
    assert bundle['query_caption_source'] == 'generated_on_search'
    assert 'DO_NOT_USE_QUERY_CAPTION' not in json.dumps(bundle)
    assert 'scoring-only-clue' not in json.dumps(bundle)
