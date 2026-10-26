import httpx

from evaluation.src.aml.memgallery_smoke import post


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
