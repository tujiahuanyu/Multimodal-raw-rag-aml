import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values
import httpx
from PIL import Image


EXAMPLES_PATH = (
    Path(__file__).resolve().parents[2] / 'config' / 'aml_official_examples.json'
)


def materialize_examples():
    examples = json.loads(EXAMPLES_PATH.read_text(encoding='utf-8'))
    for case in examples['cases']:
        for content in [case['add']['messages'][0]['content'], case['search']['query']]:
            if isinstance(content, str):
                continue
            for part in content:
                if part['type'] != 'image_url':
                    continue
                mime = part['image_url']['url'].split(';', 1)[0].split(':', 1)[1]
                buffer = io.BytesIO()
                Image.new('RGB', (16, 16), 'red').save(
                    buffer, format={'image/jpeg': 'JPEG', 'image/png': 'PNG'}[mime]
                )
                part['image_url']['url'] = (
                    'data:'
                    + mime
                    + ';base64,'
                    + base64.b64encode(buffer.getvalue()).decode()
                )
    return examples


def validate_search(body, top_k):
    if not isinstance(body, dict) or not isinstance(body.get('data'), list):
        raise ValueError('Search response must contain a data array')
    rows = body['data']
    if len(rows) > top_k:
        raise ValueError('Search response exceeds top_k')
    scores = []
    identifiers = set()
    media_bytes = 0
    for row in rows:
        identifier = row.get('id')
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
        ):
            raise ValueError('Search IDs must be nonempty and unique')
        identifiers.add(identifier)
        content = row.get('content')
        if isinstance(content, str):
            if not content.strip():
                raise ValueError('Search content must be nonempty')
        elif isinstance(content, list) and content:
            for part in content:
                if part.get('type') == 'text':
                    if (
                        not isinstance(part.get('text'), str)
                        or not part['text'].strip()
                    ):
                        raise ValueError('Invalid text part')
                elif part.get('type') == 'image_url':
                    url = part.get('image_url', {}).get('url', '')
                    header, encoded = url.split(',', 1)
                    if header not in {
                        'data:image/jpeg;base64',
                        'data:image/png;base64',
                        'data:image/webp;base64',
                    }:
                        raise ValueError('Invalid returned image URI')
                    data = base64.b64decode(encoded, validate=True)
                    if len(data) > 10 * 1024 * 1024:
                        raise ValueError('Returned image exceeds 10 MiB')
                    with Image.open(io.BytesIO(data)) as image:
                        expected = {'JPEG': 'jpeg', 'PNG': 'png', 'WEBP': 'webp'}[
                            image.format
                        ]
                        if header != 'data:image/' + expected + ';base64':
                            raise ValueError('Returned image MIME mismatch')
                        image.verify()
                    media_bytes += len(data)
                else:
                    raise ValueError('Invalid content part type')
        else:
            raise ValueError('Invalid Search content')
        if 'score' in row:
            score = row['score']
            if type(score) not in {int, float} or not math.isfinite(score):
                raise ValueError('Invalid relevance score')
            scores.append(score)
        if 'created_at' in row:
            datetime.fromisoformat(row['created_at'].replace('Z', '+00:00'))
    if media_bytes > 30 * 1024 * 1024:
        raise ValueError('Search media exceeds 30 MiB')
    if scores != sorted(scores, reverse=True):
        raise ValueError('Search scores are not relevance ordered')
    return rows


def run(client, token, report, pause_seconds=6.2):
    examples = materialize_examples()
    report.update(
        {
            'document_url': examples['source'],
            'examples_reviewed_on': examples['reviewed_on'],
            'image_placeholder_substitution': examples['media_substitution'],
            'protocol': 'Participant-run official-example compatibility test; not platform-issued Smoke or Answer/Eval',
            'checks': [],
            'passed': False,
        }
    )
    started = time.monotonic()
    next_search = 0.0

    def check(name, condition, status=None):
        report['checks'].append(
            {'name': name, 'passed': bool(condition), 'http': status}
        )
        if not condition:
            raise ValueError('Contract check failed: ' + name)

    def search(payload, headers):
        nonlocal next_search
        delay = next_search - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        next_search = time.monotonic() + pause_seconds
        return client.post('/v1/memories/search', json=payload, headers=headers)

    authorization = {'Authorization': 'Token ' + token}
    try:
        response = client.get('/health')
        check('public_health', 200 <= response.status_code < 300, response.status_code)
        for path, payload in [
            ('/v1/memories/add', examples['cases'][0]['add']),
            ('/v1/memories/search', examples['cases'][0]['search']),
        ]:
            response = client.post(path, json=payload)
            check(
                'unauthenticated_' + path.rsplit('/', 1)[1],
                response.status_code == 401,
                response.status_code,
            )
        for case in examples['cases']:
            name = case['name']
            payload = case['add']
            expected = {
                'success': True,
                **{
                    key: payload[key] for key in ['request_id', 'user_id', 'session_id']
                },
            }
            response = client.post(
                '/v1/memories/add', json=payload, headers=authorization
            )
            check(
                name + '_add',
                response.status_code == 200
                and response.json() == expected
                and response.json().get('success') is True,
                response.status_code,
            )
            replay = client.post(
                '/v1/memories/add', json=payload, headers=authorization
            )
            check(
                name + '_idempotent_replay',
                replay.status_code == 200 and replay.json() == expected,
                replay.status_code,
            )
            response = search(case['search'], authorization)
            check(
                name + '_search_http', response.status_code == 200, response.status_code
            )
            rows = validate_search(response.json(), case['search']['top_k'])
            check(name + '_search_schema', bool(rows))
            source = payload['messages'][0]['content']
            check(
                name + '_raw_source_preserved',
                any(
                    (
                        source in row['content']
                        if isinstance(source, str)
                        else isinstance(row['content'], list)
                        and row['content'][1:] == source
                    )
                    for row in rows
                ),
            )
            repeated = search(case['search'], authorization)
            check(
                name + '_stable_ids',
                repeated.status_code == 200
                and [
                    (row['id'], row['content'])
                    for row in validate_search(repeated.json(), case['search']['top_k'])
                ]
                == [(row['id'], row['content']) for row in rows],
                repeated.status_code,
            )
            bounded = search({**case['search'], 'top_k': 1}, authorization)
            check(
                name + '_top_k_one',
                bounded.status_code == 200
                and len(validate_search(bounded.json(), 1)) == 1,
                bounded.status_code,
            )
            conflict = deepcopy(payload)
            conflict['messages'][0][
                'content'
            ] = 'Different payload for the same request ID'
            response = client.post(
                '/v1/memories/add', json=conflict, headers=authorization
            )
            check(
                name + '_request_conflict',
                response.status_code == 409,
                response.status_code,
            )
        empty = {
            'query': 'unseen user',
            'user_id': 'contract-empty-' + uuid4().hex,
            'top_k': 100,
        }
        for scheme, headers in [
            ('Token', authorization),
            ('Bearer', {'Authorization': 'Bearer ' + token}),
            ('X-Api-Key', {'X-Api-Key': token}),
        ]:
            response = search(empty, headers)
            check(
                scheme + '_empty_scope',
                response.status_code == 200 and response.json() == {'data': []},
                response.status_code,
            )
        malformed = {key: value for key, value in empty.items() if key != 'top_k'}
        response = client.post(
            '/v1/memories/search', json=malformed, headers=authorization
        )
        check('missing_top_k', response.status_code in {400, 422}, response.status_code)
        malformed = {
            **empty,
            'query': [
                {
                    'type': 'image_url',
                    'image_url': {'url': 'https://example.com/not-inline.png'},
                }
            ],
        }
        response = client.post(
            '/v1/memories/search', json=malformed, headers=authorization
        )
        check(
            'invalid_image_empty_scope',
            response.status_code in {400, 422},
            response.status_code,
        )
        report['passed'] = True
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    parsed = urlsplit(arguments.base_url)
    if (
        parsed.scheme != 'https'
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        parser.error('Use an HTTPS base URL without credentials, query or fragment')
    token = dotenv_values(arguments.env_file).get('AML_API_TOKEN')
    if not token:
        parser.error('AML_API_TOKEN is missing from the private environment file')
    report = {
        'tested_at_utc': datetime.now(timezone.utc).isoformat(),
        'tls_verified': True,
    }
    try:
        with httpx.Client(
            base_url=arguments.base_url,
            timeout=1800,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            run(client, token, report)
    except Exception as error:
        report['passed'] = False
        report['error_type'] = type(error).__name__
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(
        json.dumps(
            {
                'passed': report['passed'],
                'checks': len(report.get('checks', [])),
                'seconds': report.get('elapsed_seconds'),
            }
        )
    )
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
