import argparse
import base64
from contextlib import contextmanager
from datetime import timezone
from itertools import zip_longest
import json
from pathlib import Path
import re
import time

import httpx

from .config import load_settings


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    temporary.replace(path)


def post(client, path, payload):
    for attempt in range(3):
        try:
            response = client.post(path, json=payload)
        except httpx.TransportError:
            if attempt == 2:
                raise
        else:
            if (
                response.status_code not in {408, 425, 429}
                and response.status_code < 500
            ):
                return response
            if attempt == 2:
                return response
        time.sleep(2**attempt)


def image_part(image):
    path = Path(image['path'])
    data = path.read_bytes()
    if len(data) > 10 * 1024 * 1024:
        raise ValueError('Source image exceeds AML limit; do not silently resize')
    from PIL import Image

    with Image.open(path) as opened:
        mime = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp'}[
            opened.format
        ]
    return {
        'type': 'image_url',
        'image_url': {
            'url': 'data:' + mime + ';base64,' + base64.b64encode(data).decode()
        },
    }


def prepare(arguments):
    from evaluation.src.multimodal.memgallery import load_memgallery

    dataset = load_memgallery(
        arguments.data_root, scenario=arguments.scenario, use_dataset_captions=True
    )
    conversation = dataset.conversations[0]
    questions = dataset.qa_pairs
    groups = [
        [
            question
            for question in questions
            if bool(question.metadata['question_images']) == visual
        ]
        for visual in [False, True]
    ]
    selected = [
        question
        for pair in zip_longest(*groups)
        for question in pair
        if question is not None
    ][: arguments.questions]
    user = 'memgallery-smoke:' + conversation.conversation_id
    batches = []
    current = None
    media_bytes = 0
    words = 0
    for message in conversation.messages:
        parts = [{'type': 'text', 'text': message.content}]
        images = message.metadata.get('images', [])
        for image in images:
            if image.get('caption'):
                parts.append({'type': 'text', 'text': image['caption']})
            parts.append(image_part(image))
        size = sum(Path(image['path']).stat().st_size for image in images)
        session = message.metadata['session']
        word_count = len(message.content.split())
        if (
            current is None
            or current['session_id'] != session
            or len(current['messages']) >= 20
            or media_bytes + size > 30 * 1024 * 1024
            or words + word_count > 2000
        ):
            current = {
                'request_id': user + ':batch:' + str(len(batches)),
                'user_id': user,
                'session_id': session,
                'messages': [],
            }
            batches.append(current)
            media_bytes = 0
            words = 0
        converted = {
            'role': message.sender_id,
            'content': parts if images else message.content,
        }
        if message.timestamp is not None:
            timestamp = message.timestamp
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            converted['timestamp'] = int(timestamp.timestamp() * 1000)
        current['messages'].append(converted)
        media_bytes += size
        words += word_count
    searches = []
    references = []
    for question in selected:
        images = question.metadata['question_images']
        query = [{'type': 'text', 'text': question.question}]
        query.extend(image_part(image) for image in images)
        searches.append(
            {
                'user_id': user,
                'query': query if images else question.question,
                'top_k': 100,
            }
        )
        references.append(
            {
                'question_id': question.question_id,
                'category': question.category,
                'evidence': question.evidence,
                'visual_query': bool(images),
            }
        )
    save(
        arguments.bundle,
        {
            'scenario': conversation.conversation_id,
            'history_caption_source': 'dataset',
            'query_caption_source': 'generated_on_search',
            'adds': batches,
            'searches': searches,
        },
    )
    save(arguments.bundle.with_suffix('.references.json'), references)
    print(
        json.dumps(
            {
                'scenario': conversation.conversation_id,
                'messages': len(conversation.messages),
                'images': sum(
                    len(message.metadata.get('images', []))
                    for message in conversation.messages
                ),
                'add_batches': len(batches),
                'questions': len(searches),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


@contextmanager
def client_context(arguments, settings):
    if arguments.base_url:
        with httpx.Client(
            base_url=arguments.base_url, timeout=1800, trust_env=False
        ) as client:
            yield client, None
    else:
        from fastapi.testclient import TestClient
        from .app import create_app
        from .providers import Providers

        providers = Providers(settings)
        with TestClient(
            create_app(settings, providers), raise_server_exceptions=False
        ) as client:
            yield client, providers


def run(arguments):
    settings = load_settings(arguments.env_file, arguments.data_dir)
    bundle = json.loads(arguments.bundle.read_text(encoding='utf-8'))
    references = json.loads(
        arguments.bundle.with_suffix('.references.json').read_text(encoding='utf-8')
    )
    report = {
        'scenario': bundle['scenario'],
        'protocol': 'Public-data diagnostic retrieval smoke with provided history captions and query-image captioning; no answers or gold evidence sent to retrieval; not official scoring',
        'adds': [],
        'searches': [],
        'passed': False,
    }
    started = time.monotonic()
    try:
        with client_context(arguments, settings) as (client, providers):
            client.headers['Authorization'] = 'Token ' + settings.api_token
            assert client.get('/health').status_code == 200
            for position, payload in enumerate(bundle['adds']):
                before = time.monotonic()
                response = post(client, '/v1/memories/add', payload)
                if response.status_code != 200:
                    raise RuntimeError(
                        'Add failed with HTTP ' + str(response.status_code)
                    )
                assert response.json() == {
                    'success': True,
                    **{
                        name: payload[name]
                        for name in ['request_id', 'user_id', 'session_id']
                    },
                }
                report['adds'].append(
                    {
                        'batch': position,
                        'messages': len(payload['messages']),
                        'seconds': round(time.monotonic() - before, 2),
                    }
                )
                save(arguments.output, report)
                print(
                    'Added batch ' + str(position + 1) + '/' + str(len(bundle['adds'])),
                    flush=True,
                )
            for payload, reference in zip(bundle['searches'], references):
                before = time.monotonic()
                response = post(client, '/v1/memories/search', payload)
                if response.status_code != 200:
                    raise RuntimeError(
                        'Search failed with HTTP ' + str(response.status_code)
                    )
                rows = response.json()['data']
                assert 0 < len(rows) <= payload['top_k']
                rounds = set()
                returned_images = 0
                for row in rows:
                    content = row['content']
                    parts = (
                        [{'type': 'text', 'text': content}]
                        if isinstance(content, str)
                        else content
                    )
                    for part in parts:
                        if part['type'] == 'text':
                            rounds.update(
                                re.findall(r'\[Round ([^\]]+)\]', part['text'])
                            )
                        elif part['type'] == 'image_url':
                            returned_images += 1
                expected = set(reference['evidence'])
                result = {
                    name: reference[name]
                    for name in ['question_id', 'category', 'visual_query']
                }
                result.update(
                    {
                        'returned_items': len(rows),
                        'returned_images': returned_images,
                        'retrieved_rounds': sorted(rounds),
                        'evidence_rounds': sorted(expected),
                        'evidence_recall': (
                            len(rounds & expected) / len(expected) if expected else None
                        ),
                        'seconds': round(time.monotonic() - before, 2),
                    }
                )
                report['searches'].append(result)
                save(arguments.output, report)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            report['passed'] = True
            if providers is not None:
                report['upstream_calls'] = dict(providers.calls)
    except Exception as error:
        report['error_type'] = type(error).__name__
        if isinstance(error, RuntimeError):
            report['error'] = str(error)
    report['elapsed_seconds'] = round(time.monotonic() - started, 2)
    save(arguments.output, report)
    print(
        json.dumps(
            {
                'passed': report['passed'],
                'completed_adds': len(report['adds']),
                'completed_questions': len(report['searches']),
                'seconds': report['elapsed_seconds'],
            }
        ),
        flush=True,
    )
    if not report['passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument(
        '--scenario', default='Astronomy_Physics_Scientific_Experiments_Cosmology'
    )
    parser.add_argument('--questions', type=int, default=3)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, default=Path('.env.aml'))
    parser.add_argument(
        '--data-dir', type=Path, default=Path('/tmp/aml-memgallery-smoke')
    )
    parser.add_argument(
        '--output', type=Path, default=Path('/tmp/aml-memgallery-report.json')
    )
    parser.add_argument('--base-url')
    arguments = parser.parse_args()
    if arguments.command == 'prepare' and (
        arguments.data_root is None or arguments.questions < 1
    ):
        parser.error('prepare requires --data-root and a positive question count')
    if arguments.command == 'prepare':
        prepare(arguments)
    else:
        run(arguments)


if __name__ == '__main__':
    main()
