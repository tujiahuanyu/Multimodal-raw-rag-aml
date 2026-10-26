import asyncio
import base64
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
import hashlib
import io
import json
import re
import warnings

from PIL import Image
import tiktoken

from .providers import ServiceError
from .store import digest


def decode_image(url, settings):
    match = re.fullmatch(
        r'data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=]+)', url
    )
    if not match:
        raise ServiceError(422, 'images_require_jpeg_png_webp_base64_data_uri')
    mime, encoded = match.groups()
    if len(encoded) > 4 * ((settings.max_image_bytes + 2) // 3):
        raise ServiceError(413, 'image_exceeds_10_mib')
    try:
        data = base64.b64decode(encoded, validate=True)
        if len(data) > settings.max_image_bytes:
            raise ServiceError(413, 'image_exceeds_10_mib')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > settings.max_image_pixels:
                    raise ServiceError(413, 'image_pixel_limit_exceeded')
                if (
                    image.format
                    != {'image/jpeg': 'JPEG', 'image/png': 'PNG', 'image/webp': 'WEBP'}[
                        mime
                    ]
                ):
                    raise ServiceError(422, 'image_mime_mismatch')
                image.verify()
    except ServiceError:
        raise
    except Exception:
        raise ServiceError(422, 'invalid_image') from None
    return hashlib.sha256(data).hexdigest(), mime, data


class Engine:
    def __init__(self, settings, store, providers):
        self.settings = settings
        self.store = store
        self.providers = providers
        self.encoding = tiktoken.encoding_for_model('gpt-4')
        self.locks = {}

    async def _caption(self, user, key, url, context, purpose):
        cache_key = digest((purpose, key, context))
        text = self.store.caption(user, cache_key)
        if text is None:
            text = await self.providers.caption(url, context)
            self.store.caption(user, cache_key, text)
        return text

    async def _vectors(self, user, texts):
        vectors = [self.store.embedding(user, digest(text)) for text in texts]
        missing = [
            position for position, vector in enumerate(vectors) if vector is None
        ]
        for start in range(0, len(missing), self.settings.embedding_batch):
            positions = missing[start : start + self.settings.embedding_batch]
            values = await self.providers.embed(
                [texts[position] for position in positions]
            )
            if len(values) != len(positions):
                raise ServiceError(502, 'embedding_batch_mismatch')
            for position, vector in zip(positions, values):
                self.store.embedding(user, digest(texts[position]), vector)
                vectors[position] = vector
        return vectors

    def _normalize(self, content):
        if isinstance(content, str):
            if not content.strip():
                raise ServiceError(422, 'empty_text')
            return content, [], 0
        parts = []
        images = []
        total = 0
        for part in content:
            if part.type == 'text':
                if not part.text.strip():
                    raise ServiceError(422, 'empty_text')
                parts.append({'type': 'text', 'text': part.text})
            else:
                key, mime, data = decode_image(part.image_url.url, self.settings)
                total += len(data)
                images.append((key, mime, data, part.image_url.url))
                parts.append({'type': 'image', 'hash': key})
        return parts, images, total

    async def add(self, request):
        async with self.locks.setdefault(request.user_id, asyncio.Lock()):
            payload_hash = digest(request.model_dump(mode='json'))
            previous = self.store.replay(
                request.user_id, request.request_id, payload_hash
            )
            if previous is not None:
                return previous
            normalized = [
                self._normalize(message.content) for message in request.messages
            ]
            if sum(item[2] for item in normalized) > self.settings.max_media_bytes:
                raise ServiceError(413, 'request_media_exceeds_30_mib')
            existing = self.store.messages(request.user_id, request.session_id)
            added = []
            media = {}
            for position, (message, (content, images, size)) in enumerate(
                zip(request.messages, normalized)
            ):
                text = (
                    content
                    if isinstance(content, str)
                    else '\n'.join(
                        part['text'] for part in content if part['type'] == 'text'
                    )
                )
                visual = {}
                for key, mime, data, url in images:
                    visual[key] = await self._caption(
                        request.user_id, key, url, text, 'history'
                    )
                    media[key] = (mime, data)
                moment = (
                    datetime.fromtimestamp(
                        message.timestamp / 1000, timezone.utc
                    ).isoformat()
                    if message.timestamp is not None
                    else 'unknown date'
                )
                lines = [f'[{moment}] {message.role}:']
                if isinstance(content, str):
                    lines.append(content)
                else:
                    for part in content:
                        lines.append(
                            part['text']
                            if part['type'] == 'text'
                            else f"[Image img_{part['hash'][:16]}]\nVisual description: {visual[part['hash']]}"
                        )
                added.append(
                    {
                        'message_id': digest(
                            (
                                request.user_id,
                                request.session_id,
                                request.request_id,
                                position,
                            )
                        ),
                        'user_id': request.user_id,
                        'session_id': request.session_id,
                        'ordinal': len(existing) + position,
                        'content': json.dumps(content, ensure_ascii=False),
                        'render': '\n'.join(lines),
                        'role': message.role,
                        'timestamp': message.timestamp,
                    }
                )
            chunks = self._chunks(request.user_id, request.session_id, existing + added)
            vectors = await self._vectors(
                request.user_id, [row['content'] for row in chunks]
            )
            for row, vector in zip(chunks, vectors):
                row['vector'] = vector
            return self.store.commit(request, payload_hash, added, chunks, media)

    def _chunks(self, user, session, messages):
        document = '\n'.join(row['render'] for row in messages)
        starts, ends = [], []
        offset = 0
        for row in messages:
            starts.append(offset)
            offset += len(row['render'])
            ends.append(offset)
            offset += 1
        tokens = self.encoding.encode(document, disallowed_special=())
        _, offsets = self.encoding.decode_with_offsets(tokens)
        chunks = []
        for start in range(
            0, len(tokens), self.settings.chunk_size - self.settings.chunk_overlap
        ):
            end = min(start + self.settings.chunk_size, len(tokens))
            char_start = offsets[start]
            char_end = offsets[end] if end < len(tokens) else len(document)
            first = bisect_right(ends, char_start)
            last = bisect_left(starts, char_end)
            content = self.encoding.decode(tokens[start:end])
            sources = [row['message_id'] for row in messages[first:last]]
            if not sources:
                raise ServiceError(500, 'source_mapping_failed')
            chunks.append(
                {
                    'chunk_id': digest((user, session, start, content)),
                    'content': content,
                    'source_ids': sources,
                }
            )
            if end == len(tokens):
                break
        return chunks

    async def search(self, request):
        async with self.locks.setdefault(request.user_id, asyncio.Lock()):
            if request.top_k == 0 or not self.store.has_user(request.user_id):
                return {'data': []}
            content, images, size = self._normalize(request.query)
            if size > self.settings.max_media_bytes:
                raise ServiceError(413, 'query_media_exceeds_30_mib')
            text = (
                content
                if isinstance(content, str)
                else '\n'.join(
                    part['text'] for part in content if part['type'] == 'text'
                )
            )
            for key, mime, data, url in images:
                caption = await self._caption(request.user_id, key, url, '', 'query')
                text += '\nQuery image description: ' + caption
            if request.options:
                text += '\nAnswer options:\n' + '\n'.join(request.options)
            vector = (await self._vectors(request.user_id, [text]))[0]
            candidates = await asyncio.to_thread(
                self.store.candidates,
                request.user_id,
                vector,
                self.settings.candidate_count,
            )
            limit = min(request.top_k, self.settings.result_count, len(candidates))
            if not limit:
                return {'data': []}
            ranked = await self.providers.rerank(text, candidates, limit)
            output = []
            total_media = 0
            for candidate in ranked:
                result, image_bytes = self._evidence(request.user_id, candidate)
                if total_media + image_bytes > self.settings.max_media_bytes:
                    continue
                total_media += image_bytes
                output.append(result)
            if not output:
                raise ServiceError(413, 'retrieved_evidence_exceeds_media_budget')
            result = {'data': output}
            if (
                len(json.dumps(result, ensure_ascii=False).encode())
                > self.settings.max_body_bytes
            ):
                raise ServiceError(413, 'search_response_exceeds_service_limit')
            return result

    def _evidence(self, user, candidate):
        parts = []
        image_bytes = 0
        multimodal = False
        created_at = None
        for message in self.store.sources(user, candidate['source_ids']):
            moment = (
                datetime.fromtimestamp(
                    message['timestamp'] / 1000, timezone.utc
                ).isoformat()
                if message['timestamp'] is not None
                else 'unknown date'
            )
            if message['timestamp'] is not None and created_at is None:
                created_at = moment
            parts.append(
                {
                    'type': 'text',
                    'text': f"[Session {message['session_id']} | {moment} | {message['role']}]",
                }
            )
            content = json.loads(message['content'])
            if isinstance(content, str):
                parts.append({'type': 'text', 'text': content})
            else:
                multimodal = True
                for part in content:
                    if part['type'] == 'text':
                        parts.append(part)
                    else:
                        mime, data = self.store.image(user, part['hash'])
                        image_bytes += len(data)
                        parts.append(
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': 'data:'
                                    + mime
                                    + ';base64,'
                                    + base64.b64encode(data).decode()
                                },
                            }
                        )
        result = {
            'id': candidate['chunk_id'],
            'score': candidate['score'],
            'content': (
                parts if multimodal else '\n'.join(part['text'] for part in parts)
            ),
        }
        if created_at is not None:
            result['created_at'] = created_at
        return result, image_bytes
