"""Keep Raw-RAG retrieval while rendering image descriptions with their IDs."""

import json
import time
import urllib.request

from evaluation.src.adapters.rag_adapter import RAGAdapter


class MultimodalRAGAdapter(RAGAdapter):
    def _post_json(self, url, payload, timeout_seconds):
        options = self.config.get('rerank', {})
        interval = max(0, float(options.get('min_interval_seconds', 0)))
        delay = getattr(self, '_next_rerank_time', 0) - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_rerank_time = time.monotonic() + interval
        proxy = options.get('proxy_url')
        if not proxy:
            return super()._post_json(url, payload, timeout_seconds)
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({'http': proxy, 'https': proxy})
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + self.rerank_api_key,
            },
            method='POST',
        )
        with opener.open(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode('utf-8'))

    def _format_message_line(self, message):
        text = super()._format_message_line(message)
        parts = [text]
        for image in message.metadata.get("images", []):
            parts.append(f"Image ID: {image['image_id']}")
            if image.get("caption"):
                parts.append(f"Image description: {image['caption']}")
        return "\n".join(parts)
