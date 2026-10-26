"""Optional RAG-Anything retrieval with the evaluation runner's shared reader."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from evaluation.src.core.data_models import Conversation, SearchResult


def _dependencies():
    try:
        from raganything import RAGAnything, RAGAnythingConfig
        from lightrag.utils import EmbeddingFunc
        from openai import AsyncOpenAI
        import numpy
    except ImportError as error:
        raise RuntimeError(
            "The optional RAG-Anything backend requires its isolated environment "
            "with raganything[image], lightrag-hku<1.5, numpy and openai. Install "
            "the complete official dependencies, including MinerU; the raw "
            "backend does not require them."
        ) from error
    return RAGAnything, RAGAnythingConfig, EmbeddingFunc, AsyncOpenAI, numpy


def _setting(value, *environment_names, default=None):
    text = str(value).strip() if value is not None else ""
    if text and not (text.startswith("${") and text.endswith("}")):
        return text
    return next(
        (
            os.environ[name].strip()
            for name in environment_names
            if os.environ.get(name, "").strip()
        ),
        default,
    )


def _reference(label: str, value: str) -> str:
    return f"{label}: {json.dumps(value, ensure_ascii=False)}"


def _has_reference(context: str, label: str, value: str) -> bool:
    encoded = json.dumps(value, ensure_ascii=False)
    labels = re.escape(label).replace(r"\ ", r"[ _]")
    return bool(
        re.search(
            rf"\b{labels}\s*[:=]\s*(?:{re.escape(encoded)}|{re.escape(value)})(?![\w.\-])",
            context,
            flags=re.IGNORECASE,
        )
    )


class RAGAnythingBackend:
    def __init__(self, config: dict, output_dir: Path, allowed_root: Path):
        rag_class, config_class, embedding_class, client_class, self._numpy = (
            _dependencies()
        )
        self._rag_class = rag_class
        self._config_class = config_class
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.allowed_root = Path(allowed_root).resolve(strict=True)
        self.options = dict(config.get("raganything", {}))
        self.mode = self.options.get("mode", "mix")
        if self.mode not in {"mix", "local", "global", "hybrid", "naive"}:
            raise ValueError(
                "raganything.mode must be mix, local, global, hybrid or naive"
            )
        self._indices = []
        self._closed = False

        embedding = dict(config.get("embedding", {}))
        self.embedding_model = embedding.get("model", "text-embedding-3-small")
        known_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        dimension = embedding.get("dimensions", self.options.get("embedding_dim"))
        self.embedding_dim = int(
            dimension or known_dimensions.get(self.embedding_model, 0)
        )
        if self.embedding_dim <= 0:
            raise ValueError(
                "Set embedding.dimensions or raganything.embedding_dim for this embedding model"
            )
        self._embedding_options = embedding
        self._llm_options = dict(config.get("llm", {}))
        self._vision_options = {**self._llm_options, **config.get("vision", {})}

        def client(options, embedding=False):
            key = _setting(
                options.get("api_key"),
                *("VECTORIZE_API_KEY", "EMBEDDING_API_KEY") if embedding else (),
                "LLM_API_KEY",
                "OPENAI_API_KEY",
            )
            if not key:
                raise ValueError(
                    "RAG-Anything requires configured embedding and llm API keys"
                )
            url = _setting(
                options.get("base_url"),
                *("VECTORIZE_BASE_URL", "EMBEDDING_BASE_URL") if embedding else (),
                "LLM_BASE_URL",
                "OPENAI_BASE_URL",
                default="https://api.openai.com/v1",
            )
            return client_class(
                api_key=key,
                base_url=url,
                timeout=float(options.get("timeout_seconds", 120)),
                max_retries=int(options.get("max_retries", 3)),
            )

        self._llm_client = client(self._llm_options)
        self._vision_client = (
            self._llm_client
            if not config.get("vision")
            else client(self._vision_options)
        )
        self._embedding_client = client(embedding, embedding=True)
        self._embedding_func = embedding_class(
            embedding_dim=self.embedding_dim,
            max_token_size=int(embedding.get("max_token_size", 8192)),
            func=self._embed,
        )

    async def _embed(self, texts, **kwargs):
        request = {
            "model": self.embedding_model,
            "input": texts,
            "encoding_format": "float",
        }
        if self._embedding_options.get("dimensions") is not None:
            request["dimensions"] = self.embedding_dim
        response = await self._embedding_client.embeddings.create(**request)
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts) or any(
            len(item.embedding) != self.embedding_dim for item in ordered
        ):
            raise ValueError(
                "Embedding response does not match the configured batch size or embedding dimension"
            )
        return self._numpy.asarray([item.embedding for item in ordered], dtype=float)

    async def _complete(
        self,
        options,
        client,
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages or [])
            content = prompt
            if image_data:
                url = (
                    image_data
                    if image_data.startswith("data:")
                    else f"data:image/jpeg;base64,{image_data}"
                )
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url}},
                ]
            messages.append({"role": "user", "content": content})
        request = {
            "model": options.get("model", "gpt-4o-mini"),
            "messages": messages,
            "temperature": float(options.get("temperature", 0)),
            "max_tokens": int(options.get("max_tokens", 4096)),
        }
        for name in (
            "temperature",
            "max_tokens",
            "top_p",
            "seed",
            "response_format",
            "stop",
        ):
            if name in kwargs:
                request[name] = kwargs[name]
        response = await client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    async def _llm(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        return await self._complete(
            self._llm_options,
            self._llm_client,
            prompt,
            system_prompt,
            history_messages,
            **kwargs,
        )

    async def _vision(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        return await self._complete(
            self._vision_options,
            self._vision_client,
            prompt,
            system_prompt,
            history_messages,
            image_data,
            messages,
            **kwargs,
        )

    def _image(self, image: dict, round_id: str, session_id: str) -> dict:
        image_id = str(image.get("image_id", "")).strip()
        if not image_id or not image.get("path"):
            raise ValueError("Each RAG-Anything image needs image_id and path")
        path = Path(image["path"]).expanduser()
        path = (path if path.is_absolute() else self.allowed_root / path).resolve(
            strict=True
        )
        if not path.is_relative_to(self.allowed_root) or not path.is_file():
            raise ValueError(
                "Image path must be a file under the configured allowed image root"
            )
        return {
            **image,
            "image_id": image_id,
            "path": str(path),
            "round_ids": [round_id] if round_id else [],
            "session_id": session_id,
        }

    def _content(self, conversation: Conversation):
        content_list = []
        images = {}
        round_ids = []
        for page_idx, message in enumerate(conversation.messages):
            metadata = message.metadata or {}
            round_id = str(metadata.get("round_id", ""))
            session_id = str(metadata.get("session", metadata.get("session_id", "")))
            if round_id and round_id not in round_ids:
                round_ids.append(round_id)
            lines = [
                _reference("Round ID", round_id),
                _reference("Session ID", session_id),
                f"Speaker: {message.sender_name or message.sender_id}",
            ]
            if message.timestamp:
                lines.append(f"Timestamp: {message.timestamp.isoformat()}")
            lines.append(message.content)
            message_images = []
            for value in metadata.get("images", []):
                image = self._image(value, round_id, session_id)
                previous = images.get(image["image_id"])
                if previous and previous["path"] != image["path"]:
                    raise ValueError(
                        "An image_id cannot refer to different files within one scenario"
                    )
                if previous:
                    previous["round_ids"] = list(
                        dict.fromkeys(previous["round_ids"] + image["round_ids"])
                    )
                else:
                    images[image["image_id"]] = image
                lines.extend(
                    [
                        _reference("Image ID", image["image_id"]),
                        f"Image Path: {image['path']}",
                    ]
                )
                caption = str(image.get("caption", ""))
                if caption:
                    lines.append(f"Image caption: {caption}")
                message_images.append(
                    {
                        "type": "image",
                        "img_path": image["path"],
                        "page_idx": page_idx,
                        "image_caption": [
                            _reference("Image ID", image["image_id"]),
                            caption,
                        ],
                        "image_footnote": [_reference("Round ID", round_id)],
                    }
                )
            content_list.append(
                {"type": "text", "text": "\n".join(lines), "page_idx": page_idx}
            )
            content_list.extend(message_images)
        return content_list, list(images.values()), round_ids

    async def add(self, conversations: list[Conversation]) -> Any:
        if self._closed:
            raise RuntimeError("RAG-Anything backend is closed")
        index = {}
        for conversation in conversations:
            conversation_id = conversation.conversation_id
            if conversation_id in index:
                raise ValueError(f"Duplicate scenario ID: {conversation_id}")
            content_list, images, round_ids = self._content(conversation)
            if not content_list:
                raise ValueError(f"Cannot index an empty scenario: {conversation_id}")
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "conversation_id": conversation_id,
                        "content": content_list,
                        "embedding_model": self.embedding_model,
                        "embedding_dim": self.embedding_dim,
                        "llm_model": self._llm_options.get("model"),
                        "vision_model": self._vision_options.get("model"),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            working_dir = self.output_dir / "raganything" / digest
            config = self._config_class(
                working_dir=str(working_dir),
                parser_output_dir=str(working_dir / "parsed"),
                enable_image_processing=True,
                enable_table_processing=False,
                enable_equation_processing=False,
                display_content_stats=False,
                use_full_path=True,
            )
            lightrag_options = {
                **self.options.get("lightrag_kwargs", {}),
                "working_dir": str(working_dir),
                "workspace": digest,
            }
            rag = self._rag_class(
                config=config,
                llm_model_func=self._llm,
                vision_model_func=self._vision,
                embedding_func=self._embedding_func,
                lightrag_kwargs=lightrag_options,
            )
            record = {
                "rag": rag,
                "images": images,
                "round_ids": round_ids,
                "working_dir": str(working_dir),
                "conversation_id": conversation_id,
            }
            self._indices.append(record)
            try:
                await rag.insert_content_list(
                    content_list=content_list,
                    file_path=f"scenario-{digest}.txt",
                    doc_id=f"scenario-{digest}",
                    display_stats=False,
                )
            except RuntimeError as error:
                if "parser" in str(error).lower() and "install" in str(error).lower():
                    raise RuntimeError(
                        "RAG-Anything initialization requires the complete official parser dependencies "
                        "(including MinerU), even when inserting structured content directly. "
                        "Install them in the isolated RAG-Anything environment."
                    ) from error
                raise
            index[conversation_id] = record
        return index

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        if self._closed:
            raise RuntimeError("RAG-Anything backend is closed")
        record = (index or {}).get(conversation_id)
        if record is None or record.get("conversation_id") != conversation_id:
            raise ValueError(f"Missing RAG-Anything scenario: {conversation_id}")
        query_options = {}
        for name in (
            "top_k",
            "chunk_top_k",
            "max_entity_tokens",
            "max_relation_tokens",
            "max_total_tokens",
        ):
            value = kwargs.get(name, self.options.get(name))
            if value is not None:
                query_options[name] = int(value)
        context = await record["rag"].aquery(
            query,
            mode=self.mode,
            only_need_context=True,
            vlm_enhanced=False,
            **query_options,
        )
        if context is None:
            context = ""
        if not isinstance(context, str):
            raise TypeError("RAG-Anything retrieval must return a text context")
        normalized = context.replace("\\\\", "\\").replace("\\/", "/")
        hit_images = []
        for image in record["images"]:
            path = image["path"]
            path_found = any(
                re.search(re.escape(candidate) + r"(?![\w.\-])", normalized)
                for candidate in {path, path.replace("\\", "/")}
            )
            if path_found or _has_reference(context, "Image ID", image["image_id"]):
                hit_images.append(dict(image))
        hit_rounds = [
            round_id
            for round_id in record["round_ids"]
            if _has_reference(context, "Round ID", round_id)
        ]
        metadata = {
            "backend": "raganything",
            "mode": self.mode,
            "formatted_context": context,
            "images": hit_images,
            "round_ids": hit_rounds,
            **query_options,
        }
        results = (
            [
                {
                    "content": context,
                    "score": 1.0,
                    "metadata": {"images": hit_images, "round_ids": hit_rounds},
                }
            ]
            if context
            else []
        )
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata=metadata,
            question_id=kwargs.get("question_id", ""),
        )

    async def close(self):
        if self._closed:
            return
        self._closed = True
        clients = {
            id(value): value
            for value in (self._llm_client, self._vision_client, self._embedding_client)
        }
        outcomes = await asyncio.gather(
            *(record["rag"].finalize_storages() for record in self._indices),
            *(client.close() for client in clients.values()),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
