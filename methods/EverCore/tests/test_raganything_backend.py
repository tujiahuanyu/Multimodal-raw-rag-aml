"""Offline contract tests for the optional RAG-Anything retrieval backend."""

import asyncio
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from evaluation.src.core.data_models import Conversation, Message
from evaluation.src.multimodal import raganything_backend


class FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.close = AsyncMock()
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="description")
                            )
                        ]
                    )
                )
            )
        )
        self.embeddings = SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    data=[
                        SimpleNamespace(index=1, embedding=[3.0, 4.0]),
                        SimpleNamespace(index=0, embedding=[1.0, 2.0]),
                    ]
                )
            )
        )
        self.instances.append(self)


class FakeRAG:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.insert_content_list = AsyncMock()
        self.aquery = AsyncMock(return_value="")
        self.finalize_storages = AsyncMock()


def fake_dependencies():
    return (
        FakeRAG,
        lambda **kwargs: SimpleNamespace(**kwargs),
        lambda **kwargs: SimpleNamespace(**kwargs),
        FakeClient,
        SimpleNamespace(asarray=lambda values, dtype=None: values),
    )


class RAGAnythingBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_dir = self.root / "images"
        self.image_dir.mkdir()
        self.first_image = self.image_dir / "first.png"
        self.second_image = self.image_dir / "second.png"
        self.first_image.write_bytes(b"first-image")
        self.second_image.write_bytes(b"second-image")
        self.config = {
            "llm": {"api_key": "test-key", "model": "vision-model"},
            "embedding": {
                "api_key": "embedding-key",
                "model": "test-embed",
                "dimensions": 2,
            },
        }
        FakeClient.instances = []
        self.dependencies = patch.object(
            raganything_backend, "_dependencies", side_effect=fake_dependencies
        )
        self.dependencies.start()
        self.backend = raganything_backend.RAGAnythingBackend(
            self.config, self.root / "output", self.image_dir
        )

    async def asyncTearDown(self):
        await self.backend.close()
        self.dependencies.stop()
        self.temporary.cleanup()

    def conversation(self, scenario="scenario-one"):
        return Conversation(
            scenario,
            [
                Message(
                    "alice",
                    "Alice",
                    "The blue mug is on my desk.",
                    metadata={
                        "round_id": "round-1",
                        "session": "session-1",
                        "images": [
                            {
                                "image_id": "img1",
                                "path": str(self.first_image),
                                "caption": "Blue mug",
                            }
                        ],
                    },
                ),
                Message(
                    "alice",
                    "Alice",
                    "A separate vacation picture.",
                    metadata={
                        "round_id": "round-10",
                        "session": "session-2",
                        "images": [
                            {
                                "image_id": "img10",
                                "path": str(self.second_image),
                                "caption": "Vacation",
                            }
                        ],
                    },
                ),
            ],
        )

    async def test_structured_insert_preserves_original_text_image_and_round_ids(self):
        index = await self.backend.add([self.conversation()])
        record = index["scenario-one"]
        rag = record["rag"]
        insertion = rag.insert_content_list.await_args.kwargs
        blocks = insertion["content_list"]
        self.assertEqual(
            [block["type"] for block in blocks], ["text", "image", "text", "image"]
        )
        self.assertIn("The blue mug is on my desk.", blocks[0]["text"])
        self.assertIn('Round ID: "round-1"', blocks[0]["text"])
        self.assertIn('Image ID: "img1"', blocks[0]["text"])
        self.assertEqual(blocks[1]["img_path"], str(self.first_image.resolve()))
        self.assertEqual(blocks[1]["page_idx"], blocks[0]["page_idx"])
        self.assertFalse(hasattr(rag, "_parser_installation_checked"))
        self.assertEqual(
            rag.options["lightrag_kwargs"]["working_dir"], record["working_dir"]
        )

    async def test_scenarios_have_separate_storage_and_graph_namespaces(self):
        index = await self.backend.add(
            [self.conversation("scenario-one"), self.conversation("scenario-two")]
        )
        first, second = index.values()
        self.assertNotEqual(first["working_dir"], second["working_dir"])
        self.assertNotEqual(
            first["rag"].options["lightrag_kwargs"]["workspace"],
            second["rag"].options["lightrag_kwargs"]["workspace"],
        )
        with self.assertRaisesRegex(ValueError, "Missing RAG-Anything scenario"):
            await self.backend.search("question", "unknown", index)

    async def test_search_retrieves_context_only_and_does_not_return_entire_image_library(
        self,
    ):
        index = await self.backend.add([self.conversation()])
        rag = index["scenario-one"]["rag"]
        rag.aquery.return_value = 'Round ID: "round-1"\nImage ID: "img1"\nBlue mug.'
        result = await self.backend.search(
            "What color?", "scenario-one", index, question_id="q1", top_k=3
        )
        self.assertEqual(result.question_id, "q1")
        self.assertEqual(
            [image["image_id"] for image in result.retrieval_metadata["images"]],
            ["img1"],
        )
        self.assertEqual(result.results[0]["metadata"]["round_ids"], ["round-1"])
        rag.aquery.assert_awaited_once_with(
            "What color?",
            mode="mix",
            only_need_context=True,
            vlm_enhanced=False,
            top_k=3,
        )
        for client in FakeClient.instances:
            client.chat.completions.create.assert_not_awaited()

    async def test_image_path_matching_and_identifier_boundaries(self):
        index = await self.backend.add([self.conversation()])
        rag = index["scenario-one"]["rag"]
        rag.aquery.return_value = 'Image ID: "img10"\nRound ID: "round-10"'
        result = await self.backend.search("picture", "scenario-one", index)
        self.assertEqual(
            [image["image_id"] for image in result.retrieval_metadata["images"]],
            ["img10"],
        )
        self.assertEqual(result.retrieval_metadata["round_ids"], ["round-10"])
        rag.aquery.return_value = f"Image Path: {self.first_image.resolve()}"
        result = await self.backend.search("picture", "scenario-one", index)
        self.assertEqual(
            [image["image_id"] for image in result.retrieval_metadata["images"]],
            ["img1"],
        )
        rag.aquery.return_value = (
            "No image references. img1 is only an incidental mention."
        )
        result = await self.backend.search("picture", "scenario-one", index)
        self.assertEqual(result.retrieval_metadata["images"], [])

    async def test_image_outside_allowed_root_is_rejected_before_insertion(self):
        outside = self.root / "outside.png"
        outside.write_bytes(b"private")
        conversation = self.conversation()
        conversation.messages[0].metadata["images"][0]["path"] = str(outside)
        with self.assertRaisesRegex(ValueError, "allowed image root"):
            await self.backend.add([conversation])
        self.assertEqual(self.backend._indices, [])

    async def test_embeddings_are_reordered_and_validated(self):
        embeddings = await self.backend._embed(["one", "two"])
        self.assertEqual(embeddings, [[1.0, 2.0], [3.0, 4.0]])
        self.backend._embedding_client.embeddings.create.return_value.data[
            0
        ].embedding = [1.0]
        with self.assertRaisesRegex(ValueError, "embedding dimension"):
            await self.backend._embed(["one", "two"])

    async def test_vision_callback_builds_image_message_without_forwarding_internal_arguments(
        self,
    ):
        answer = await self.backend._vision(
            "describe",
            system_prompt="system",
            image_data="encoded-image",
            hashing_kv=object(),
            keyword_extraction=False,
        )
        self.assertEqual(answer, "description")
        request = self.backend._vision_client.chat.completions.create.await_args.kwargs
        self.assertNotIn("hashing_kv", request)
        self.assertEqual(
            request["messages"][1]["content"][1]["image_url"]["url"],
            "data:image/jpeg;base64,encoded-image",
        )

    async def test_close_finalizes_storage_and_clients_once(self):
        index = await self.backend.add([self.conversation()])
        await self.backend.close()
        await self.backend.close()
        index["scenario-one"]["rag"].finalize_storages.assert_awaited_once()
        for client in FakeClient.instances:
            client.close.assert_awaited_once()

    async def test_parser_dependency_failure_has_actionable_error(self):
        with patch.object(FakeRAG, "__init__", autospec=True) as constructor:

            def initialize(instance, **kwargs):
                instance.insert_content_list = AsyncMock(
                    side_effect=RuntimeError("Parser mineru is not properly installed")
                )
                instance.finalize_storages = AsyncMock()

            constructor.side_effect = initialize
            with self.assertRaisesRegex(
                RuntimeError, "complete official parser dependencies"
            ):
                await self.backend.add([self.conversation()])


class OptionalDependencyTests(unittest.TestCase):
    def test_module_can_import_without_raganything_and_initialization_explains_missing_dependency(
        self,
    ):
        with patch.dict(sys.modules, {"raganything": None}):
            module = importlib.reload(raganything_backend)
            with self.assertRaisesRegex(RuntimeError, "isolated environment"):
                module._dependencies()


if __name__ == "__main__":
    unittest.main()
