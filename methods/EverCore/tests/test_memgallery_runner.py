import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evaluation.src.core.data_models import (
    Conversation,
    Dataset,
    Message,
    QAPair,
    SearchResult,
)
from evaluation.src.multimodal.raw_backend import MultimodalRAGAdapter
from evaluation.src.multimodal import runner
from evaluation.src.multimodal.runner import (
    attach_source_media,
    answer_metrics,
    retrieval_metrics,
    summarize,
)


def test_media_only_comes_from_selected_source_turns():
    conversation = Conversation(
        "scenario",
        [
            Message(
                "u",
                "User",
                "first",
                metadata={
                    "round_id": "D1:1",
                    "images": [{"image_id": "D1:IMG_001", "path": "first.png"}],
                },
            ),
            Message("a", "Assistant", "reply", metadata={"round_id": "D1:1"}),
            Message(
                "u",
                "User",
                "unselected",
                metadata={
                    "round_id": "D1:2",
                    "images": [{"image_id": "D1:IMG_002", "path": "second.png"}],
                },
            ),
        ],
    )
    result = SearchResult(
        "query",
        "scenario",
        [{"content": "hit", "metadata": {"start_turn": 0, "end_turn": 2}}],
    )
    images, rounds = attach_source_media(result, conversation)
    assert [image["image_id"] for image in images] == ["D1:IMG_001"]
    assert rounds == ["D1:1"]
    assert result.results[0]["metadata"]["round_ids"] == ["D1:1"]


def test_invalid_source_offsets_are_not_silently_accepted():
    result = SearchResult(
        "query", "scenario", [{"metadata": {"start_turn": -1, "end_turn": 3}}]
    )
    with pytest.raises(ValueError, match="offsets"):
        attach_source_media(result, Conversation("scenario", []))


def test_empty_clue_is_excluded_from_retrieval_averages():
    assert retrieval_metrics(["D1:1"], []) == {
        "recall": None,
        "precision": None,
        "hit": None,
    }
    rows = [
        {
            "category": "AR",
            "answer_metrics": {"exact_match": 1, "token_f1": 1},
            "retrieval_metrics": retrieval_metrics([], []),
        },
        {
            "category": "FR",
            "answer_metrics": {"exact_match": 1, "token_f1": 1},
            "retrieval_metrics": retrieval_metrics(["D1:1"], ["D1:1"]),
        },
    ]
    report = summarize(rows)
    assert report["overall"]["retrieval_eligible_questions"] == 1
    assert report["overall"]["retrieval_recall"] == 1
    assert report["by_category"]["AR"]["retrieval_recall"] is None


def test_answer_diagnostics_are_not_the_official_judge():
    assert answer_metrics("The red camera.", "red camera")["token_f1"] == 1
    assert answer_metrics("D1:IMG_001", "D1:IMG_002")["exact_match"] == 0
    assert "not the official" in summarize([])["protocol"]


def test_caption_is_rendered_into_raw_retrieval_without_pixel_payload():
    adapter = object.__new__(MultimodalRAGAdapter)
    message = Message(
        "u",
        "User",
        "my camera",
        metadata={
            "images": [
                {
                    "image_id": "D1:IMG_001",
                    "path": "private/path.png",
                    "caption": "a red camera",
                }
            ]
        },
    )
    rendered = adapter._format_message_line(message)
    assert "Image ID: D1:IMG_001" in rendered
    assert "a red camera" in rendered
    assert "private/path.png" not in rendered


@pytest.mark.asyncio
async def test_raw_pipeline_build_search_and_source_mapping(tmp_path):
    config = {
        "chunking": {
            "strategy": "token",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "expand_turns_on_display": True,
        },
        "embedding": {"api_key": "offline", "base_url": "http://127.0.0.1:1/v1"},
        "llm": {"api_key": "offline", "base_url": "http://127.0.0.1:1/v1"},
        "rerank": {"enabled": False},
        "search": {"embedding_top_k": 128, "rerank_top_k": 15},
    }
    adapter = MultimodalRAGAdapter(config, tmp_path)
    adapter._embed_texts = AsyncMock(
        side_effect=lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
    )
    conversation = Conversation(
        "scenario",
        [
            Message(
                "u",
                "User",
                "[Round D1:1] my camera",
                metadata={
                    "round_id": "D1:1",
                    "images": [
                        {
                            "image_id": "D1:IMG_001",
                            "path": "camera.png",
                            "caption": "a blue camera",
                        }
                    ],
                },
            )
        ],
    )
    try:
        index = await adapter.add([conversation])
        assert (
            "a blue camera"
            in index["conversations"]["scenario"]["chunks"][0]["content"]
        )
        result = await adapter.search("what color camera", "scenario", index)
        images, rounds = attach_source_media(result, conversation)
        assert images[0]["image_id"] == "D1:IMG_001"
        assert rounds == ["D1:1"]
        assert "Image ID: D1:IMG_001" in result.retrieval_metadata["formatted_context"]
    finally:
        await adapter.close()


def test_checkpoint_rejects_orphaned_results(tmp_path):
    (tmp_path / "results.jsonl").write_text('{"question_id":"old"}\n')
    with pytest.raises(ValueError, match="no run manifest"):
        runner._prepare_checkpoint(tmp_path, {"question_ids": ["new"]})


def test_checkpoint_rejects_mixed_settings_and_duplicate_ids(tmp_path):
    identity = {"question_ids": ["first"], "max_images": 10}
    runner._prepare_checkpoint(tmp_path, identity)
    with pytest.raises(ValueError, match="different data or settings"):
        runner._prepare_checkpoint(tmp_path, {**identity, "max_images": 1})
    (tmp_path / "results.jsonl").write_text('{"question_id":"first"}\n' * 2)
    with pytest.raises(ValueError, match="duplicate"):
        runner._prepare_checkpoint(tmp_path, identity)


@pytest.mark.asyncio
async def test_runner_exports_predictions_and_resumes_without_gold_in_search(
    tmp_path, monkeypatch
):
    image = {
        "image_id": "D1:IMG_001",
        "path": "history.png",
        "caption": "a blue camera",
        "caption_key": "scenario::D1:IMG_001",
    }
    conversation = Conversation(
        "scenario",
        [
            Message(
                "u",
                "User",
                "[Round D1:1] my camera",
                metadata={"round_id": "D1:1", "images": [image]},
            )
        ],
    )
    question = QAPair(
        "scenario:0",
        "Which camera?",
        "hidden-reference",
        "VS",
        ["D1:1"],
        {"conversation_id": "scenario", "session_ids": ["D1"], "question_images": []},
    )
    dataset = Dataset("memgallery", [conversation], [question])
    fake_backend = SimpleNamespace(
        add=AsyncMock(return_value={}),
        search=AsyncMock(
            return_value=SearchResult(
                "Which camera?",
                "scenario",
                [{"content": "evidence", "metadata": {"start_turn": 0, "end_turn": 1}}],
                {"formatted_context": "Image ID: D1:IMG_001"},
            )
        ),
        close=AsyncMock(),
    )
    fake_client = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(runner, "_dataset", lambda arguments: dataset)
    monkeypatch.setattr(
        runner, "_load_config", lambda path: {"vision": {"model": "offline"}}
    )
    monkeypatch.setattr(
        runner, "_vision_client", lambda config: (fake_client, {"model": "offline"})
    )
    monkeypatch.setattr(
        "evaluation.src.multimodal.raw_backend.MultimodalRAGAdapter",
        lambda *args, **kwargs: fake_backend,
    )
    reader = AsyncMock(
        return_value={"answer": "D1:IMG_001", "usage": {}, "image_ids": ["D1:IMG_001"]}
    )
    monkeypatch.setattr(runner, "read_with_images", reader)
    arguments = SimpleNamespace(
        data_root=tmp_path,
        config=None,
        output=tmp_path / "output",
        backend="raw-rag",
        allow_uncaptioned_history=False,
        limit_questions=None,
        caption_only=False,
        max_images=10,
    )
    await runner.run(arguments)
    await runner.run(arguments)
    assert reader.await_count == 1
    search_arguments = fake_backend.search.call_args
    assert "hidden-reference" not in repr(search_arguments)
    assert "session_ids" not in search_arguments.kwargs
    exported = json.loads(
        (arguments.output / "official_export/scenario.json").read_text()
    )
    assert exported[0]["system_answer"] == "D1:IMG_001"
    assert exported[0]["retrieved_ids"] == ["D1:1"]
