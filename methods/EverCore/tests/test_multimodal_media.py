import base64
from types import SimpleNamespace

import pytest

from evaluation.src.multimodal import media
from evaluation.src.multimodal.media import (
    build_reader_content,
    describe_image,
    image_data_url,
    read_with_images,
)


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO7+B9kAAAAASUVORK5CYII="
)


def write_image(root, name, contents=PNG):
    path = root / name
    path.write_bytes(contents)
    return path


def test_image_data_url_uses_contents_instead_of_extension(tmp_path):
    path = write_image(tmp_path, "misleading.jpg")

    result = image_data_url(path, tmp_path)

    assert result.startswith("data:image/png;base64,")
    assert base64.b64decode(result.split(",", 1)[1]) == PNG


@pytest.mark.parametrize(
    ("contents", "mime_type"),
    [
        (b"\xff\xd8\xff\xe0" + b"jpeg", "image/jpeg"),
        (b"GIF89a" + b"gif", "image/gif"),
        (b"RIFF\x04\x00\x00\x00WEBP", "image/webp"),
    ],
)
def test_image_data_url_accepts_supported_signatures(tmp_path, contents, mime_type):
    path = write_image(tmp_path, "image.bin", contents)

    assert image_data_url(path, tmp_path).startswith(f"data:{mime_type};base64,")


@pytest.mark.parametrize("contents", [b"%PDF-1.7", b"", b"RIFFabcdefgh"])
def test_image_data_url_rejects_non_images(tmp_path, contents):
    path = write_image(tmp_path, "looks-valid.png", contents)

    with pytest.raises(ValueError, match="Unsupported image contents"):
        image_data_url(path, tmp_path)


def test_image_data_url_rejects_large_payloads(tmp_path, monkeypatch):
    path = write_image(tmp_path, "image.png")
    monkeypatch.setattr(media, "MAX_IMAGE_BYTES", len(PNG) - 1)

    with pytest.raises(ValueError, match="byte limit"):
        image_data_url(path, tmp_path)


def test_image_data_url_rejects_paths_outside_allowed_root(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = write_image(tmp_path, "outside.png")

    with pytest.raises(ValueError, match="outside allowed_root"):
        image_data_url(outside, allowed)
    with pytest.raises(ValueError, match="outside allowed_root"):
        image_data_url("../outside.png", allowed)


def test_image_data_url_resolves_paths_relative_to_allowed_root(tmp_path):
    write_image(tmp_path, "image.png")

    assert image_data_url("image.png", tmp_path).startswith("data:image/png;")


def test_image_data_url_rejects_symlink_escape(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = write_image(tmp_path, "outside.png")
    link = allowed / "link.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Creating symlinks is not supported in this environment")

    with pytest.raises(ValueError, match="outside allowed_root"):
        image_data_url(link, allowed)


def test_reader_keeps_query_and_history_labels_next_to_corresponding_images(tmp_path):
    query_path = write_image(tmp_path, "query.png")
    first_path = write_image(tmp_path, "first.png", PNG + b"first")
    second_path = write_image(tmp_path, "second.png", PNG + b"second")

    content, metadata = build_reader_content(
        "Which image?",
        "Historical conversation",
        [
            {"image_id": "original-A", "path": first_path},
            {"image_id": "original-B", "path": second_path},
        ],
        [{"path": query_path, "image_id": "QUERY-ID-MUST-NOT-LEAK"}],
        tmp_path,
        category="VS",
    )

    image_positions = [
        index for index, item in enumerate(content) if item["type"] == "image_url"
    ]
    assert len(image_positions) == 3
    assert [content[index - 1]["text"] for index in image_positions] == [
        "Question image 1 (reference, not a historical candidate)",
        "Historical image_id: original-A",
        "Historical image_id: original-B",
    ]
    assert [content[index]["image_url"]["url"] for index in image_positions] == [
        image_data_url(query_path, tmp_path),
        image_data_url(first_path, tmp_path),
        image_data_url(second_path, tmp_path),
    ]
    assert "QUERY-ID-MUST-NOT-LEAK" not in str(content)
    assert metadata["image_ids"] == ["original-A", "original-B"]
    assert metadata["question_image_count"] == 1
    assert metadata["history_image_count"] == 2


def test_reader_reserves_query_budget_before_truncating_history(tmp_path):
    paths = [write_image(tmp_path, f"image-{number}.png") for number in range(4)]

    content, metadata = build_reader_content(
        "question",
        "context",
        [
            {"image_id": "history-1", "path": paths[0]},
            {"image_id": "history-2", "path": paths[1]},
        ],
        [{"path": paths[2]}, {"path": paths[3]}],
        tmp_path,
        max_images=3,
    )

    assert metadata["image_ids"] == ["history-1"]
    assert metadata["question_image_count"] == 2
    assert metadata["truncated_count"] == 1
    assert sum(item["type"] == "image_url" for item in content) == 3


def test_reader_raises_instead_of_silently_dropping_query_images(tmp_path):
    path = write_image(tmp_path, "image.png")

    with pytest.raises(ValueError, match="rather than dropping question images"):
        build_reader_content(
            "question",
            "context",
            [],
            [{"path": path}, {"path": path}],
            tmp_path,
            max_images=1,
        )


def test_reader_deduplicates_history_by_id_or_resolved_path(tmp_path):
    first = write_image(tmp_path, "first.png")
    second = write_image(tmp_path, "second.png")
    third = write_image(tmp_path, "third.png")

    _, metadata = build_reader_content(
        "question",
        "context",
        [
            {"image_id": "A", "path": first},
            {"image_id": "A", "path": second},
            {"image_id": "B", "path": first},
            {"image_id": "C", "path": third},
        ],
        [],
        tmp_path,
    )

    assert metadata["image_ids"] == ["A", "C"]
    assert metadata["duplicate_count"] == 2
    assert metadata["truncated_count"] == 0


def test_no_history_images_ablation_still_sends_query_image(tmp_path):
    query_path = write_image(tmp_path, "query.png")

    content, metadata = build_reader_content(
        "question",
        "caption with historical image_id A",
        [{"image_id": "A", "path": tmp_path / "unavailable-history.png"}],
        [{"path": query_path}],
        tmp_path,
        max_images=1,
        include_history_images=False,
    )

    assert sum(item["type"] == "image_url" for item in content) == 1
    assert metadata["image_ids"] == []
    assert metadata["question_image_count"] == 1
    assert metadata["history_image_count"] == 0
    assert metadata["excluded_history_count"] == 1
    assert metadata["truncated_count"] == 0


def test_reader_zero_budget_can_return_text_only(tmp_path):
    path = write_image(tmp_path, "history.png")

    content, metadata = build_reader_content(
        "question",
        "context",
        [{"image_id": "A", "path": path}],
        [],
        tmp_path,
        max_images=0,
    )

    assert all(item["type"] == "text" for item in content)
    assert metadata["truncated_count"] == 1


@pytest.mark.parametrize(
    "category,expected",
    [
        ("VS", "return only the original historical image_id"),
        ("AR", "abstain with: I don't know"),
    ],
)
def test_reader_task_specific_instructions(tmp_path, category, expected):
    content, _ = build_reader_content(
        "question", "context", [], [], tmp_path, category=category
    )

    assert expected in content[0]["text"]


class RecordingCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" answer "))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2, total_tokens=9),
        )


def recording_client():
    completions = RecordingCompletions()
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def run_immediate(coroutine):
    with pytest.raises(StopIteration) as completed:
        coroutine.send(None)
    return completed.value.value


def test_describe_image_sends_context_and_original_image(tmp_path):
    path = write_image(tmp_path, "image.png")
    client, completions = recording_client()

    answer = run_immediate(
        describe_image(client, "vision-model", path, tmp_path, "actual turn")
    )

    assert answer == "answer"
    request = completions.calls[0]
    assert request["model"] == "vision-model"
    assert "actual turn" in request["messages"][1]["content"][0]["text"]
    assert request["messages"][1]["content"][1]["image_url"]["url"] == image_data_url(
        path, tmp_path
    )


def test_read_with_images_returns_usage_and_actual_image_metadata(tmp_path):
    path = write_image(tmp_path, "image.png")
    client, completions = recording_client()

    result = run_immediate(
        read_with_images(
            client,
            "vision-model",
            "question",
            "context",
            [{"image_id": "original-id", "path": path}],
            [],
            tmp_path,
            max_tokens=17,
            temperature=0.2,
            category="VS",
        )
    )

    assert result["answer"] == "answer"
    assert result["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }
    assert result["image_ids"] == ["original-id"]
    request = completions.calls[0]
    assert request["max_tokens"] == 17
    assert request["temperature"] == 0.2
    assert "comma-separated list" in request["messages"][0]["content"]
