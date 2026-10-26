import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pytest

from evaluation.src.multimodal.memgallery import load_memgallery


def _payload():
    return {
        "character_profile": {"name": "PROFILE_ANNOTATION_SECRET"},
        "multi_session_dialogues": [
            {
                "session_id": "D1",
                "date": "2024-06-17",
                "dialogues": [
                    {
                        "round": "D1:3",
                        "user": "Here are the two robots.",
                        "assistant": "The second one has wheels.",
                        "image_id": ["D1:IMG_001", "D1:IMG_002"],
                        "input_image": [
                            "../image/Robots/D1_IMG_001.jpg",
                            "../image/Robots/D1_IMG_002.jpg",
                        ],
                        "image_caption": ["DATASET_CAPTION_ONE", "DATASET_CAPTION_TWO"],
                    }
                ],
            },
            {
                "session_id": "D2",
                "date": "2024-07-07",
                "dialogues": [
                    {
                        "round": "D2:1",
                        "user": "The second robot is my favorite.",
                        "assistant": "",
                    }
                ],
            },
        ],
        "human-annotated QAs": [
            {
                "point": "VS",
                "question": "Which robot matches the attached image?",
                "question_image": "../image/Robots/QA_IMG_001.png",
                "image_caption": "DATASET_QUERY_CAPTION",
                "answer": "D1:IMG_002",
                "session_id": ["D1"],
                "clue": ["D1:3"],
            },
            {
                "point": "AR",
                "question": "QA_QUESTION_ANNOTATION_SECRET",
                "answer": "QA_ANSWER_ANNOTATION_SECRET",
                "session_id": ["D2"],
                "clue": [],
            },
        ],
    }


def _write_scenario(data_directory: Path, payload=None, name="Robots"):
    dialog_directory = data_directory / "dialog"
    dialog_directory.mkdir(parents=True, exist_ok=True)
    image_directory = data_directory / "image" / "Robots"
    image_directory.mkdir(parents=True, exist_ok=True)
    for filename in ("D1_IMG_001.jpg", "D1_IMG_002.jpg", "QA_IMG_001.png"):
        (image_directory / filename).write_bytes(b"image fixture")
    source_file = dialog_directory / f"{name}.json"
    source_file.write_text(json.dumps(payload or _payload()), encoding="utf-8")
    return source_file


@pytest.mark.parametrize("snapshot_root", [False, True])
def test_load_real_schema_and_both_data_root_layouts(tmp_path, snapshot_root):
    data_directory = tmp_path / "data" if snapshot_root else tmp_path
    _write_scenario(data_directory)

    dataset = load_memgallery(tmp_path)

    assert dataset.dataset_name == "memgallery"
    assert dataset.metadata["data_directory"] == str(data_directory.resolve())
    assert dataset.metadata["caption_source"] == "none"
    assert not dataset.metadata["caption_assisted"]
    assert len(dataset.conversations) == 1
    conversation = dataset.conversations[0]
    assert conversation.conversation_id == "Robots"
    assert len(conversation.messages) == 4
    user, assistant = conversation.messages[:2]
    assert user.sender_id == "user"
    assert assistant.sender_id == "assistant"
    assert user.timestamp == datetime(2024, 6, 17)
    assert "D1:3" in user.content and "D1:3" in assistant.content
    assert user.metadata["image_ids"] == ["D1:IMG_001", "D1:IMG_002"]
    assert len(user.metadata["images"]) == 2
    assert user.metadata["images"][0]["caption_key"] == "Robots::D1:IMG_001"
    assert user.metadata["images"][0]["caption"] == ""
    assert user.metadata["session"] == "D1"
    assert user.metadata["session_date"] == "2024-06-17"
    assert "images" not in assistant.metadata
    assert "image_ids" not in assistant.metadata
    assert conversation.messages[-1].content == "[Round D2:1]\n"
    qa = dataset.qa_pairs[0]
    assert qa.question_id == "Robots:0"
    assert qa.category == "VS"
    assert qa.answer == "D1:IMG_002"
    assert qa.evidence == ["D1:3"]
    assert qa.metadata["session_ids"] == ["D1"]
    assert qa.metadata["conversation_id"] == "Robots"
    assert qa.metadata["local_qa_index"] == 0
    assert qa.metadata["question_caption"] == ""
    assert qa.metadata["question_images"] == [
        {
            "path": str((data_directory / "image/Robots/QA_IMG_001.png").resolve()),
            "caption_key": "Robots::query:0:0",
            "caption": "",
        }
    ]
    assert dataset.qa_pairs[1].category == "AR"
    assert dataset.qa_pairs[1].evidence == []
    assert len(conversation.messages) == 4


def test_annotations_and_query_images_never_enter_history(tmp_path):
    source_file = _write_scenario(tmp_path)
    original = load_memgallery(tmp_path)
    history = json.dumps([asdict(item) for item in original.conversations], default=str)

    for forbidden in (
        "PROFILE_ANNOTATION_SECRET",
        "QA_QUESTION_ANNOTATION_SECRET",
        "QA_ANSWER_ANNOTATION_SECRET",
        "DATASET_QUERY_CAPTION",
        "QA_IMG_001.png",
        "human-annotated QAs",
        '"clue"',
    ):
        assert forbidden not in history

    modified = _payload()
    modified["character_profile"] = {"name": "DIFFERENT_PROFILE"}
    modified["human-annotated QAs"][0]["answer"] = "DIFFERENT_GOLD"
    modified["human-annotated QAs"][0]["clue"] = ["GOLD_CLUE_SECRET"]
    source_file.write_text(json.dumps(modified), encoding="utf-8")

    assert load_memgallery(tmp_path).conversations == original.conversations


def test_caption_sources_are_explicit_and_scoped_keys_take_priority(tmp_path):
    _write_scenario(tmp_path)
    captions = {
        "D1:IMG_001": "Shared image caption",
        "Robots::D1:IMG_001": "Scoped image caption",
        "Robots::query:0:0": "Query pixels only caption",
    }

    provided = load_memgallery(tmp_path, captions=captions)
    images = provided.conversations[0].messages[0].metadata["images"]
    assert [image["caption"] for image in images] == ["Scoped image caption", ""]
    assert (
        provided.qa_pairs[0].metadata["question_caption"] == "Query pixels only caption"
    )
    assert (
        provided.qa_pairs[0].metadata["question_images"][0]["caption"]
        == "Query pixels only caption"
    )
    assert provided.metadata["caption_source"] == "provided"
    assert provided.metadata["caption_assisted"]

    assisted = load_memgallery(tmp_path, use_dataset_captions=True, captions=captions)
    assert assisted.metadata["caption_source"] == "provided_with_dataset_fallback"
    assert (
        assisted.conversations[0].messages[0].metadata["images"][1]["caption"]
        == "DATASET_CAPTION_TWO"
    )

    dataset_captions = load_memgallery(tmp_path, use_dataset_captions=True)
    assert dataset_captions.metadata["caption_source"] == "dataset"
    assert dataset_captions.metadata["caption_assisted"]
    assert (
        dataset_captions.qa_pairs[0].metadata["question_caption"]
        == "DATASET_QUERY_CAPTION"
    )


def test_scenarios_are_isolated_and_can_share_image_ids(tmp_path):
    _write_scenario(tmp_path)
    _write_scenario(tmp_path, name="Other")

    dataset = load_memgallery(
        tmp_path,
        captions={"Robots::D1:IMG_001": "First", "Other::D1:IMG_001": "Second"},
    )

    assert [item.conversation_id for item in dataset.conversations] == [
        "Other",
        "Robots",
    ]
    assert [
        item.messages[0].metadata["images"][0]["caption"]
        for item in dataset.conversations
    ] == ["Second", "First"]
    assert [item.question_id for item in dataset.qa_pairs] == [
        "Other:0",
        "Other:1",
        "Robots:0",
        "Robots:1",
    ]
    assert len(load_memgallery(tmp_path, scenario="Robots.json").conversations) == 1


def test_duplicate_image_ids_across_sessions_raise_with_context(tmp_path):
    payload = _payload()
    second_round = payload["multi_session_dialogues"][1]["dialogues"][0]
    second_round.update(
        {"image_id": ["D1:IMG_001"], "input_image": ["../image/Robots/D1_IMG_002.jpg"]}
    )
    _write_scenario(tmp_path, payload)

    with pytest.raises(ValueError, match="Duplicate image_id.*session D1.*session D2"):
        load_memgallery(tmp_path)


@pytest.mark.parametrize("target", ["history", "query"])
@pytest.mark.parametrize("absolute_path", [False, True])
def test_images_cannot_escape_image_root(tmp_path, target, absolute_path):
    payload = _payload()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    path_value = str(outside) if absolute_path else "../outside.png"
    if target == "history":
        payload["multi_session_dialogues"][0]["dialogues"][0]["input_image"][
            0
        ] = path_value
    else:
        payload["human-annotated QAs"][0]["question_image"] = path_value
    _write_scenario(tmp_path, payload)

    with pytest.raises(ValueError, match="escapes image root"):
        load_memgallery(tmp_path)


def test_image_symlink_cannot_escape_image_root(tmp_path):
    source_file = _write_scenario(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    link = tmp_path / "image/Robots/link.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symbolic links is unavailable on this host")
    payload = _payload()
    payload["human-annotated QAs"][0]["question_image"] = "../image/Robots/link.png"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes image root"):
        load_memgallery(tmp_path)


def test_missing_image_error_explains_download_location(tmp_path):
    _write_scenario(tmp_path)
    (tmp_path / "image/Robots/D1_IMG_002.jpg").unlink()

    with pytest.raises(FileNotFoundError, match="D1_IMG_002.*download.*data/image"):
        load_memgallery(tmp_path)


def test_image_ids_must_align_with_all_image_paths(tmp_path):
    payload = _payload()
    payload["multi_session_dialogues"][0]["dialogues"][0]["image_id"] = ["D1:IMG_001"]
    _write_scenario(tmp_path, payload)

    with pytest.raises(ValueError, match="Image path/ID count mismatch.*D1:3"):
        load_memgallery(tmp_path)


def test_query_image_uses_its_own_caption_key_and_retains_known_image_id(tmp_path):
    payload = _payload()
    payload["human-annotated QAs"][0][
        "question_image"
    ] = "../image/Robots/D1_IMG_001.jpg"
    _write_scenario(tmp_path, payload)

    dataset = load_memgallery(
        tmp_path,
        captions={
            "Robots::D1:IMG_001": "History caption",
            "Robots::query:0:0": "Query caption",
        },
    )

    image = dataset.qa_pairs[0].metadata["question_images"][0]
    assert image["image_id"] == "D1:IMG_001"
    assert image["caption"] == "Query caption"
    assert len(dataset.conversations[0].messages[0].metadata["images"]) == 2


def test_multiple_query_images_have_independent_caption_keys(tmp_path):
    payload = _payload()
    payload["human-annotated QAs"][0]["question_image"] = [
        "../image/Robots/QA_IMG_001.png",
        "../image/Robots/D1_IMG_002.jpg",
    ]
    _write_scenario(tmp_path, payload)

    dataset = load_memgallery(
        tmp_path,
        captions={
            "Robots::query:0:0": "First query image",
            "Robots::query:0:1": "Second query image",
        },
    )

    assert (
        dataset.qa_pairs[0].metadata["question_caption"]
        == "First query image\nSecond query image"
    )
    assert len(dataset.qa_pairs[0].metadata["question_images"]) == 2


@pytest.mark.parametrize(
    "scenario", ["../Robots", "nested/Robots", "nested\\Robots", "C:Robots", "..", ""]
)
def test_scenario_selection_rejects_path_traversal(tmp_path, scenario):
    _write_scenario(tmp_path)

    with pytest.raises(ValueError, match="single scenario filename"):
        load_memgallery(tmp_path, scenario=scenario)


def test_missing_dataset_paths_raise_helpful_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="expected dialog/ or data/dialog/"):
        load_memgallery(tmp_path)
    (tmp_path / "dialog").mkdir()
    with pytest.raises(FileNotFoundError, match="No Mem-Gallery scenario JSON"):
        load_memgallery(tmp_path)
    with pytest.raises(FileNotFoundError, match="scenario not found"):
        load_memgallery(tmp_path, scenario="missing")


def test_loading_does_not_mutate_supplied_captions(tmp_path):
    _write_scenario(tmp_path)
    captions = {"Robots::D1:IMG_001": "Only supplied pixels"}
    original = deepcopy(captions)

    load_memgallery(tmp_path, captions=captions)

    assert captions == original
