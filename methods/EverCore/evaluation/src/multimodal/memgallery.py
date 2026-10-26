"""Load Mem-Gallery histories without exposing QA annotations to retrieval."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from ..core.data_models import Conversation, Dataset, Message, QAPair


def _data_directory(data_root: Path) -> Path:
    root = data_root.expanduser().resolve()
    if (root / "dialog").is_dir():
        return root
    if (root / "data" / "dialog").is_dir():
        return root / "data"
    raise FileNotFoundError(
        f"Mem-Gallery dialogs not found under {root}; expected dialog/ or data/dialog/."
    )


def _image_path(value: str, source_file: Path, image_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid image path in {source_file}: {value!r}")
    local_path = Path(value)
    if PureWindowsPath(value).drive and not local_path.is_absolute():
        raise ValueError(f"Unsupported absolute image path in {source_file}: {value!r}")
    resolved = (source_file.parent / local_path).resolve()
    if not resolved.is_relative_to(image_root):
        raise ValueError(
            f"Image path {value!r} in {source_file.name} escapes image root {image_root}."
        )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Image {value!r} referenced by {source_file.name} does not exist at "
            f"{resolved}; download the dataset's data/image/ assets."
        )
    return resolved


def _list(value: Any, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"Mem-Gallery {label} must be a list.")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Mem-Gallery {label} must be a string.")
    return value


def _caption(
    caption_key: str,
    captions: dict[str, str],
    dataset_caption: Any = "",
    image_id: str | None = None,
) -> str:
    for key in (caption_key, image_id):
        if key is not None and key in captions:
            return _text(captions[key], f"caption for {key}")
    return _text(dataset_caption, f"caption for {caption_key}")


def _history_images(
    dialogue: dict,
    source_file: Path,
    image_root: Path,
    scenario_name: str,
    session_id: str,
    round_id: str,
    captions: dict[str, str],
    use_dataset_captions: bool,
    image_bindings: dict[str, tuple[str, str, str]],
) -> list[dict]:
    paths = _list(dialogue.get("input_image") or [], "input_image")
    image_ids = _list(dialogue.get("image_id") or [], "image_id")
    if len(paths) != len(image_ids):
        raise ValueError(
            f"Image path/ID count mismatch in {source_file.name}, round {round_id}."
        )
    dataset_captions = (
        _list(dialogue.get("image_caption") or [], "image_caption")
        if use_dataset_captions
        else []
    )
    images = []
    for image_index, (path_value, image_id) in enumerate(zip(paths, image_ids)):
        image_id = _text(image_id, f"image_id in round {round_id}")
        if not image_id:
            raise ValueError(f"Empty image_id in {source_file.name}, round {round_id}.")
        resolved = _image_path(path_value, source_file, image_root)
        if image_id in image_bindings:
            previous = image_bindings[image_id]
            raise ValueError(
                f"Duplicate image_id {image_id!r} in scenario {scenario_name}: "
                f"session {previous[0]}, round {previous[1]}, path {previous[2]} "
                f"conflicts with session {session_id}, round {round_id}, path {resolved}."
            )
        image_bindings[image_id] = (session_id, round_id, str(resolved))
        caption_key = f"{scenario_name}::{image_id}"
        dataset_caption = (
            dataset_captions[image_index] if image_index < len(dataset_captions) else ""
        )
        images.append(
            {
                "image_id": image_id,
                "path": str(resolved),
                "caption_key": caption_key,
                "caption": _caption(caption_key, captions, dataset_caption, image_id),
            }
        )
    return images


def _question_images(
    qa: dict,
    source_file: Path,
    image_root: Path,
    scenario_name: str,
    qa_index: int,
    captions: dict[str, str],
    use_dataset_captions: bool,
    image_bindings: dict[str, tuple[str, str, str]],
) -> list[dict]:
    raw_paths = qa.get("question_image")
    if not raw_paths:
        return []
    paths = raw_paths if isinstance(raw_paths, list) else [raw_paths]
    dataset_captions = qa.get("image_caption", "") if use_dataset_captions else ""
    if not isinstance(dataset_captions, list):
        dataset_captions = [dataset_captions]
    images = []
    for image_index, path_value in enumerate(paths):
        resolved = _image_path(path_value, source_file, image_root)
        caption_key = f"{scenario_name}::query:{qa_index}:{image_index}"
        dataset_caption = (
            dataset_captions[image_index] if image_index < len(dataset_captions) else ""
        )
        image = {
            "path": str(resolved),
            "caption_key": caption_key,
            "caption": _caption(caption_key, captions, dataset_caption),
        }
        matching_ids = [
            image_id
            for image_id, binding in image_bindings.items()
            if binding[2] == str(resolved)
        ]
        if len(matching_ids) == 1:
            image["image_id"] = matching_ids[0]
        images.append(image)
    return images


def load_memgallery(
    data_root: Path,
    scenario: str | None = None,
    use_dataset_captions: bool = False,
    captions: dict[str, str] | None = None,
) -> Dataset:
    """Load scenario-isolated histories and QA pairs; captions are opt-in."""
    data_directory = _data_directory(Path(data_root))
    dialog_directory = data_directory / "dialog"
    image_root = (data_directory / "image").resolve()
    if scenario is not None:
        if (
            not scenario
            or scenario in {".", ".."}
            or "/" in scenario
            or "\\" in scenario
            or ":" in scenario
        ):
            raise ValueError("Mem-Gallery scenario must be a single scenario filename.")
        filename = scenario if scenario.endswith(".json") else f"{scenario}.json"
        source_files = [dialog_directory / filename]
        if not source_files[0].is_file():
            raise FileNotFoundError(
                f"Mem-Gallery scenario not found: {source_files[0]}"
            )
    else:
        source_files = sorted(dialog_directory.glob("*.json"))
        if not source_files:
            raise FileNotFoundError(
                f"No Mem-Gallery scenario JSON files in {dialog_directory}."
            )
    supplied_captions = captions if captions is not None else {}
    if not isinstance(supplied_captions, dict):
        raise ValueError(
            "Mem-Gallery captions must map image IDs or caption keys to text."
        )
    caption_source = "none"
    if use_dataset_captions:
        caption_source = "dataset"
    if supplied_captions:
        caption_source = (
            "provided_with_dataset_fallback" if use_dataset_captions else "provided"
        )
    conversations = []
    qa_pairs = []
    for source_file in source_files:
        scenario_name = source_file.stem
        with source_file.open(encoding="utf-8-sig") as source:
            payload = json.load(source)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Mem-Gallery scenario must be a JSON object: {source_file}"
            )
        messages = []
        image_bindings = {}
        round_ids = set()
        sessions = _list(
            payload.get("multi_session_dialogues", []), "multi_session_dialogues"
        )
        for session in sessions:
            session_id = _text(session.get("session_id", ""), "session_id")
            session_date = _text(
                session.get("date", ""), f"date for session {session_id}"
            )
            try:
                timestamp = (
                    datetime.fromisoformat(session_date.replace("Z", "+00:00"))
                    if session_date
                    else None
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid session date {session_date!r} in {source_file.name}."
                ) from error
            for dialogue in _list(session.get("dialogues", []), "dialogues"):
                round_id = _text(dialogue.get("round", ""), "round")
                if not round_id or round_id in round_ids:
                    raise ValueError(
                        f"Missing or duplicate round ID {round_id!r} in {source_file.name}."
                    )
                round_ids.add(round_id)
                images = _history_images(
                    dialogue,
                    source_file,
                    image_root,
                    scenario_name,
                    session_id,
                    round_id,
                    supplied_captions,
                    use_dataset_captions,
                    image_bindings,
                )
                metadata = {
                    "round_id": round_id,
                    "session": session_id,
                    "session_date": session_date,
                }
                for role in ("user", "assistant"):
                    message_metadata = dict(metadata)
                    if role == "user":
                        message_metadata.update(
                            {
                                "images": images,
                                "image_ids": [image["image_id"] for image in images],
                            }
                        )
                    text = _text(dialogue.get(role, ""), f"{role} in round {round_id}")
                    messages.append(
                        Message(
                            sender_id=role,
                            sender_name=role.capitalize(),
                            content=f"[Round {round_id}]\n{text}",
                            timestamp=timestamp,
                            metadata=message_metadata,
                        )
                    )
        conversations.append(
            Conversation(
                conversation_id=scenario_name,
                messages=messages,
                metadata={
                    "source_file": str(source_file),
                    "caption_source": caption_source,
                },
            )
        )
        for qa_index, qa in enumerate(
            _list(payload.get("human-annotated QAs", []), "human-annotated QAs")
        ):
            question_images = _question_images(
                qa,
                source_file,
                image_root,
                scenario_name,
                qa_index,
                supplied_captions,
                use_dataset_captions,
                image_bindings,
            )
            session_ids = qa.get("session_id", [])
            if isinstance(session_ids, str):
                session_ids = [session_ids]
            session_ids = _list(session_ids, "QA session_id")
            point = _text(qa.get("point", ""), "QA point")
            qa_pairs.append(
                QAPair(
                    question_id=f"{scenario_name}:{qa_index}",
                    question=_text(qa.get("question", ""), "QA question"),
                    answer=_text(qa.get("answer", ""), "QA answer"),
                    category=point,
                    evidence=list(_list(qa.get("clue", []), "QA clue")),
                    metadata={
                        "conversation_id": scenario_name,
                        "session_ids": list(session_ids),
                        "question_images": question_images,
                        "question_caption": "\n".join(
                            image["caption"]
                            for image in question_images
                            if image["caption"]
                        ),
                        "point": point,
                        "local_qa_index": qa_index,
                    },
                )
            )
    return Dataset(
        dataset_name="memgallery",
        conversations=conversations,
        qa_pairs=qa_pairs,
        metadata={
            "data_root": str(Path(data_root).expanduser().resolve()),
            "data_directory": str(data_directory),
            "caption_source": caption_source,
            "caption_assisted": bool(supplied_captions) or use_dataset_captions,
            "use_dataset_captions": use_dataset_captions,
        },
    )
