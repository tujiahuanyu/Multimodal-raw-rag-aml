"""Run source-preserving multimodal retrieval on Mem-Gallery."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import re
import string
import time

from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.multimodal.media import describe_image, read_with_images
from evaluation.src.multimodal.memgallery import load_memgallery
from evaluation.src.utils.fingerprint import fingerprint, public_config


def attach_source_media(
    result: SearchResult, conversation: Conversation
) -> tuple[list, list]:
    """Map selected raw windows to their source turns without using QA evidence."""
    images = []
    round_ids = []
    seen_images = set()
    for hit in result.results:
        metadata = hit.setdefault("metadata", {})
        hit_images = list(metadata.get("images", []))
        hit_rounds = list(metadata.get("round_ids", []))
        start = metadata.get("start_turn")
        end = metadata.get("end_turn")
        if start is not None and end is not None:
            if not 0 <= int(start) <= int(end) <= len(conversation.messages):
                raise ValueError(
                    "Retrieved window contains invalid source-turn offsets"
                )
            for message in conversation.messages[int(start) : int(end)]:
                hit_rounds.append(message.metadata["round_id"])
                hit_images.extend(message.metadata.get("images", []))
        metadata["round_ids"] = list(dict.fromkeys(hit_rounds))
        metadata["images"] = list(
            {image["image_id"]: image for image in hit_images}.values()
        )
        round_ids.extend(metadata["round_ids"])
        for image in metadata["images"]:
            if image["image_id"] not in seen_images:
                images.append(image)
                seen_images.add(image["image_id"])
    return images, list(dict.fromkeys(round_ids))


def retrieval_metrics(retrieved_ids: list[str], evidence: list[str]) -> dict:
    expected = set(evidence)
    if not expected:
        return {"recall": None, "precision": None, "hit": None}
    retrieved = set(retrieved_ids)
    correct = len(expected & retrieved)
    return {
        "recall": correct / len(expected),
        "precision": correct / len(retrieved) if retrieved else 0.0,
        "hit": float(correct > 0),
    }


def _normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_metrics(prediction: str, reference: str) -> dict:
    predicted = _normalize(prediction).split()
    expected = _normalize(reference).split()
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not predicted or not expected:
        score = float(predicted == expected)
    else:
        score = 2 * overlap / (len(predicted) + len(expected))
    return {"exact_match": float(predicted == expected), "token_f1": score}


def summarize(rows: list[dict]) -> dict:
    categories = sorted({row["category"] for row in rows})

    def group_metrics(group):
        eligible = [
            row for row in group if row["retrieval_metrics"]["recall"] is not None
        ]
        return {
            "questions": len(group),
            "exact_match": sum(row["answer_metrics"]["exact_match"] for row in group)
            / len(group),
            "token_f1": sum(row["answer_metrics"]["token_f1"] for row in group)
            / len(group),
            "retrieval_eligible_questions": len(eligible),
            "retrieval_recall": (
                sum(row["retrieval_metrics"]["recall"] for row in eligible)
                / len(eligible)
                if eligible
                else None
            ),
            "retrieval_precision": (
                sum(row["retrieval_metrics"]["precision"] for row in eligible)
                / len(eligible)
                if eligible
                else None
            ),
            "retrieval_hit": (
                sum(row["retrieval_metrics"]["hit"] for row in eligible) / len(eligible)
                if eligible
                else None
            ),
        }

    return {
        "protocol": "Diagnostic EM/F1 and round retrieval metrics; not the official LLM judge",
        "overall": group_metrics(rows) if rows else {"questions": 0},
        "by_category": {
            category: group_metrics(
                [row for row in rows if row["category"] == category]
            )
            for category in categories
        },
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _load_captions(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(
        not isinstance(text, str) for text in value.values()
    ):
        raise ValueError(
            "Caption cache must be a JSON object mapping caption keys to strings"
        )
    return value


def _prepare_checkpoint(output: Path, identity: dict) -> list[dict]:
    manifest_path = output / "run_manifest.json"
    rows_path = output / "results.jsonl"
    if manifest_path.exists():
        existing_identity = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_identity != identity:
            raise ValueError(
                "Output directory belongs to different data or settings; select a new --output"
            )
    elif rows_path.exists():
        raise ValueError("Existing results have no run manifest; select a new --output")
    rows = (
        [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rows_path.exists()
        else []
    )
    question_ids = [row["question_id"] for row in rows]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Checkpoint contains duplicate question IDs")
    if not set(question_ids).issubset(identity["question_ids"]):
        raise ValueError("Checkpoint contains questions outside this run")
    write_json(manifest_path, identity)
    return rows


def _load_config(path: Path | None) -> dict:
    if path is None:
        raise ValueError("--config is required for caption generation and model runs")
    from dotenv import load_dotenv
    from evaluation.src.utils.config import load_yaml

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
    return load_yaml(str(path))


def _vision_client(config: dict):
    from openai import AsyncOpenAI

    options = config.get("vision", config.get("llm", {}))
    if not options.get("api_key") or not options.get("model"):
        raise ValueError(
            "Configure vision.api_key and vision.model before making VLM calls"
        )
    return (
        AsyncOpenAI(
            api_key=options["api_key"],
            base_url=options.get("base_url") or "https://api.openai.com/v1",
            max_retries=2,
            timeout=120,
        ),
        options,
    )


def _dataset(arguments):
    return load_memgallery(
        arguments.data_root,
        scenario=arguments.scenario,
        use_dataset_captions=arguments.use_dataset_captions,
        captions=_load_captions(arguments.captions),
    )


def validate(arguments) -> None:
    dataset = _dataset(arguments)
    history_images = [
        image
        for conversation in dataset.conversations
        for message in conversation.messages
        for image in message.metadata.get("images", [])
    ]
    question_images = [
        image
        for question in dataset.qa_pairs
        for image in question.metadata.get("question_images", [])
    ]
    report = {
        "conversations": len(dataset.conversations),
        "messages": sum(
            len(conversation.messages) for conversation in dataset.conversations
        ),
        "questions": len(dataset.qa_pairs),
        "history_images": len(history_images),
        "question_images": len(question_images),
        "history_images_with_captions": sum(
            bool(image.get("caption")) for image in history_images
        ),
        "categories": dict(Counter(question.category for question in dataset.qa_pairs)),
        "dataset_metadata": dataset.metadata,
        "api_calls": 0,
    }
    if arguments.output:
        write_json(arguments.output / "validation.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


async def generate_captions(arguments) -> None:
    if arguments.captions is None:
        raise ValueError(
            "Provide --captions as the destination for the generated caption cache"
        )
    dataset = load_memgallery(arguments.data_root, scenario=arguments.scenario)
    config = _load_config(arguments.config)
    client, options = _vision_client(config)
    captions = _load_captions(arguments.captions)
    manifest_path = arguments.captions.with_suffix(".manifest.json")
    identity = {
        "model": options["model"],
        "base_url": options.get("base_url"),
        "prompt_version": 1,
    }
    if captions and (
        not manifest_path.exists()
        or json.loads(manifest_path.read_text(encoding="utf-8")) != identity
    ):
        await client.close()
        raise ValueError(
            "Caption cache has a different or missing model manifest; use a new cache path"
        )
    write_json(manifest_path, identity)
    pending = []
    for conversation in dataset.conversations:
        for message in conversation.messages:
            for image in message.metadata.get("images", []):
                pending.append((image, message.content))
    for question in dataset.qa_pairs:
        for image in question.metadata.get("question_images", []):
            pending.append((image, ""))
    pending = [
        (image, context)
        for image, context in pending
        if image['caption_key'] not in captions
    ]
    if arguments.limit_images:
        pending = pending[: arguments.limit_images]
    workers = getattr(arguments, 'caption_workers', 1)
    if workers < 1:
        await client.close()
        raise ValueError('caption_workers must be positive')
    queue = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)
    processed = 0

    async def worker():
        nonlocal processed
        while True:
            try:
                image, context = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            caption = await describe_image(
                client,
                options["model"],
                Path(image["path"]),
                arguments.data_root.resolve(),
                context=context,
            )
            if not caption:
                raise RuntimeError(f"Empty caption for {image['caption_key']}")
            captions[image["caption_key"]] = caption
            write_json(arguments.captions, captions)
            processed += 1
            print(
                f"Captioned {processed}/{len(pending)}: {image['caption_key']}",
                flush=True,
            )

    tasks = [
        asyncio.create_task(worker()) for position in range(min(workers, len(pending)))
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()


async def run(arguments) -> None:
    dataset = _dataset(arguments)
    config = _load_config(arguments.config)
    output = arguments.output
    if output is None:
        raise ValueError("--output is required for model runs")
    if arguments.backend == "raw-rag":
        from evaluation.src.multimodal.raw_backend import MultimodalRAGAdapter

        missing = [
            image["caption_key"]
            for conversation in dataset.conversations
            for message in conversation.messages
            for image in message.metadata.get("images", [])
            if not image.get("caption")
        ]
        if missing and not arguments.allow_uncaptioned_history:
            raise ValueError(
                f"{len(missing)} historical images have no caption; run caption first, explicitly use dataset captions, or opt into --allow-uncaptioned-history"
            )
        backend = MultimodalRAGAdapter(config, output_dir=output / "index")
    else:
        from evaluation.src.multimodal.raganything_backend import RAGAnythingBackend

        backend = RAGAnythingBackend(
            config, output / "index", arguments.data_root.resolve()
        )
    client, options = _vision_client(config)
    questions = (
        dataset.qa_pairs[: arguments.limit_questions]
        if arguments.limit_questions
        else dataset.qa_pairs
    )
    identity = {
        "dataset": fingerprint(dataset),
        "config": public_config(config),
        "backend": arguments.backend,
        "include_history_images": not arguments.caption_only,
        "max_images": arguments.max_images,
        "question_ids": [question.question_id for question in questions],
        "protocol_version": 1,
    }
    manifest_path = output / "run_manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "results.jsonl"
    try:
        rows = _prepare_checkpoint(output, identity)
    except Exception:
        await client.close()
        await backend.close()
        raise
    completed = {row["question_id"] for row in rows}
    conversations = {
        conversation.conversation_id: conversation
        for conversation in dataset.conversations
    }
    query_caption_path = output / "query_captions.json"
    query_captions = _load_captions(query_caption_path)
    try:
        index = await backend.add(dataset.conversations)
        for question in questions:
            if question.question_id in completed:
                continue
            started = time.perf_counter()
            retrieval_query = question.question
            descriptions = []
            for image in question.metadata.get("question_images", []):
                key = image["caption_key"]
                caption = image.get("caption") or query_captions.get(key)
                if not caption:
                    caption = await describe_image(
                        client,
                        options["model"],
                        Path(image["path"]),
                        arguments.data_root.resolve(),
                    )
                    if not caption:
                        raise RuntimeError(f"Empty query image caption: {key}")
                    query_captions[key] = caption
                    write_json(query_caption_path, query_captions)
                descriptions.append(caption)
            if descriptions:
                retrieval_query += "\nQuery image descriptions:\n" + "\n".join(
                    descriptions
                )
            conversation_id = question.metadata["conversation_id"]
            result = await backend.search(
                retrieval_query,
                conversation_id,
                index,
                question_id=question.question_id,
            )
            images, retrieved_ids = attach_source_media(
                result, conversations[conversation_id]
            )
            context = result.retrieval_metadata.get("formatted_context") or "\n\n".join(
                hit["content"] for hit in result.results
            )
            reply = await read_with_images(
                client=client,
                model=options["model"],
                question=question.question,
                context=context,
                history_images=images,
                question_images=question.metadata.get("question_images", []),
                allowed_root=arguments.data_root.resolve(),
                max_images=arguments.max_images,
                max_tokens=int(options.get("max_tokens", 512)),
                temperature=float(options.get("temperature", 0)),
                category=question.category or "",
                include_history_images=not arguments.caption_only,
            )
            if not reply["answer"].strip():
                raise RuntimeError(f"Empty reader response for {question.question_id}")
            row = {
                "sample_id": conversation_id,
                "question_id": question.question_id,
                "session_id": question.metadata.get("session_ids", []),
                "question": question.question,
                "system_answer": reply["answer"],
                "original_answer": question.answer,
                "category": question.category,
                "retrieved_ids": retrieved_ids,
                "clue": question.evidence,
                "retrieval_metrics": retrieval_metrics(
                    retrieved_ids, question.evidence
                ),
                "answer_metrics": answer_metrics(reply["answer"], question.answer),
                "reader": {
                    key: value for key, value in reply.items() if key != "answer"
                },
                "elapsed_seconds": time.perf_counter() - started,
                "retrieval_query": retrieval_query,
                "formatted_context": context,
                "search_results": result.results,
            }
            with rows_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            rows.append(row)
            write_json(output / "summary.json", summarize(rows))
            print(
                f"Answered {len(rows)}/{len(questions)}: {question.question_id}",
                flush=True,
            )
        for conversation_id in conversations:
            write_json(
                output / "official_export" / f"{conversation_id}.json",
                [row for row in rows if row["sample_id"] == conversation_id],
            )
    finally:
        await client.close()
        await backend.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "caption", "run"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--use-dataset-captions", action="store_true")
    parser.add_argument("--allow-uncaptioned-history", action="store_true")
    parser.add_argument(
        "--backend", choices=["raw-rag", "rag-anything"], default="raw-rag"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--caption-only",
        action="store_true",
        help="Disable historical pixels; query images are still available to the reader",
    )
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument('--caption-workers', type=int, default=1)
    parser.add_argument("--limit-questions", type=int)
    arguments = parser.parse_args(argv)
    if arguments.max_images < 0:
        parser.error("--max-images must be non-negative")
    if arguments.caption_workers < 1:
        parser.error('--caption-workers must be positive')
    for name in ("limit_images", "limit_questions"):
        value = getattr(arguments, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.command == "validate":
        validate(arguments)
    else:
        asyncio.run(
            generate_captions(arguments)
            if arguments.command == "caption"
            else run(arguments)
        )


if __name__ == "__main__":
    main()
