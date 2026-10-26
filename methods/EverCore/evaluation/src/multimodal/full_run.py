import argparse
import asyncio
import json
from pathlib import Path
import os
import time

from dotenv import load_dotenv

from . import runner


async def execute(arguments):
    load_dotenv(arguments.env_file, override=False)
    arguments.output.mkdir(parents=True, exist_ok=True)
    status_path = arguments.output / (
        'job_status.json'
        if arguments.stage == 'captions'
        else 'evaluation_job_status.json'
    )
    started = time.time()
    status = {'stage': arguments.stage, 'state': 'running', 'started_unix': started}
    runner.write_json(status_path, status)
    options = argparse.Namespace(
        data_root=arguments.data_root,
        scenario=None,
        config=arguments.config,
        captions=arguments.output / 'captions.json',
        use_dataset_captions=False,
        allow_uncaptioned_history=False,
        backend='raw-rag',
        output=arguments.output / 'evaluation',
        caption_only=False,
        max_images=10,
        limit_images=None,
        limit_questions=None,
        caption_workers=arguments.caption_workers,
    )
    try:
        if arguments.stage == 'captions':
            await runner.generate_captions(options)
            status['state'] = 'captions_complete_waiting_for_rerank_quota'
        else:
            config = (
                runner._load_config(arguments.config)
                if arguments.config is not None
                else {}
            )
            local = config.get('rerank', {}).get('provider') == 'local_qwen3'
            if not local and not arguments.confirm_full_rerank_quota:
                raise ValueError(
                    'Confirm enough authorized Cohere quota for all 1711 questions before evaluation'
                )
            if getattr(arguments, 'wait_for_prerequisites', False):
                await wait_for_prerequisites(
                    arguments, options, config, status, status_path
                )
            await runner.run(options)
            status['state'] = 'evaluation_complete'
    except BaseException as error:
        status['state'] = 'failed_or_interrupted'
        status['error_type'] = type(error).__name__
        raise
    finally:
        status['elapsed_seconds'] = round(time.time() - started, 2)
        if options.captions.exists():
            status['cached_captions'] = len(json.loads(options.captions.read_text()))
        runner.write_json(status_path, status)


async def wait_for_prerequisites(arguments, options, config, status, status_path):
    from .memgallery import load_memgallery

    dataset = load_memgallery(arguments.data_root)
    keys = {
        image['caption_key']
        for conversation in dataset.conversations
        for message in conversation.messages
        for image in message.metadata.get('images', [])
    }
    keys.update(
        image['caption_key']
        for question in dataset.qa_pairs
        for image in question.metadata.get('question_images', [])
    )
    rerank = config.get('rerank', {})
    model_marker = (
        Path(rerank['model_path']) / rerank['ready_marker']
        if rerank.get('ready_marker')
        else None
    )
    deadline = time.monotonic() + arguments.max_wait_seconds
    while True:
        cached = runner._load_captions(options.captions)
        missing = len(keys - cached.keys())
        model_ready = model_marker is None or model_marker.is_file()
        status.update(
            {
                'state': 'waiting_for_prerequisites',
                'missing_captions': missing,
                'model_ready': model_ready,
            }
        )
        runner.write_json(status_path, status)
        if missing == 0 and model_ready:
            status['state'] = 'running_evaluation'
            runner.write_json(status_path, status)
            return
        for filename, incomplete in [
            ('captions.pid', missing > 0),
            ('reranker-download.pid', not model_ready),
        ]:
            pid_file = arguments.output / filename
            if incomplete and pid_file.is_file():
                try:
                    os.kill(int(pid_file.read_text().strip()), 0)
                except ProcessLookupError:
                    raise RuntimeError(
                        'A prerequisite job stopped before completion'
                    ) from None
        if time.monotonic() >= deadline:
            raise TimeoutError(
                'Timed out waiting for captions or local model validation'
            )
        await asyncio.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['captions', 'evaluate'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, default=Path('.env.aml'))
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('evaluation/config/systems/memgallery_aml_eval.yaml'),
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--caption-workers', type=int, default=2)
    parser.add_argument('--confirm-full-rerank-quota', action='store_true')
    parser.add_argument('--wait-for-prerequisites', action='store_true')
    parser.add_argument('--max-wait-seconds', type=int, default=43200)
    arguments = parser.parse_args()
    if arguments.caption_workers < 1:
        parser.error('--caption-workers must be positive')
    asyncio.run(execute(arguments))


if __name__ == '__main__':
    main()
