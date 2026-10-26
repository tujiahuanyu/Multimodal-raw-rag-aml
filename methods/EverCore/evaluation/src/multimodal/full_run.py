import argparse
import asyncio
import json
from pathlib import Path
import time

from dotenv import load_dotenv

from . import runner


async def execute(arguments):
    load_dotenv(arguments.env_file, override=False)
    arguments.output.mkdir(parents=True, exist_ok=True)
    status_path = arguments.output / 'job_status.json'
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
            if not arguments.confirm_full_rerank_quota:
                raise ValueError(
                    'Confirm enough authorized Cohere quota for all 1711 questions before evaluation'
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
    arguments = parser.parse_args()
    if arguments.caption_workers < 1:
        parser.error('--caption-workers must be positive')
    asyncio.run(execute(arguments))


if __name__ == '__main__':
    main()
