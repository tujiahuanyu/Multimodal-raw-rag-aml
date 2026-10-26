import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    status = {'model': 'Qwen/Qwen3-Reranker-4B', 'state': 'downloading'}
    status_path = arguments.output / 'download_status.json'
    status_path.write_text(json.dumps(status))
    started = time.monotonic()
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        snapshot_download(
            model_id=status['model'],
            local_dir=str(arguments.output),
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE'],
            max_workers=4,
        )
        print(
            'Official model download completed; validating local inference', flush=True
        )
        from .local_reranker import LocalQwenReranker

        reranker = LocalQwenReranker(
            {'model_path': str(arguments.output), 'batch_size': 4, 'max_length': 8192}
        )
        checks = []
        for query, positive, negative in [
            (
                'What is the capital of France?',
                'Paris is the capital of France.',
                'Berlin is the capital of Germany.',
            ),
            ('法国的首都是哪里？', '法国的首都是巴黎。', '德国的首都是柏林。'),
        ]:
            ranked = reranker.rank(
                query,
                [
                    {'label': 'positive', 'content': positive},
                    {'label': 'negative', 'content': negative},
                ],
                2,
            )
            checks.append(
                {
                    'positive_first': ranked[0]['label'] == 'positive',
                    'scores': [row['rerank_score'] for row in ranked],
                }
            )
        if not all(check['positive_first'] for check in checks):
            raise ValueError('Local model relevance sanity check failed')
        before = time.monotonic()
        candidates = [
            {
                'label': str(position),
                'content': (
                    'Paris is the capital of France. '
                    if position % 2 == 0
                    else 'Berlin is the capital of Germany. '
                )
                * 12,
            }
            for position in range(128)
        ]
        reranker.rank('What is the capital of France?', candidates, 15)
        status.update(
            {
                'state': 'load_test_passed',
                'checks': checks,
                'candidate_128_seconds': round(time.monotonic() - before, 2),
                'gpu_peak_allocated_mib': round(
                    reranker.torch.cuda.max_memory_allocated() / 1024 / 1024, 1
                ),
                'seconds': round(time.monotonic() - started, 2),
            }
        )
        reranker.close()
        (arguments.output / 'load_test_passed.json').write_text(
            json.dumps(status, indent=2)
        )
        print(json.dumps(status), flush=True)
    except BaseException as error:
        status.update(
            {'state': 'failed_or_interrupted', 'error_type': type(error).__name__}
        )
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
