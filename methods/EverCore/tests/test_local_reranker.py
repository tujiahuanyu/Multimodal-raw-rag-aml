from types import SimpleNamespace

import pytest

from evaluation.src.multimodal.local_reranker import LocalQwenReranker


def ranker():
    torch = pytest.importorskip('torch')
    instance = object.__new__(LocalQwenReranker)
    instance.torch = torch
    instance.device = 'cpu'
    instance.batch_size = 1
    instance.max_length = 8192
    instance.prefix = [7]
    instance.suffix = [8]
    instance.no_id = 0
    instance.yes_id = 1
    instance.calls = []

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == {'add_special_tokens': False, 'truncation': False}
            return {
                'input_ids': [
                    [2 if '<Document>: relevant' in text else 1] for text in texts
                ]
            }

        def pad(self, batch, **kwargs):
            return {'input_ids': torch.tensor(batch['input_ids'])}

    class Model:
        def __call__(self, input_ids, use_cache, logits_to_keep):
            assert use_cache is False
            assert logits_to_keep == 1
            instance.calls.append(input_ids)
            signal = torch.where(input_ids[:, 1] == 2, 2.0, -2.0)
            return SimpleNamespace(
                logits=torch.stack([-signal, signal], dim=1).unsqueeze(1)
            )

    instance.tokenizer = Tokenizer()
    instance.model = Model()
    return instance


def test_local_yes_no_scores_are_ranked_without_mutating_candidates():
    instance = ranker()
    candidates = [{'content': 'unrelated'}, {'content': 'relevant'}]
    results = instance.rank('query', candidates, 1)
    assert results[0]['content'] == 'relevant'
    assert results[0]['rerank_score'] == pytest.approx(0.98201379)
    assert len(instance.calls) == 2
    assert all('rerank_score' not in candidate for candidate in candidates)


def test_local_pairs_are_not_silently_truncated():
    instance = ranker()
    instance.max_length = 2
    with pytest.raises(ValueError, match='no silent truncation'):
        instance.rank('query', [{'content': 'relevant'}], 1)
    assert not instance.calls


def test_empty_local_candidates_do_not_run_the_model():
    instance = ranker()
    assert instance.rank('query', [], 1) == []
    assert not instance.calls
