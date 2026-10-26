import math
from pathlib import Path


PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = (
    'Given a web search query, retrieve relevant passages that answer the query'
)


class LocalQwenReranker:
    def __init__(self, options):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(options['model_path']).expanduser().resolve()
        if not path.is_dir():
            raise ValueError('Download the local reranker into model_path first')
        self.batch_size = int(options.get('batch_size', 4))
        self.max_length = int(options.get('max_length', 8192))
        if self.batch_size < 1 or self.max_length < 256:
            raise ValueError('Invalid local reranker batch size or context limit')
        self.device = options.get('device', 'cuda:0')
        if not torch.cuda.is_available():
            raise ValueError('This local Qwen configuration requires a CUDA GPU')
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, padding_side='left', local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                attn_implementation='sdpa',
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            .to(self.device)
            .eval()
        )
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.no_id = self.tokenizer.convert_tokens_to_ids('no')
        self.yes_id = self.tokenizer.convert_tokens_to_ids('yes')
        if self.no_id == self.yes_id:
            raise ValueError('Invalid local reranker yes/no token mapping')

    def rank(self, query, candidates, limit):
        if not candidates or limit < 1:
            return []
        texts = [
            '<Instruct>: '
            + INSTRUCTION
            + '\n<Query>: '
            + query
            + '\n<Document>: '
            + row['content']
            for row in candidates
        ]
        tokenized = self.tokenizer(texts, add_special_tokens=False, truncation=False)
        sequences = [
            self.prefix + tokens + self.suffix for tokens in tokenized['input_ids']
        ]
        if any(len(tokens) > self.max_length for tokens in sequences):
            raise ValueError(
                'A rerank pair exceeds max_length; no silent truncation is allowed'
            )
        scores = []
        with self.torch.inference_mode():
            for offset in range(0, len(sequences), self.batch_size):
                batch = self.tokenizer.pad(
                    {'input_ids': sequences[offset : offset + self.batch_size]},
                    padding=True,
                    return_tensors='pt',
                )
                batch = {name: value.to(self.device) for name, value in batch.items()}
                logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[
                    :, -1, :
                ]
                pair_logits = logits[:, [self.no_id, self.yes_id]].float()
                scores.extend(
                    self.torch.softmax(pair_logits, dim=1)[:, 1].cpu().tolist()
                )
        if len(scores) != len(candidates) or not all(
            math.isfinite(score) for score in scores
        ):
            raise ValueError('Local reranker produced invalid scores')
        ranked = [
            {**candidate, 'rerank_score': float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        return sorted(ranked, key=lambda row: row['rerank_score'], reverse=True)[:limit]

    def close(self):
        self.model = None
        self.torch.cuda.empty_cache()
