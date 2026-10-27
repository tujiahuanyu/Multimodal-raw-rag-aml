from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    api_token: str
    vision_url: str
    vision_key: str
    vision_model: str
    embedding_url: str
    embedding_key: str
    embedding_model: str
    rerank_url: str
    rerank_key: str
    rerank_model: str
    rerank_proxy_url: str = ''
    chunk_size: int = 256
    chunk_overlap: int = 32
    candidate_count: int = 128
    result_count: int = 15
    embedding_batch: int = 10
    concurrency: int = 2
    max_body_bytes: int = 128 * 1024 * 1024
    max_image_bytes: int = 10 * 1024 * 1024
    max_media_bytes: int = 30 * 1024 * 1024
    max_image_pixels: int = 100_000_000
    request_timeout: float = 120

    def __post_init__(self):
        if len(self.api_token) < 32:
            raise ValueError('AML_API_TOKEN must contain at least 32 characters')
        if self.vision_model not in {'gpt-4o-mini', 'gpt-4o-mini-2024-07-18'}:
            raise ValueError('The academic configuration requires gpt-4o-mini')
        if self.embedding_model != 'text-embedding-v4':
            raise ValueError('The academic configuration requires text-embedding-v4')
        for value in (self.vision_url, self.embedding_url, self.rerank_url):
            parsed = urlsplit(value)
            if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
                raise ValueError('Configure valid upstream HTTP(S) URLs')
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(
                    'Upstream URLs must not contain credentials or queries'
                )
        if not all((self.vision_key, self.embedding_key, self.rerank_key)):
            raise ValueError('Configure all three upstream API keys')
        if self.rerank_proxy_url:
            parsed = urlsplit(self.rerank_proxy_url)
            if (
                parsed.scheme not in {'http', 'https'}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.path not in {'', '/'}
            ):
                raise ValueError('Configure a valid HTTP(S) rerank proxy URL')
            try:
                parsed.port
            except ValueError:
                raise ValueError('Configure a valid rerank proxy port') from None
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError('Invalid window size or overlap')
        if not 1 <= self.result_count <= self.candidate_count:
            raise ValueError('Invalid retrieval counts')
        if not 1 <= self.embedding_batch <= 10 or self.concurrency < 1:
            raise ValueError('Invalid batch size or concurrency')

    def identity(self) -> str:
        values = asdict(self)
        for name in (
            'data_dir',
            'api_token',
            'vision_key',
            'embedding_key',
            'rerank_key',
            'rerank_proxy_url',
        ):
            values.pop(name)
        values['protocol_version'] = 2
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def endpoint(base: str, suffix: str) -> str:
    base = base.rstrip('/')
    suffix = '/' + suffix.lstrip('/')
    return base if base.endswith(suffix) else base + suffix


def load_settings(env_file: Path, data_dir: Path) -> Settings:
    if not env_file.is_file():
        raise ValueError('The private AML environment file is missing')
    load_dotenv(env_file, override=False)

    def required(name):
        value = os.environ.get(name, '').strip()
        if not value:
            raise ValueError(f'Configure {name} in the private AML environment')
        return value

    return Settings(
        data_dir=data_dir.resolve(),
        api_token=required('AML_API_TOKEN'),
        vision_url=endpoint(required('VLM_BASE_URL'), '/chat/completions'),
        vision_key=required('VLM_API_KEY'),
        vision_model=required('VLM_MODEL'),
        embedding_url=endpoint(required('VECTORIZE_BASE_URL'), '/embeddings'),
        embedding_key=required('VECTORIZE_API_KEY'),
        embedding_model=required('VECTORIZE_MODEL'),
        rerank_url=endpoint(
            required('RERANK_BASE_URL'), os.getenv('RERANK_ENDPOINT', '/v2/rerank')
        ),
        rerank_key=required('RERANK_API_KEY'),
        rerank_model=required('RERANK_MODEL'),
        rerank_proxy_url=os.getenv('RERANK_PROXY_URL', '').strip(),
    )
