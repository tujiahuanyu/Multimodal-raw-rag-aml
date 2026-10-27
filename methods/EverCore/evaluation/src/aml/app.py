import asyncio
from contextlib import asynccontextmanager
import hmac
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .contracts import AddRequest, SearchRequest
from .engine import Engine
from .providers import Providers, ServiceError
from .store import Store


class Gate:
    def __init__(self, app, settings):
        self.app = app
        self.settings = settings
        self.capacity = asyncio.Semaphore(settings.concurrency)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] == '/health':
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope['headers']}
        authorization = headers.get(b'authorization', b'').decode('latin1')
        token = headers.get(b'x-api-key', b'').decode('latin1')
        if authorization.startswith(('Token ', 'Bearer ')):
            token = authorization.split(' ', 1)[1]
        if not hmac.compare_digest(token.encode(), self.settings.api_token.encode()):
            await JSONResponse({'error': 'unauthorized'}, status_code=401)(
                scope, receive, send
            )
            return
        acquired = False
        try:
            try:
                await asyncio.wait_for(self.capacity.acquire(), timeout=0.05)
                acquired = True
            except asyncio.TimeoutError:
                await JSONResponse(
                    {'error': 'capacity_exceeded'},
                    status_code=429,
                    headers={'Retry-After': '1'},
                )(scope, receive, send)
                return
            try:
                length = int(headers.get(b'content-length', b'0'))
                if length < 0:
                    raise ValueError
            except ValueError:
                await JSONResponse(
                    {'error': 'invalid_content_length'}, status_code=400
                )(scope, receive, send)
                return
            if length > self.settings.max_body_bytes:
                await JSONResponse({'error': 'body_too_large'}, status_code=413)(
                    scope, receive, send
                )
                return
            body = bytearray()
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                body.extend(message.get('body', b''))
                if len(body) > self.settings.max_body_bytes:
                    await JSONResponse({'error': 'body_too_large'}, status_code=413)(
                        scope, receive, send
                    )
                    return
                if not message.get('more_body', False):
                    break
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {
                        'type': 'http.request',
                        'body': bytes(body),
                        'more_body': False,
                    }
                return await receive()

            await self.app(scope, replay, send)
        finally:
            if acquired:
                self.capacity.release()


def create_app(settings, providers=None):
    @asynccontextmanager
    async def lifespan(application):
        upstream = providers or Providers(settings)
        store = None
        try:
            store = Store(settings.data_dir, settings.identity())
            application.state.engine = Engine(settings, store, upstream)
            yield
        finally:
            if store is not None:
                store.close()
            await upstream.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(Gate, settings=settings)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exception: ServiceError):
        return JSONResponse({'error': exception.code}, status_code=exception.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exception: RequestValidationError):
        details = [
            {'location': list(row['loc']), 'type': row['type']}
            for row in exception.errors()
        ]
        return JSONResponse(
            {'error': 'invalid_request', 'details': details}, status_code=422
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exception: Exception):
        logging.getLogger(__name__).error(
            'Request failed: %s', type(exception).__name__
        )
        return JSONResponse({'error': 'internal_error'}, status_code=500)

    @app.get('/health')
    async def health():
        return {'status': 'ok', 'version': 'raw-rag-aml-v1.3-topk'}

    @app.post('/v1/memories/add')
    async def add(payload: AddRequest):
        return await app.state.engine.add(payload)

    @app.post('/v1/memories/search')
    async def search(payload: SearchRequest):
        return await app.state.engine.search(payload)

    return app
