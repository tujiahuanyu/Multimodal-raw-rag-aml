from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import hmac
import json
import logging
import re


_request = ContextVar('aml_diagnostic_request', default=None)
_logger = logging.getLogger(__name__)


def _fingerprint(value, secret):
    return hmac.new(
        secret.encode(), value.encode('utf-8', errors='surrogatepass'), hashlib.sha256
    ).hexdigest()[:24]


def _record(event, fields):
    try:
        _logger.warning(
            'AML_DIAGNOSTIC %s',
            json.dumps({'event': event, **(_request.get() or {}), **fields}),
        )
    except Exception:
        pass


@contextmanager
def add_trace(user_id, request_id, secret):
    token = _request.set(
        {'add_fingerprint': _fingerprint(json.dumps([user_id, request_id]), secret)}
    )
    try:
        _record('add_started', {})
        yield
    except Exception as error:
        code = getattr(error, 'code', '')
        safe_code = (
            code
            if isinstance(code, str)
            and re.fullmatch(
                r'(?:vision|embedding|rerank)_(?:http_[1-5][0-9]{2}|connection_failed|invalid_json|invalid_response|invalid_image_details)',
                code,
            )
            else 'other'
        )
        status = getattr(error, 'status', None)
        _record(
            'add_failed',
            {
                'error_code': safe_code,
                'http_status': status if type(status) is int else None,
            },
        )
        raise
    else:
        _record('add_completed', {})
    finally:
        _request.reset(token)


def _response_fields(data):
    fields = {'response_is_object': isinstance(data, dict)}
    if not isinstance(data, dict):
        return {**fields, 'validation_reason': 'response_not_object'}
    usage = data.get('usage')
    if isinstance(usage, dict):
        for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = usage.get(name)
            if type(value) is int and value >= 0:
                fields[name] = value
    choices = data.get('choices')
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {**fields, 'validation_reason': 'missing_or_invalid_choice'}
    choice = choices[0]
    reason = choice.get('finish_reason')
    fields['finish_reason'] = (
        reason
        if reason in ('stop', 'length', 'content_filter', 'tool_calls', 'function_call', None)
        else 'other'
    )
    message = choice.get('message')
    if not isinstance(message, dict):
        return {**fields, 'validation_reason': 'missing_or_invalid_message'}
    fields['refusal_present'] = bool(message.get('refusal'))
    content = message.get('content')
    fields['content_is_string'] = isinstance(content, str)
    fields['content_characters'] = len(content) if isinstance(content, str) else None
    if reason != 'stop':
        return {**fields, 'validation_reason': 'finish_reason_not_stop'}
    if not isinstance(content, str):
        return {**fields, 'validation_reason': 'content_not_string'}
    try:
        value = json.loads(content)
    except ValueError as error:
        fields['json_error_position'] = getattr(error, 'pos', None)
        return {**fields, 'validation_reason': 'content_json_parse_failed'}
    if not isinstance(value, dict):
        return {**fields, 'validation_reason': 'content_json_not_object'}
    if 'descriptions' not in value:
        return {**fields, 'validation_reason': 'descriptions_missing'}
    descriptions = value['descriptions']
    if not isinstance(descriptions, list):
        return {**fields, 'validation_reason': 'descriptions_not_array'}
    fields['description_count'] = len(descriptions)
    fields['description_characters'] = [
        len(text) if isinstance(text, str) else None for text in descriptions[:8]
    ]
    if len(descriptions) > 4:
        return {**fields, 'validation_reason': 'too_many_descriptions'}
    for text in descriptions:
        if not isinstance(text, str):
            return {**fields, 'validation_reason': 'description_not_string'}
        if not text.strip():
            return {**fields, 'validation_reason': 'description_empty'}
        if len(text) > 1200:
            return {**fields, 'validation_reason': 'description_too_long'}
    return {**fields, 'validation_reason': 'valid'}


def _safe_identifier(value):
    if isinstance(value, str) and re.fullmatch(
        r'(?:req[-_]|chatcmpl-|[0-9]{14})[A-Za-z0-9_.:-]{1,140}', value
    ):
        return value
    return None


def vision_response(response, attempt, max_tokens, data=None):
    try:
        if data is None:
            try:
                data = response.json()
            except ValueError:
                data = None
        fields = {
            'http_status': response.status_code,
            'http_attempt': attempt,
            'max_output_tokens': max_tokens,
            'provider_request_id': _safe_identifier(response.headers.get('x-request-id')),
            **_response_fields(data),
        }
        if isinstance(data, dict):
            fields['completion_id'] = _safe_identifier(data.get('id'))
        _record('vision_response', fields)
    except Exception:
        pass


def vision_transport_failure(attempt, error):
    name = type(error).__name__
    _record(
        'vision_transport_failed',
        {
            'http_attempt': attempt,
            'error_type': name
            if name in (
                'ReadTimeout', 'ConnectTimeout', 'WriteTimeout', 'PoolTimeout',
                'ConnectError', 'ReadError', 'WriteError', 'RemoteProtocolError',
            )
            else 'other',
        },
    )


def image_details_failure(data, url, secret):
    try:
        _record(
            'image_details_invalid',
            {'image_fingerprint': _fingerprint(url, secret), **_response_fields(data)},
        )
    except Exception:
        pass
