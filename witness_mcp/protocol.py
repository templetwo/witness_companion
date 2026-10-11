"""Minimal MCP-style JSON-RPC 2.0 stdio subset, not full MCP compliance.

Implements initialize (protocolVersion 2024-11-05), tools/list, tools/call,
and silently accepts notifications/initialized. Tool results contain a single
text content block encoding a JSON object, plus isError. Requests are serial;
IDs are strings, integers or null. Initialize must precede tool requests.
Initialization offers our one version if a client requests another; the client
must decide whether to continue. No strict initialized-notification handshake.

Transport: UTF-8 newline-delimited JSON, with optional Content-Length byte
framing (CRLF headers, optional Content-Type). Responses use the request's
framing. Maximum message size is 1 MiB. Malformed framing closes the stream
after a parse error; malformed JSON in a complete frame allows recovery.
Batches, resources, prompts, cancellation, progress, subscriptions, pagination,
server requests, authentication and streaming results are not implemented.
Notifications never execute tools and never receive responses. Unknown tools
use -32602 (Invalid params); execution failures use traceable isError results.

The installed OpenClaw version has not been verified against this subset.
No claim of OpenClaw compatibility follows from the fake stdio tests.
"""

import json
from typing import BinaryIO

from .tools import InvalidParams, ToolRegistry, UnknownTool

PROTOCOL_VERSION = '2024-11-05'
MAX_MESSAGE_BYTES = 1024 * 1024


def error(request_id, code: int, message: str, data=None) -> dict:
    detail = {'code': code, 'message': message}
    if data is not None:
        detail['data'] = data
    return {'jsonrpc': '2.0', 'id': request_id, 'error': detail}


class Server:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.initialized = False

    def handle(self, request):
        if not isinstance(request, dict):
            return error(None, -32600, 'Invalid Request')
        request_id = request.get('id')
        if (request.get('jsonrpc') != '2.0'
                or not isinstance(request.get('method'), str)
                or not (request_id is None or type(request_id) in (str, int))):
            return error(None, -32600, 'Invalid Request')
        if 'id' not in request:
            return None
        params = request.get('params', {})
        if not isinstance(params, dict):
            return error(request_id, -32602, 'Invalid params', 'params must be an object')
        method = request['method']
        if method not in {'initialize', 'tools/list', 'tools/call'}:
            return error(request_id, -32601, 'Method not found')
        try:
            if method == 'initialize':
                client = params.get('clientInfo')
                if (not isinstance(params.get('protocolVersion'), str)
                        or not isinstance(params.get('capabilities'), dict)
                        or not isinstance(client, dict)
                        or not isinstance(client.get('name'), str)
                        or not isinstance(client.get('version'), str)):
                    raise InvalidParams('initialize needs protocolVersion, capabilities and clientInfo')
                self.initialized = True
                result = {'protocolVersion': PROTOCOL_VERSION,
                          'capabilities': {'tools': {'listChanged': False}},
                          'serverInfo': {'name': 'witness-mcp-scaffold', 'version': '0.1.0'}}
            elif not self.initialized:
                return error(request_id, -32002, 'Server not initialized')
            elif method == 'tools/list':
                if params:
                    raise InvalidParams('tools/list takes no parameters; pagination is unsupported')
                result = {'tools': self.registry.list_tools()}
            else:
                if set(params) - {'name', 'arguments'} or not isinstance(params.get('name'), str):
                    raise InvalidParams('tools/call needs a tool name and optional arguments object')
                result = self.registry.call(params['name'], params.get('arguments', {}))
        except UnknownTool:
            return error(request_id, -32602, 'Unknown tool', {'name': params['name']})
        except InvalidParams as exc:
            return error(request_id, -32602, 'Invalid params', str(exc))
        except Exception:
            return error(request_id, -32603, 'Internal error')
        return {'jsonrpc': '2.0', 'id': request_id, 'result': result}


def _reject_constant(value):
    raise ValueError('non-JSON numeric constant')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _read_message(stream: BinaryIO):
    line = stream.readline(MAX_MESSAGE_BYTES + 1)
    if not line:
        return None, False
    if len(line) > MAX_MESSAGE_BYTES:
        raise ValueError('message too large')
    if not line.lower().startswith(b'content-length:'):
        return line, False
    headers = {}
    header_bytes = 0
    while line not in (b'\r\n', b'\n'):
        header_bytes += len(line)
        if not line or header_bytes > 8192 or b':' not in line:
            raise ValueError('invalid header')
        key, value = line.split(b':', 1)
        key = key.strip().lower()
        if key in headers or key not in {b'content-length', b'content-type'}:
            raise ValueError('unsupported or duplicate header')
        headers[key] = value.strip()
        line = stream.readline(8193)
    raw_length = headers[b'content-length']
    if not raw_length.isdigit():
        raise ValueError('invalid content length')
    size = int(raw_length)
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError('invalid content length')
    body = stream.read(size)
    if len(body) != size:
        raise ValueError('truncated body')
    return body, True


def _write_message(stream: BinaryIO, response: dict, framed: bool):
    body = json.dumps(response, ensure_ascii=True, allow_nan=False).encode('utf-8')
    if framed:
        stream.write(('Content-Length: %d\r\n\r\n' % len(body)).encode('ascii') + body)
    else:
        stream.write(body + b'\n')
    stream.flush()


def serve(server: Server, input_stream: BinaryIO, output_stream: BinaryIO):
    """Process messages until EOF. Diagnostics must never go to stdout."""
    while True:
        try:
            body, framed = _read_message(input_stream)
        except ValueError:
            _write_message(output_stream, error(None, -32700, 'Parse error'), False)
            return
        if body is None:
            return
        try:
            request = json.loads(body.decode('utf-8'), parse_constant=_reject_constant,
                                 object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError, RecursionError):
            response = error(None, -32700, 'Parse error')
        else:
            response = server.handle(request)
        if response is not None:
            _write_message(output_stream, response, framed)
