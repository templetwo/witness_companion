# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:32:40 EDT
"""Five tool schemas, strict argument validation and traceable results."""

from copy import deepcopy
import json
import math
import uuid

from .services import WitnessServices, timestamp, utc_now


class InvalidParams(ValueError):
    pass


class UnknownTool(ValueError):
    pass


def object_schema(properties: dict, required=()) -> dict:
    return {'type': 'object', 'properties': properties,
            'required': list(required), 'additionalProperties': False}


TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 10000, 'pattern': r'\S'}
TOOL_SCHEMAS = [
    {'name': 'witness_snapshot',
     'description': 'Request a camera image with capture/receive timestamps; report unknown capture age honestly.',
     'inputSchema': object_schema({'max_age_seconds': {
         'type': 'number', 'minimum': 0, 'maximum': 3600, 'default': 5}})},
    {'name': 'witness_speak', 'description': 'Request speech; fake mode records without playback.',
     'inputSchema': object_schema({'text': TEXT}, ['text'])},
    {'name': 'witness_status', 'description': 'Report mic, camera, backend, controller and armed state.',
     'inputSchema': object_schema({})},
    {'name': 'witness_propose_note',
     'description': 'Append a pending note only to the host-configured local JSON file.',
     'inputSchema': object_schema({'text': TEXT}, ['text'])},
    {'name': 'robot_request_action',
     'description': ('Live actions: stop {}, drive {direction,duty:1..25,duration_seconds:0.05..0.5}, '
                     'joints {positions:{configured_name:pulse},duration_seconds:0.1..1}, '
                     'led {on_seconds,off_seconds,repeat}, buzzer adds frequency_hz, '
                     'rgb/sonar_rgb {index:1..2,red,green,blue}, display {text}. '
                     'Use witness_status for configured limits. Motion requires local arming; '
                     'SDK success does not verify physical execution. Fake-only legacy move: nonce, expires_at '
                     '(timezone-aware ISO 8601), speed [-1,1], duration_seconds (0,5]. '
                     'Stop is always accepted. No tool can arm the guard.'),
     'inputSchema': object_schema({
         'name': {'type': 'string', 'minLength': 1, 'maxLength': 100, 'pattern': r'\S'},
         'params': {'type': 'object', 'description': 'Action-specific guard parameters.'}},
         ['name', 'params'])},
]


class ToolRegistry:
    def __init__(self, services: WitnessServices):
        self.services = services

    def list_tools(self) -> list:
        return deepcopy(TOOL_SCHEMAS)

    def call(self, name: str, arguments: dict) -> dict:
        schema = next((t['inputSchema'] for t in TOOL_SCHEMAS if t['name'] == name), None)
        if schema is None:
            raise UnknownTool(name)
        if not isinstance(arguments, dict):
            raise InvalidParams('arguments must be an object')
        if set(arguments) - schema['properties'].keys():
            raise InvalidParams('unexpected tool arguments')
        if any(key not in arguments for key in schema['required']):
            raise InvalidParams('missing required tool arguments')
        for key, value in arguments.items():
            field = schema['properties'][key]
            if field['type'] == 'string':
                if (not isinstance(value, str) or not value.strip()
                        or len(value) > field['maxLength']):
                    raise InvalidParams(key + ' must be a nonempty bounded string')
            elif field['type'] == 'object' and not isinstance(value, dict):
                raise InvalidParams(key + ' must be an object')
            elif field['type'] == 'number':
                if (type(value) not in (int, float)
                        or not field['minimum'] <= value <= field['maximum']
                        or not math.isfinite(value)):
                    raise InvalidParams(key + ' must be a finite number between 0 and 3600')
        image_data = None
        trace = {'trace_id': str(uuid.uuid4()), 'issued_at': timestamp(utc_now()),
                 'backend_provenance': {'backend_id': 'unknown', 'kind': 'none',
                                        'model': None, 'where_computed': 'unknown'}}
        try:
            provenance = self.services.backend_provenance()
            if (not isinstance(provenance, dict)
                    or not isinstance(provenance.get('backend_id'), str)
                    or provenance.get('kind') not in {
                        'jetson_local', 'studio_local', 'studio_gateway', 'cloud', 'none'}
                    or not isinstance(provenance.get('where_computed'), str)
                    or not (provenance.get('model') is None
                            or isinstance(provenance.get('model'), str))):
                raise ValueError('invalid backend provenance')
            trace['backend_provenance'] = {key: provenance[key] for key in
                                         ('backend_id', 'kind', 'model', 'where_computed')}
            if name == 'witness_snapshot':
                payload = self.services.snapshot(arguments.get('max_age_seconds', 5))
            elif name == 'witness_speak':
                payload = self.services.speak(arguments['text'])
            elif name == 'witness_status':
                payload = self.services.status()
            elif name == 'witness_propose_note':
                payload = self.services.propose_note(arguments['text'], trace['trace_id'],
                                                     trace['issued_at'])
            else:
                payload = self.services.request_action(arguments['name'], arguments['params'])
            if not isinstance(payload, dict) or type(payload.get('ok')) is not bool:
                raise ValueError('invalid service result')
            image_data = payload.pop('image_base64', None)
            result = dict(payload, **trace)
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        except Exception:
            # Do not expose adapter exceptions, file contents or credentials.
            result = dict(trace, ok=False, reason='service_failure')
            encoded = json.dumps(result, ensure_ascii=False)
        content = [{'type': 'text', 'text': encoded}]
        if result['ok'] and image_data:
            content.append({'type': 'image', 'mimeType': 'image/jpeg', 'data': image_data})
        return {'content': content, 'isError': not result['ok']}
