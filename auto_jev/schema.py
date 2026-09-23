"""A small, explicit JSON Schema subset for LLM node outputs. Unknown keywords fail loudly."""
import math

SUPPORTED = {'type', 'properties', 'required', 'additionalProperties', 'items', 'enum',
             'minimum', 'maximum', 'minLength', 'maxLength', 'minItems', 'maxItems', 'description', 'title'}
TYPES = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'null': type(None)}


def _type_ok(value, name):
    if name == 'number':
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if name == 'integer':
        return isinstance(value, int) and not isinstance(value, bool)
    if name not in TYPES:
        raise ValueError(f'unsupported schema type {name!r}')
    return isinstance(value, TYPES[name])


def validate_json_schema(value, schema, path='$'):
    if not isinstance(schema, dict):
        raise ValueError(f'{path}: schema must be an object')
    unknown = set(schema) - SUPPORTED
    if unknown:
        raise ValueError(f'{path}: unsupported schema keywords {sorted(unknown)}')
    if 'type' in schema:
        names = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        if not any(_type_ok(value, name) for name in names):
            raise ValueError(f'{path}: expected type {schema["type"]}, got {type(value).__name__}')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: {value!r} not in enum')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 'minimum' in schema and value < schema['minimum']:
            raise ValueError(f'{path}: {value} < minimum {schema["minimum"]}')
        if 'maximum' in schema and value > schema['maximum']:
            raise ValueError(f'{path}: {value} > maximum {schema["maximum"]}')
    if isinstance(value, str):
        if 'minLength' in schema and len(value) < schema['minLength']:
            raise ValueError(f'{path}: shorter than minLength {schema["minLength"]}')
        if 'maxLength' in schema and len(value) > schema['maxLength']:
            raise ValueError(f'{path}: longer than maxLength {schema["maxLength"]}')
    if isinstance(value, list):
        if 'minItems' in schema and len(value) < schema['minItems']:
            raise ValueError(f'{path}: fewer than minItems {schema["minItems"]}')
        if 'maxItems' in schema and len(value) > schema['maxItems']:
            raise ValueError(f'{path}: more than maxItems {schema["maxItems"]}')
        if 'items' in schema:
            for index, item in enumerate(value):
                validate_json_schema(item, schema['items'], f'{path}[{index}]')
    if isinstance(value, dict):
        properties = schema.get('properties', {})
        for key in schema.get('required', []):
            if key not in value:
                raise ValueError(f'{path}: missing required {key!r}')
        for key, child in value.items():
            if key in properties:
                validate_json_schema(child, properties[key], f'{path}.{key}')
            elif schema.get('additionalProperties', True) is False:
                raise ValueError(f'{path}: unexpected property {key!r}')
    return value


def minimal_instance(schema, label='mock'):
    """Smallest value that satisfies the schema; used by the mock LLM."""
    if 'enum' in schema:
        return schema['enum'][0]
    names = schema.get('type', 'object')
    name = names[0] if isinstance(names, list) else names
    if name == 'object':
        return {key: minimal_instance(schema.get('properties', {}).get(key, {'type': 'string'}), label)
                for key in schema.get('required', [])}
    if name == 'array':
        count = schema.get('minItems', 0)
        return [minimal_instance(schema.get('items', {'type': 'string'}), f'{label} {index + 1}') for index in range(count)]
    if name == 'string':
        return label
    if name in ('number', 'integer'):
        return schema.get('minimum', 0)
    if name == 'boolean':
        return False
    if name == 'null':
        return None
    raise ValueError(f'unsupported schema type {name!r}')
