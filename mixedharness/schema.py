"""A small JSON Schema subset: enough to constrain what an llm node returns and to describe task outputs.

Supported keywords: type (a name or a list), properties, required, additionalProperties (false only),
items, enum, const, minimum, maximum, minLength, maxLength, minItems, maxItems. Anything else is rejected at
schema-validation time so a typo cannot silently widen a contract.
"""
import math

TYPES = {'object': dict, 'array': list, 'string': str, 'number': (int, float), 'integer': int, 'boolean': bool, 'null': type(None)}
KEYWORDS = {'type', 'properties', 'required', 'additionalProperties', 'items', 'enum', 'const',
            'minimum', 'maximum', 'minLength', 'maxLength', 'minItems', 'maxItems', 'description', 'title'}


class SchemaError(ValueError):
    pass


def check_schema(schema, path='schema'):
    """Reject schemas that use keywords outside the subset."""
    if not isinstance(schema, dict):
        raise SchemaError(f'{path}: schema must be an object')
    unknown = set(schema) - KEYWORDS
    if unknown:
        raise SchemaError(f'{path}: unsupported keywords {sorted(unknown)}')
    if 'type' in schema:
        names = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        for name in names:
            if name not in TYPES:
                raise SchemaError(f'{path}: unknown type {name!r}')
    if 'properties' in schema:
        if not isinstance(schema['properties'], dict):
            raise SchemaError(f'{path}: properties must be an object')
        for key, sub in schema['properties'].items():
            check_schema(sub, f'{path}.properties.{key}')
    if 'items' in schema:
        check_schema(schema['items'], f'{path}.items')
    if 'additionalProperties' in schema and schema['additionalProperties'] is not False:
        raise SchemaError(f'{path}: additionalProperties may only be false')
    if 'required' in schema and (not isinstance(schema['required'], list) or not all(isinstance(k, str) for k in schema['required'])):
        raise SchemaError(f'{path}: required must be a list of strings')
    if 'enum' in schema and (not isinstance(schema['enum'], list) or not schema['enum']):
        raise SchemaError(f'{path}: enum must be a nonempty list')
    for key in ('minimum', 'maximum', 'minLength', 'maxLength', 'minItems', 'maxItems'):
        if key in schema and (isinstance(schema[key], bool) or not isinstance(schema[key], (int, float))):
            raise SchemaError(f'{path}: {key} must be a number')
    return schema


def _matches_type(value, name):
    if name == 'number':
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if name == 'integer':
        return isinstance(value, int) and not isinstance(value, bool)
    if name == 'boolean':
        return isinstance(value, bool)
    return isinstance(value, TYPES[name])


def validate(value, schema, path='$'):
    """Raise SchemaError describing the first violation; return value unchanged."""
    if 'type' in schema:
        names = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        if not any(_matches_type(value, name) for name in names):
            raise SchemaError(f'{path}: expected {" or ".join(names)}, got {type(value).__name__}')
    if 'const' in schema and value != schema['const']:
        raise SchemaError(f'{path}: must equal {schema["const"]!r}')
    if 'enum' in schema and value not in schema['enum']:
        raise SchemaError(f'{path}: not one of {schema["enum"]!r}')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 'minimum' in schema and value < schema['minimum']:
            raise SchemaError(f'{path}: below minimum {schema["minimum"]}')
        if 'maximum' in schema and value > schema['maximum']:
            raise SchemaError(f'{path}: above maximum {schema["maximum"]}')
    if isinstance(value, str):
        if 'minLength' in schema and len(value) < schema['minLength']:
            raise SchemaError(f'{path}: shorter than {schema["minLength"]}')
        if 'maxLength' in schema and len(value) > schema['maxLength']:
            raise SchemaError(f'{path}: longer than {schema["maxLength"]}')
    if isinstance(value, list):
        if 'minItems' in schema and len(value) < schema['minItems']:
            raise SchemaError(f'{path}: fewer than {schema["minItems"]} items')
        if 'maxItems' in schema and len(value) > schema['maxItems']:
            raise SchemaError(f'{path}: more than {schema["maxItems"]} items')
        if 'items' in schema:
            for index, item in enumerate(value):
                validate(item, schema['items'], f'{path}[{index}]')
    if isinstance(value, dict):
        for key in schema.get('required', []):
            if key not in value:
                raise SchemaError(f'{path}: missing required {key!r}')
        properties = schema.get('properties', {})
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f'{path}.{key}')
            elif schema.get('additionalProperties') is False:
                raise SchemaError(f'{path}: unexpected property {key!r}')
    return value


def minimal_instance(schema):
    """The smallest value that satisfies the schema: used by the mock LLM and to self-check schemas."""
    if 'const' in schema:
        return schema['const']
    if 'enum' in schema:
        return schema['enum'][0]
    names = schema.get('type', 'object')
    name = names[0] if isinstance(names, list) else names
    if name == 'object':
        return {key: minimal_instance(schema.get('properties', {}).get(key, {'type': 'string'})) for key in schema.get('required', [])}
    if name == 'array':
        count = int(schema.get('minItems', 0))
        return [minimal_instance(schema.get('items', {'type': 'string'})) for _ in range(count)]
    if name == 'string':
        return 'x' * int(schema.get('minLength', 0))
    if name == 'integer':
        return int(schema.get('minimum', 0))
    if name == 'number':
        return float(schema.get('minimum', 0))
    if name == 'boolean':
        return False
    return None
