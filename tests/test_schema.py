import pytest

from auto_jev.schema import minimal_instance, validate_json_schema

PERSON = {'type': 'object', 'required': ['name', 'age'], 'additionalProperties': False,
          'properties': {'name': {'type': 'string', 'minLength': 1},
                         'age': {'type': 'integer', 'minimum': 0, 'maximum': 150},
                         'tags': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 2},
                         'mood': {'type': 'string', 'enum': ['good', 'bad']}}}


def test_valid_object_passes_and_is_returned():
    value = {'name': 'a', 'age': 3, 'tags': ['x'], 'mood': 'good'}
    assert validate_json_schema(value, PERSON) is value


@pytest.mark.parametrize('bad', [
    {'name': 'a'}, {'name': '', 'age': 3}, {'name': 'a', 'age': -1}, {'name': 'a', 'age': 3.5},
    {'name': 'a', 'age': 3, 'tags': ['x', 'y', 'z']}, {'name': 'a', 'age': 3, 'mood': 'meh'},
    {'name': 'a', 'age': 3, 'extra': 1}, 'not an object', {'name': 'a', 'age': True},
])
def test_invalid_values_raise(bad):
    with pytest.raises(ValueError):
        validate_json_schema(bad, PERSON)


def test_type_list_and_null():
    assert validate_json_schema(None, {'type': ['string', 'null']}) is None
    with pytest.raises(ValueError):
        validate_json_schema(1, {'type': ['string', 'null']})


def test_unsupported_keyword_is_rejected_loudly():
    with pytest.raises(ValueError, match='unsupported'):
        validate_json_schema({}, {'type': 'object', 'patternProperties': {}})


def test_minimal_instance_satisfies_schema():
    instance = minimal_instance(PERSON)
    assert validate_json_schema(instance, PERSON) == instance
    assert instance == {'name': 'mock', 'age': 0}
    assert minimal_instance({'type': 'array', 'items': {'type': 'string'}, 'minItems': 3}) == ['mock 1', 'mock 2', 'mock 3']
    assert minimal_instance({'type': 'string', 'enum': ['x', 'y']}) == 'x'
