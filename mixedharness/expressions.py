"""The restricted expression language used by spec fields (`state`, `prompt`, `output`, `memory`, ...).

An expression is a single Python expression over three names, `obs`, `nodes` and `memory`, restricted to
JSON values: literals, dict/list construction, indexing and slicing, arithmetic, comparisons, boolean
operators, the conditional expression, and calls to a fixed set of functions. No attribute access, no
comprehensions, no string formatting, no names other than the three inputs and the functions below.
Every intermediate value must be JSON-compatible; the evaluator checks that as it goes.
"""
import ast
import math
import statistics

MAX_LENGTH = 12_000
MAX_NODES = 512
MAX_ITEMS = 100_000


def _std(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _last(seq, default=None):
    return seq[-1] if seq else default


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _clip(value, low, high):
    return min(high, max(low, value))


def _column(rows, key):
    return [row[key] for row in rows]


def _text(value):
    return value if isinstance(value, str) else ('' if value is None else str(value))


FUNCTIONS = {
    'len': len, 'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
    'mean': statistics.mean, 'median': statistics.median, 'std': _std,
    'last': _last, 'get': _get, 'clip': _clip, 'column': _column, 'text': _text,
    'sorted': sorted, 'any': any, 'all': all,
}
NAMES = ('obs', 'nodes', 'memory')

_ALLOWED = (ast.Expression, ast.Constant, ast.Dict, ast.List, ast.Tuple, ast.Subscript, ast.Slice, ast.Name,
            ast.Load, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.UnaryOp,
            ast.USub, ast.UAdd, ast.Not, ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.Eq, ast.NotEq, ast.Lt,
            ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.IfExp, ast.Call)


class ExpressionError(ValueError):
    pass


def check_json(value, depth=0):
    """Raise ExpressionError unless value is a finite, JSON-compatible tree. Returns the value."""
    if depth > 64:
        raise ExpressionError('value nests too deeply')
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExpressionError('non-finite number')
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_ITEMS:
            raise ExpressionError('sequence too long')
        for item in value:
            check_json(item, depth + 1)
        return list(value) if isinstance(value, tuple) else value
    if isinstance(value, dict):
        if len(value) > MAX_ITEMS:
            raise ExpressionError('object too large')
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExpressionError('object keys must be strings')
            check_json(item, depth + 1)
        return value
    raise ExpressionError(f'value of type {type(value).__name__} is not JSON')


def parse(expression):
    """Parse and vet an expression; returns the AST. Raises ExpressionError."""
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionError('expression must be a nonempty string')
    if len(expression) > MAX_LENGTH:
        raise ExpressionError('expression too long')
    try:
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ExpressionError(f'invalid expression syntax: {exc.msg if hasattr(exc, "msg") else exc}') from None
    walked = list(ast.walk(tree))
    if len(walked) > MAX_NODES:
        raise ExpressionError('expression too complex')
    for node in walked:
        if not isinstance(node, _ALLOWED):
            raise ExpressionError(f'disallowed syntax: {type(node).__name__}')
        if isinstance(node, ast.Name) and node.id not in NAMES and node.id not in FUNCTIONS:
            raise ExpressionError(f'unknown name: {node.id}')
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise ExpressionError('only the built-in functions may be called, positionally')
        if isinstance(node, ast.Dict) and any(key is None for key in node.keys):
            raise ExpressionError('dict unpacking is not allowed')
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'nodes'
                and not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str))):
            raise ExpressionError("nodes[...] must use a literal node id so dependencies can be inferred")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (str, int, float, bool, type(None))):
            raise ExpressionError('only JSON literals are allowed')
    return tree


def references(expression):
    """Node ids read through nodes['id'] with a literal key. Used to infer dependencies."""
    found = []
    for node in ast.walk(parse(expression)):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'nodes'
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            if node.slice.value not in found:
                found.append(node.slice.value)
    return found


def evaluate(expression, context):
    """Evaluate against {'obs': ..., 'nodes': {...}, 'memory': {...}}. Returns a JSON value."""
    tree = parse(expression)
    for name in NAMES:
        if name not in context:
            raise ExpressionError(f'context lacks {name}')

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in FUNCTIONS:
                raise ExpressionError('a function is not a value')
            return context[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return check_json([visit(item) for item in node.elts])
        if isinstance(node, ast.Dict):
            return check_json({visit(key): visit(value) for key, value in zip(node.keys, node.values)})
        if isinstance(node, ast.Slice):
            return slice(*(visit(part) if part is not None else None for part in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.Subscript):
            container = visit(node.value)
            index = visit(node.slice)
            if isinstance(index, slice):
                if not isinstance(container, (list, str)):
                    raise ExpressionError('slicing a non-sequence')
                return container[index]
            if isinstance(container, dict):
                if not isinstance(index, str):
                    raise ExpressionError('object keys must be strings')
                if index not in container:
                    raise ExpressionError(f'missing key: {index}')
                return container[index]
            if isinstance(container, (list, str)):
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ExpressionError('sequence index must be an integer')
                try:
                    return container[index]
                except IndexError:
                    raise ExpressionError('index out of range') from None
            raise ExpressionError('indexing a scalar')
        if isinstance(node, ast.IfExp):
            return visit(node.body) if visit(node.test) else visit(node.orelse)
        if isinstance(node, ast.UnaryOp):
            operand = visit(node.operand)
            if isinstance(node.op, ast.Not):
                return not operand
            _number(operand)
            return -operand if isinstance(node.op, ast.USub) else +operand
        if isinstance(node, ast.BoolOp):
            value = None
            for child in node.values:
                value = visit(child)
                if isinstance(node.op, ast.And) and not value:
                    return value
                if isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            return _binary(node.op, left, right)
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = visit(comparator)
                if not _compare(op, left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            args = [visit(arg) for arg in node.args]
            try:
                return check_json(FUNCTIONS[node.func.id](*args))
            except ExpressionError:
                raise
            except (TypeError, ValueError, statistics.StatisticsError, ZeroDivisionError) as exc:
                raise ExpressionError(f'{node.func.id}: {exc}') from None
        raise ExpressionError(f'unsupported syntax: {type(node).__name__}')

    return check_json(visit(tree))


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExpressionError('arithmetic on a non-number')
    return value


def _binary(op, left, right):
    if isinstance(op, ast.Add):
        if isinstance(left, str) and isinstance(right, str):
            if len(left) + len(right) > MAX_ITEMS:
                raise ExpressionError('string too long')
            return left + right
        if isinstance(left, list) and isinstance(right, list):
            if len(left) + len(right) > MAX_ITEMS:
                raise ExpressionError('list too long')
            return left + right
        return _number(left) + _number(right)
    if isinstance(op, ast.Mult):
        if isinstance(left, (str, list)) and isinstance(right, int) and not isinstance(right, bool):
            if len(left) * max(right, 0) > MAX_ITEMS:
                raise ExpressionError('repetition too long')
            return left * right
        return _number(left) * _number(right)
    left, right = _number(left), _number(right)
    try:
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
    except ZeroDivisionError:
        raise ExpressionError('division by zero') from None
    raise ExpressionError('unsupported operator')


def _compare(op, left, right):
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Is):
        return left is right
    if isinstance(op, ast.IsNot):
        return left is not right
    if isinstance(op, (ast.In, ast.NotIn)):
        if not isinstance(right, (list, str, dict)):
            raise ExpressionError('membership test on a scalar')
        try:
            inside = left in right
        except TypeError:
            raise ExpressionError('unhashable membership test') from None
        return inside if isinstance(op, ast.In) else not inside
    try:
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
    except TypeError:
        raise ExpressionError('ordering incomparable values') from None
    raise ExpressionError('unsupported comparison')
