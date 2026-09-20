"""Parse the model's Python, judge every node, and say why when it refuses.

Inputs:  the source of one generated tool
Outputs: a Verdict - whether it may run, and the repair message if it may not

Nothing here talks to a model and nothing here trusts a string. The source
becomes a syntax tree and every node is checked against a policy that allows
rather than forbids, so a construct nobody thought about is refused by default
instead of admitted by default.

The source is NOT rewritten. A SQL guard has to re-emit its query from the
checked tree, because a SQL parser models statements approximately and
anything it did not understand must not survive. Python's `ast` is not like
that: every construct has a node type, and the walk below is default-deny over
node types. Re-emitting would also cost the point of the project - `ast.unparse`
deletes every comment, and `generated/*.py` is meant to be opened and read. So
the source reaches disk exactly as written, and a parse round-trip at the end
proves the tree the checks walked is the whole program.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from src import config, protocol

# Node types a generated tool may use. Absent by design: ClassDef (a class body
# runs code through metaclass= and __init_subclass__, which the shape check
# never looks at), the async family, Global and Nonlocal. A future grammar
# addition is refused until somebody adds it here on purpose.
ALLOWED_NODES = frozenset("""
Module FunctionDef Return Assign AugAssign AnnAssign For While If With withitem
Raise Try TryStar ExceptHandler Assert Import ImportFrom alias Expr Pass Break
Continue Delete BoolOp NamedExpr BinOp UnaryOp Lambda IfExp Dict Set ListComp
SetComp DictComp GeneratorExp Compare Call FormattedValue JoinedStr Constant
Attribute Subscript Starred Name List Tuple Slice arguments arg keyword
comprehension Load Store Del And Or Add Sub Mult MatMult Div Mod Pow LShift
RShift BitOr BitXor BitAnd FloorDiv Invert Not UAdd USub Eq NotEq Lt LtE Gt GtE
Is IsNot In NotIn Match match_case MatchValue MatchSingleton MatchSequence
MatchMapping MatchClass MatchStar MatchAs MatchOr
""".split())

# An allowlist, not a list of bad ones: exec, eval, compile, open, input and
# __import__ are covered by absence, which is stronger than enumerating them.
# getattr and friends are absent on purpose - they are the bypass a name check
# cannot see, because getattr(obj, "e" + "val") contains no forbidden name at
# all. hasattr stays: it returns a bool and hands back no object. print stays,
# because a tool nobody can debug is worse than one that prints.
ALLOWED_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "hasattr", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "oct", "ord",
    "pow", "print", "range", "repr", "reversed", "round", "set", "slice", "sorted",
    "str", "sum", "tuple", "type", "zip", "ValueError", "TypeError", "KeyError",
    "IndexError", "ZeroDivisionError", "OverflowError", "ArithmeticError",
    "StopIteration", "Exception",
})

_NODE_REASONS = {
    "ClassDef": "a class is not allowed here; write plain functions",
    "AsyncFunctionDef": "there is no event loop in the sandbox; write a normal def",
    "Await": "there is no event loop in the sandbox",
    "Global": "global is not allowed; pass the value as an argument instead",
    "Nonlocal": "nonlocal is not allowed; pass the value as an argument instead",
    "Yield": "run() must return a value, not yield one",
}

_COOKIE = re.compile(r"coding[:=]\s*([-\w.]+)")


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether one module may run, and why not when it may not."""

    ok: bool
    source: str = ""
    reason: str = ""
    layer: str = ""         # size|syntax|names|shape|statements|imports|attributes|calls|sentinel


def _block(layer: str, reason: str) -> Verdict:
    return Verdict(ok=False, reason=reason, layer=layer)


def _is_literal(node: ast.AST) -> bool:
    """A module-level constant must be a literal, not something computed."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_literal(i) for i in [*node.keys, *node.values] if i)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_literal(node.operand)
    return False


def _bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    """(names that came from an allowed import, names bound to a value).

    An attribute chain is only as trustworthy as whatever it hangs off:
    `math.floor` is fine and `sys.modules` is not, and the only difference is
    which name the tool actually imported.
    """
    modules: set[str] = set()
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.FunctionDef):
            values.add(node.name)
            args = node.args
            values.update(a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs])
            values.update(a.arg for a in (args.vararg, args.kwarg) if a)
        elif isinstance(node, ast.Lambda):
            values.update(a.arg for a in [*node.args.posonlyargs, *node.args.args])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            values.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            values.add(node.name)
    return modules, values


def _check_names(tree: ast.Module) -> str:
    """Node types, dunders, bare excepts, and the names the runner reserves."""
    for node in ast.walk(tree):
        kind = type(node).__name__
        if kind not in ALLOWED_NODES:
            return _NODE_REASONS.get(kind, f"{kind} is not allowed here.")
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            return (f"the decorator on line {node.lineno} runs at definition time, "
                    "which is the one moment nothing is supposed to run.")
        name = (node.id if isinstance(node, ast.Name) else
                node.attr if isinstance(node, ast.Attribute) else
                node.name if isinstance(node, ast.FunctionDef) else "")
        line = getattr(node, "lineno", 0)
        if name == "__name__":
            # Worth its own message: models write `if __name__ == "__main__":`
            # by reflex, and the generic dunder answer costs a repair attempt
            # plus a whole re-send of the source.
            return (f"remove the __main__ block on line {line}. This module is "
                    "imported and called, never run as a script, and your checks "
                    "belong in the `tests` field of write_tool.")
        if name.startswith("__") and name.endswith("__"):
            return (f"'{name}' on line {line} reaches the interpreter's own "
                    "internals. Names starting and ending with two underscores "
                    "are never needed to compute an answer.")
        if name.startswith(protocol.RESERVED_PREFIX):
            return f"'{name}' is reserved by the runner. Rename it."
        if isinstance(node, ast.ExceptHandler):
            caught = node.type
            if caught is None or (isinstance(caught, ast.Name)
                                  and caught.id == "BaseException"):
                return (f"the bare except on line {node.lineno} would swallow the "
                        "sandbox's own errors. Catch what you expect, such as "
                        "'except ValueError:'.")
    return ""


def _check_shape(tree: ast.Module) -> str:
    """Exactly one top-level run(**kwargs). Helper defs are welcome."""
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if names.count("run") != 1:
        return ("the module must define exactly one top-level function named "
                f"'run'; found {', '.join(names) or 'none'}.")
    run = next(n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "run")
    args = run.args
    if (args.args or args.posonlyargs or args.kwonlyargs or args.vararg
            or args.defaults or args.kw_defaults or args.kwarg is None):
        return "run must take **kwargs and nothing else - 'def run(**kwargs):'."
    return ""


def _check_top_level(tree: ast.Module) -> str:
    """Only imports, defs and literal constants may sit at the top level."""
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue                                       # the module docstring
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is None or _is_literal(node.value):
                continue
            return ("a module-level constant must be a literal. Write the table "
                    "out as a nested tuple rather than building it with a call.")
        if isinstance(node, (ast.Assert, ast.Expr)):
            return (f"top-level {type(node).__name__} on line {node.lineno} runs "
                    "the moment the module loads. Your checks belong in the "
                    "`tests` field of write_tool, where they are run for you.")
        return (f"top-level {type(node).__name__} on line {node.lineno} runs the "
                "moment the module loads. Move it inside run() or a helper.")
    return ""


def _check_imports(tree: ast.Module) -> str:
    """Every import must name an allowed module, and name it plainly."""
    allowed = ", ".join(sorted(config.ALLOWED_IMPORTS))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                return f"relative import on line {node.lineno}: there is no package here."
            if any(a.name == "*" for a in node.names):
                return ("a star import puts names in scope that cannot be traced "
                        "to a module. Import the names you need explicitly.")
            tops = [(node.module or "").split(".")[0]]
        elif isinstance(node, ast.Import):
            tops = [a.name.split(".")[0] for a in node.names]
        else:
            continue
        for top in tops:
            if top not in config.ALLOWED_IMPORTS:
                return (f"'{top}' cannot be imported here. Allowed: {allowed}. "
                        "Rewrite the tool using only those.")
    return ""


def _check_attributes(tree: ast.Module, modules: set[str], values: set[str]) -> str:
    """Every attribute chain must hang off something the guard actually saw."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        root: ast.AST = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name):
            continue                    # a method on data: "".join, m.groups()
        if root.id in modules or root.id in values or root.id in ALLOWED_BUILTINS:
            continue
        return (f"'{root.id}' on line {node.lineno} is not a module you imported "
                f"and not a value in scope, so '{root.id}.{node.attr}' cannot be "
                "resolved. Compute the answer from the arguments you were given.")
    return ""


def _check_calls(tree: ast.Module, modules: set[str], values: set[str]) -> str:
    """Called names must be allowed builtins, or something defined here."""
    reflective = ("getattr", "setattr", "delattr", "vars", "globals", "locals", "dir")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name == "type" and len(node.args) == 3:
            return ("three-argument type() builds a class, and classes are not "
                    "allowed here. Use type(x) to check a type, nothing more.")
        if name in values or name in modules or name in ALLOWED_BUILTINS:
            continue
        if name in reflective:
            return (f"{name}() is not available. Build the value you need directly - "
                    "an attribute chosen at run time cannot be checked before it "
                    f"runs. Offending call: {ast.dump(node, annotate_fields=False)[:100]}")
        return (f"{name}() is not available here. A tool computes its answer from "
                "its keyword arguments; it does not read, execute or reflect on "
                "anything.")
    return ""


def _check_sentinel(tree: ast.Module) -> str:
    """No string may contain the marker that separates results from output."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and protocol.SENTINEL in node.value):
            return (f"the string on line {node.lineno} contains "
                    f"{protocol.SENTINEL!r}, which the runner uses to tell a "
                    "result from ordinary output. Remove it.")
    return ""


def guard(source: str) -> Verdict:
    """Run every static check over one generated module and return a verdict."""
    if not source.strip():
        return _block("size", "the module is empty.")
    if len(source) > config.MAX_SOURCE_CHARS:
        return _block("size", f"the module is {len(source):,} characters and the "
                              f"limit is {config.MAX_SOURCE_CHARS:,}.")
    if source.startswith("\ufeff"):
        return _block("syntax", "remove the byte-order mark at the start of the file.")
    for line in source.splitlines()[:2]:
        if line.lstrip().startswith("#") and (found := _COOKIE.search(line)):
            return _block("syntax", f"remove the encoding declaration "
                                    f"{found.group(1)!r}; the file is read as UTF-8.")
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        return _block("syntax", f"line {err.lineno}: {err.msg}")

    modules, values = _bindings(tree)
    # Node types are judged first: a construct that is not allowed at all
    # deserves to be told so plainly, and the reason text is the only thing the
    # model gets to act on.
    for layer, check in (
        ("names", lambda: _check_names(tree)),
        ("shape", lambda: _check_shape(tree)),
        ("statements", lambda: _check_top_level(tree)),
        ("imports", lambda: _check_imports(tree)),
        ("attributes", lambda: _check_attributes(tree, modules, values)),
        ("calls", lambda: _check_calls(tree, modules, values)),
        ("sentinel", lambda: _check_sentinel(tree)),
    ):
        if problem := check():
            return _block(layer, problem)

    # Not a rewrite - a proof. If unparsing and re-parsing gives a different
    # tree, something in the text is not represented in the tree we walked.
    try:
        if ast.dump(tree) != ast.dump(ast.parse(ast.unparse(tree))):
            return _block("syntax", "this source does not survive a parse round "
                                    "trip. Rewrite it more plainly.")
    except (SyntaxError, ValueError, RecursionError) as err:
        return _block("syntax", f"this source could not be re-parsed: {err}")
    return Verdict(ok=True, source=source, layer="passed")


if __name__ == "__main__":
    honest = (
        "import string\n"
        "def _fold(product):\n"
        "    q, r = divmod(product, 36)\n"
        "    return q + r\n"
        "def run(**kwargs):\n"
        "    'Return the GSTIN check character.'\n"
        "    alphabet = string.digits + string.ascii_uppercase\n"
        "    total = sum(_fold(alphabet.index(c) * (1 if i % 2 == 0 else 2))\n"
        "                for i, c in enumerate(kwargs['prefix']))\n"
        "    return alphabet[(36 - total % 36) % 36]\n")
    verdict = guard(honest)
    assert verdict.ok, (verdict.layer, verdict.reason)
    assert verdict.source == honest, "the source must reach disk unrewritten"

    cases = [
        ("imports", "import os\ndef run(**kwargs):\n    return os.getcwd()\n"),
        ("imports", "import os.path\ndef run(**kwargs):\n    return 1\n"),
        ("imports", "from math import *\ndef run(**kwargs):\n    return 1\n"),
        ("names", "def run(**kwargs):\n    return __import__('os')\n"),
        ("names", "def run(**kwargs):\n    return ().__class__.__mro__\n"),
        ("names", "def run(**kwargs):\n    return 1\n"
                  "if __name__ == '__main__':\n    print(run())\n"),
        ("names", "class X:\n    pass\ndef run(**kwargs):\n    return 1\n"),
        ("names", "import os\ndef run(**kwargs):\n    try:\n        os.system('x')\n"
                  "    except:\n        pass\n    return 1\n"),
        ("calls", "def run(**kwargs):\n    return getattr(str, 'up' + 'per')\n"),
        ("calls", "def run(**kwargs):\n    return eval('1+1')\n"),
        ("calls", "def run(**kwargs):\n    return open('a.txt').read()\n"),
        ("attributes", "def run(**kwargs):\n    return sys.modules\n"),
        ("statements", "print('side effect')\ndef run(**kwargs):\n    return 1\n"),
        ("statements", "T = [i for i in range(3)]\ndef run(**kwargs):\n    return T\n"),
        ("shape", "def run(n, **kwargs):\n    return n\n"),
        ("sentinel", "def run(**kwargs):\n    print('@@TOOLSMITH@@{}')\n    return 1\n"),
        ("syntax", "# coding: unicode_escape\ndef run(**kwargs):\n    return 1\n"),
    ]
    for expected, body in cases:
        got = guard(body)
        assert not got.ok, f"should have been blocked: {body!r}"
        assert got.layer == expected, f"{body!r}\n  blocked at {got.layer}: {got.reason}"

    # Ordinary defensive code must survive, or the guard taxes every honest tool.
    fine = ("import decimal\ndef run(**kwargs):\n"
            "    with decimal.localcontext() as ctx:\n        ctx.prec = 28\n"
            "        try:\n            value = decimal.Decimal(str(kwargs['n']))\n"
            "        except ValueError:\n            return None\n"
            "    print(f'debug {value}')\n    return float(value)\n")
    assert guard(fine).ok, (guard(fine).layer, guard(fine).reason)
    print(f"{len(cases)} hostile bodies blocked, each at the layer it should be")
    print("with/decimal, try/except ValueError, f-strings and print all allowed")
    print("OK - guard")
