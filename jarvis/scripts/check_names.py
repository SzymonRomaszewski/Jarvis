#!/usr/bin/env python3
"""Tiny static check (a mini-pyflakes): report names that are used but never defined, and unused imports.

Usage: python scripts/check_names.py jarvis.py
"""
import ast
import builtins
import sys


def targets(node):
    """Names bound by an assignment-like target."""
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            yield from targets(e)
    elif isinstance(node, ast.Starred):
        yield from targets(node.value)


class Scope:
    def __init__(self, parent=None):
        self.parent, self.names = parent, set()

    def defined(self, name):
        s = self
        while s:
            if name in s.names:
                return True
            s = s.parent
        return False


def bind_all(body_nodes, scope):
    """Collect every name a block binds (without descending into nested functions/classes' bodies)."""
    for top in body_nodes:
        stack = [top]
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope.names.add(n.name)
                continue
            if isinstance(n, ast.Lambda):
                continue
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    scope.names.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    scope.names.update(targets(t))
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
                scope.names.update(targets(n.target))
            elif isinstance(n, (ast.For, ast.AsyncFor)):
                scope.names.update(targets(n.target))
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for item in n.items:
                    if item.optional_vars is not None:
                        scope.names.update(targets(item.optional_vars))
            elif isinstance(n, ast.ExceptHandler) and n.name:
                scope.names.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                scope.names.update(n.names)
            elif isinstance(n, ast.NamedExpr):
                scope.names.update(targets(n.target))
            elif isinstance(n, ast.comprehension):
                scope.names.update(targets(n.target))
            stack.extend(ast.iter_child_nodes(n))


problems = []


def check(node, scope, path):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            inner = Scope(scope)
            a = child.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
                inner.names.add(arg.arg)
            body = child.body if isinstance(child.body, list) else [child.body]
            bind_all(body, inner)
            for d in getattr(child, "decorator_list", []) + a.defaults + [x for x in a.kw_defaults if x]:
                check_expr(d, scope)
            for stmt in body:
                check_expr(stmt, inner)
        elif isinstance(child, ast.ClassDef):
            inner = Scope(scope)
            bind_all(child.body, inner)
            for stmt in child.body:
                check_expr(stmt, inner)
        else:
            check_expr(child, scope)


def check_expr(node, scope):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
        wrapper = ast.Module(body=[node], type_ignores=[])
        check(wrapper, scope, None)
        return
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        if not scope.defined(node.id) and not hasattr(builtins, node.id):
            problems.append((node.lineno, f"undefined name '{node.id}'"))
    for child in ast.iter_child_nodes(node):
        check_expr(child, scope)


def main(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    module = Scope()
    module.names.update({"__file__", "__name__", "__doc__"})
    bind_all(tree.body, module)
    for stmt in tree.body:
        check_expr(stmt, module)
    # unused imports (module level only)
    imported = {}
    for n in tree.body:
        if isinstance(n, ast.Import):
            for a in n.names:
                imported[(a.asname or a.name).split(".")[0]] = n.lineno
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                imported[a.asname or a.name] = n.lineno
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    for name, line in imported.items():
        if name not in used:
            problems.append((line, f"unused import '{name}'"))
    for line, msg in sorted(set(problems)):
        print(f"{path}:{line}: {msg}")
    print(f"{len(set(problems))} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "jarvis.py"))
