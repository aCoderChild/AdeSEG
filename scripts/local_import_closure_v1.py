"""Discover static project-local Python imports for source provenance.

Dynamic imports and external/native libraries need separate explicit bindings.
The caller supplies allowed package roots and its own initial entrypoints.
"""
import ast
from pathlib import Path


def local_import_closure(root, entrypoints, packages=('scripts', 'modeling', 'datasets', 'evaluation', 'training')):
    root = Path(root).resolve()
    allowed = set(packages)
    pending = [Path(name) for name in entrypoints if str(name).endswith('.py')]
    visited = set()

    def resolve(module):
        parts = module.split('.') if module else []
        if not parts or parts[0] not in allowed:
            return []
        found = []
        for end in range(1, len(parts) + 1):
            init = Path(*parts[:end]) / '__init__.py'
            if (root / init).is_file():
                found.append(init)
        file = Path(*parts).with_suffix('.py')
        if (root / file).is_file():
            found.append(file)
        return found

    while pending:
        relative = pending.pop()
        if relative in visited:
            continue
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Local source path escapes project')
        visited.add(relative)
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(relative))
        parts = list(relative.with_suffix('').parts)
        package = parts[:-1] if parts[-1] != '__init__' else parts[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pending.extend(resolve(alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        continue
                    base = package[:len(package) - node.level + 1]
                    if node.module:
                        base += node.module.split('.')
                    module = '.'.join(base)
                else:
                    module = node.module or ''
                pending.extend(resolve(module))
                for alias in node.names:
                    if alias.name != '*':
                        pending.extend(resolve(module + '.' + alias.name))
    return frozenset(str(path) for path in visited)
