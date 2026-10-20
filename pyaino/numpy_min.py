"""pyaino.numpy_min

Make a pyaino program runnable with Python + NumPy only.

A model trained with pyaino, together with the program that runs it, is cut
out into a self-contained directory (``numpy_package_min``) that contains only
the pyaino definitions the program really uses.  The installed pyaino is not
needed to run the result.

Public API (three levels)
-------------------------
make_numpy_min(target, ...)                       -- normal entry point
    Discover runtime resources from the target source, run a runtime trace
    when needed, then build the package.  An individual program only calls this.

trace_usage(target, ...)                          -- step 1
    Run the target once and record the pyaino classes/functions actually used.

build_numpy_min(target, ...)                      -- step 2
    Extract only the required pyaino definitions, strip training-only methods,
    and copy the target plus the listed runtime resources.

Helpers: discover_resources, check_source_set, load_trace, trace_is_complete,
current_pyaino_dir, installed_pyaino_dir.

Usage from an individual program::

    from pathlib import Path
    from pyaino.numpy_min import make_numpy_min

    BASE_DIR = Path(__file__).resolve().parent
    if __name__ == '__main__':
        make_numpy_min(BASE_DIR / 'my_generate.py',
                       work_dir=r'D:\\Python\\pyaino_standalone')

Layout of this file
-------------------
 1. Policy constants
 2. Locating pyaino sources
 3. Indexing a pyaino module (AST)
 4. Dependency collection
 5. Inference-only transformation
 6. Emitting reduced modules
 7. Runtime resources
 8. Runtime trace
 9. Package build              (build_numpy_min)
10. One-shot driver            (make_numpy_min)
11. Command-line interface

This module contains no application-specific settings.
"""
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import ast
import copy
import importlib.util
import json
import os
import runpy
import shutil
import sys
import tempfile

__all__ = [
    # high level
    'make_numpy_min',
    # building blocks
    'trace_usage',
    'build_numpy_min',
    # helpers
    'discover_resources',
    'check_source_set',
    'load_trace',
    'trace_is_complete',
    'current_pyaino_dir',
    'installed_pyaino_dir',
]


# ===========================================================================
# 1. Policy constants
# ===========================================================================

# Training-side methods removed from an inference-only package by default.
TRAINING_METHODS = {
    'backward', '__backward__', 'update', 'step',
    'cleargrad', 'cleargrads', 'clear_grad', 'clear_grads',
    'zerograd', 'zero_grad', 'zero_grads',
    'set_grad', 'set_grads', 'get_grad', 'get_grads', 'fix_grads',
    'generate_dot_graph', '_dot_var', 'backtrace',
    'backup', 'recover', 'set_gradient', 'flush_gradient', 'accommodate',
}

# Config is small and performs backend initialization at import time.
# Keeping it intact is safer than slicing it.
FULL_COPY_MODULES = {'Config'}

# safe_np contains backend-conditional definitions.  It is replaced by a tiny
# NumPy-only compatibility layer that contains only the referenced names.
SAFE_NP_MODULE = 'safe_np'

# Pseudo symbols used in dependency keys (module, symbol).
WHOLE_MODULE = '*'          # every top-level definition of the module
MODULE_ONLY = '@module'     # only the module itself (an import statement)

# Name of the module that represents the package root "pyaino".
PACKAGE_INIT = '__init__'

# Defaults for make_numpy_min / discover_resources.
DEFAULT_OUTPUT_NAME = 'numpy_package_min'
DEFAULT_LOAD_METHODS = ('load', 'load_parameters')
DEFAULT_RESOURCE_SUFFIXES = ('.pkl',)
DEFAULT_TRACE_MESSAGE = ('対象プログラムが起動したら、少なくとも1回は'
                         '実行・応答させてから終了してください。')

MANIFEST_NAME = 'PACKING_MIN.txt'
TRACE_FORMAT = 1


# ===========================================================================
# 2. Locating pyaino sources
# ===========================================================================

def installed_pyaino_dir():
    """Directory of the importable pyaino package, or None."""
    spec = importlib.util.find_spec('pyaino')
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def current_pyaino_dir():
    """Directory of the pyaino package that contains this module."""
    return Path(__file__).resolve().parent


def module_source(source_dir, module):
    """Source file of pyaino module ``module`` ('a.b' form), or None."""
    if module == PACKAGE_INIT:
        p = source_dir / '__init__.py'
        return p if p.exists() else None
    rel = Path(*module.split('.'))
    p = source_dir / rel.with_suffix('.py')
    if p.exists():
        return p
    p = source_dir / rel / '__init__.py'
    return p if p.exists() else None


def module_destination(package_dir, module, src):
    """Output path that mirrors the location of ``src`` in the source tree."""
    if module == PACKAGE_INIT:
        return package_dir / '__init__.py'
    rel = Path(*module.split('.'))
    if src.name == '__init__.py':
        return package_dir / rel / '__init__.py'
    return package_dir / rel.with_suffix('.py')


def _pyaino_submodule(dotted_name):
    """'pyaino' -> '__init__', 'pyaino.x.y' -> 'x.y', anything else -> None."""
    if dotted_name == 'pyaino':
        return PACKAGE_INIT
    if dotted_name.startswith('pyaino.'):
        return dotted_name[len('pyaino.'):]
    return None


# ===========================================================================
# 3. Indexing a pyaino module (AST)
# ===========================================================================

DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
ASSIGN_NODES = (ast.Assign, ast.AnnAssign, ast.AugAssign)
IMPORT_NODES = (ast.Import, ast.ImportFrom)


def assigned_names(node):
    """Names bound at top level by an assignment statement."""
    names = []

    def add_target(t):
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)

    if isinstance(node, ast.Assign):
        for t in node.targets:
            add_target(t)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        add_target(node.target)
    return names


def is_main_guard(node):
    """True for ``if __name__ == '__main__':``."""
    if not isinstance(node, ast.If):
        return False
    t = node.test
    return (isinstance(t, ast.Compare)
            and isinstance(t.left, ast.Name) and t.left.id == '__name__'
            and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
            and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == '__main__')


def is_docstring(node):
    return (isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


class ModuleInfo:
    """Parsed module with its top-level definitions and pyaino import bindings."""

    def __init__(self, module, path, source_dir):
        self.module = module
        self.path = path
        self.source_dir = source_dir
        self.tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))

        self.defs = {}               # name -> def/class node
        self.assignments = {}        # name -> assignment node
        self.module_aliases = {}     # local name -> pyaino module
        self.symbol_aliases = {}     # local name -> (pyaino module, symbol)
        self.star_modules = []       # pyaino modules imported with "*"
        self.pyaino_import_nodes = []
        self.non_pyaino_import_nodes = []
        self._index()

    @property
    def local_names(self):
        return set(self.defs) | set(self.assignments)

    def _index(self):
        for node in self.tree.body:
            if isinstance(node, DEF_NODES):
                self.defs[node.name] = node
            elif isinstance(node, ASSIGN_NODES):
                for name in assigned_names(node):
                    self.assignments[name] = node
            elif isinstance(node, IMPORT_NODES):
                if self._bind_pyaino_import(node):
                    self.pyaino_import_nodes.append(node)
                else:
                    self.non_pyaino_import_nodes.append(node)

    def _bind_pyaino_import(self, node):
        """Record local bindings created by a pyaino import.  False if not pyaino."""
        if isinstance(node, ast.Import):
            hit = False
            for a in node.names:
                module = _pyaino_submodule(a.name)
                if module is None:
                    continue
                local = a.asname or a.name.split('.')[-1]
                self.module_aliases[local] = module
                hit = True
            return hit

        module = _pyaino_submodule(node.module or '')
        if module is None:
            return False
        for a in node.names:
            local = a.asname or a.name
            if a.name == '*':
                self.star_modules.append(module)
            elif module == PACKAGE_INIT and module_source(self.source_dir, a.name) is not None:
                # "from pyaino import Neuron": Neuron is normally a submodule.
                self.module_aliases[local] = a.name
            else:
                self.symbol_aliases[local] = (module, a.name)
        return True

    def star_provider(self, name):
        """pyaino module that supplies ``name`` through a star import, or None."""
        for module in self.star_modules:
            if module_source(self.source_dir, module) is None:
                continue
            try:
                if name in get_info(module, self.source_dir).local_names:
                    return module
            except Exception:
                continue
        return None


_INFO_CACHE = {}


def get_info(module, source_dir):
    """Cached ModuleInfo for a pyaino module."""
    key = (str(Path(source_dir).resolve()), module)
    if key not in _INFO_CACHE:
        src = module_source(Path(source_dir), module)
        if src is None:
            raise FileNotFoundError(module)
        _INFO_CACHE[key] = ModuleInfo(module, src, Path(source_dir))
    return _INFO_CACHE[key]


class DependencyVisitor(ast.NodeVisitor):
    """Collect names a code fragment needs.

    local    : top-level names of the same module
    external : (pyaino module, symbol) pairs; symbol may be WHOLE_MODULE or MODULE_ONLY
    """

    def __init__(self, info):
        self.info = info
        self.local = set()
        self.external = set()

    def visit_Attribute(self, node):
        root, attrs = self._chain(node)
        if root in self.info.module_aliases and attrs:
            # e.g. "Neuron.LinearLayer" -> ('Neuron', 'LinearLayer')
            self.external.add((self.info.module_aliases[root], attrs[0]))
            return
        self.generic_visit(node)

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            return
        name = node.id
        info = self.info
        if name in info.module_aliases:
            self.external.add((info.module_aliases[name], WHOLE_MODULE))
        elif name in info.symbol_aliases:
            self.external.add(info.symbol_aliases[name])
        elif name in info.local_names:
            self.local.add(name)
        else:
            provider = info.star_provider(name)
            if provider is not None:
                self.external.add((provider, name))

    def visit_Import(self, node):
        for a in node.names:
            module = _pyaino_submodule(a.name)
            if module is not None:
                self.external.add((module, MODULE_ONLY))

    def visit_ImportFrom(self, node):
        module = _pyaino_submodule(node.module or '')
        if module is None:
            return
        for a in node.names:
            if module == PACKAGE_INIT:
                if a.name == '*':
                    self.external.add((PACKAGE_INIT, MODULE_ONLY))
                elif module_source(self.info.source_dir, a.name) is not None:
                    self.external.add((a.name, MODULE_ONLY))
                else:
                    self.external.add((PACKAGE_INIT, a.name))
            else:
                self.external.add((module, WHOLE_MODULE if a.name == '*' else a.name))

    @staticmethod
    def _chain(node):
        """``a.b.c`` -> ('a', ['b', 'c'])."""
        attrs = []
        cur = node
        while isinstance(cur, ast.Attribute):
            attrs.append(cur.attr)
            cur = cur.value
        attrs.reverse()
        return (cur.id if isinstance(cur, ast.Name) else None), attrs


def scan_node(node, info):
    v = DependencyVisitor(info)
    v.visit(node)
    return v


# ===========================================================================
# 4. Dependency collection
# ===========================================================================

@dataclass
class Selection:
    """Result of dependency collection."""
    defs: dict = field(default_factory=lambda: defaultdict(set))         # module -> def names
    assignments: dict = field(default_factory=lambda: defaultdict(set))  # module -> assigned names
    required: dict = field(default_factory=lambda: defaultdict(set))     # module -> requested symbols
    required_by: dict = field(default_factory=lambda: defaultdict(set))  # (module, symbol) -> parents
    missing: dict = field(default_factory=lambda: defaultdict(set))      # (module, symbol) -> parents
    modules: set = field(default_factory=set)                            # modules to be emitted


def collect_symbols(target, source_dir, keep_methods, traced_defs=None):
    """Walk dependencies from the target program and select pyaino definitions."""
    traced_defs = traced_defs or {}
    sel = Selection()
    pending = deque()

    def require(module, symbol, parent):
        # A wildcard may later be narrowed by the runtime trace, so it never
        # suppresses an explicit request for the same module.  Example:
        #   Activators:*  --trace-->  {Mish, Softmax}
        #   Mish -> ActivatorBase   (must still enter the queue)
        sel.required_by[(module, symbol)].add(parent)
        if symbol in sel.required[module]:
            return
        sel.required[module].add(symbol)
        pending.append((module, symbol, parent))

    def require_deps_of(node, info, parent):
        v = scan_node(node, info)
        for local in v.local:
            require(info.module, local, parent)
        for depmod, depsym in v.external:
            require(depmod, depsym, parent)

    target_info = ModuleInfo('<target>', target, source_dir)
    for module, symbol in scan_node(target_info.tree, target_info).external:
        require(module, symbol, '<target>')

    while pending:
        module, symbol, parent = pending.popleft()
        if module_source(source_dir, module) is None:
            sel.missing[(module, symbol)].add(parent)
            continue
        info = get_info(module, source_dir)
        sel.modules.add(module)

        if module == SAFE_NP_MODULE:
            # Emitted as a NumPy-only shim (see write_safe_np); its original
            # definitions are backend-conditional and are not indexed.
            continue
        if module in FULL_COPY_MODULES:
            # Copied intact; only follow its pyaino dependencies.
            for depmod, depsym in scan_node(info.tree, info).external:
                if depmod != module:
                    require(depmod, depsym, module + ':*')
            continue
        if symbol == MODULE_ONLY:
            continue

        for name in _expand_symbol(symbol, info, traced_defs.get(module, ())):
            origin = f'{module}.{name}'
            if name in info.defs:
                if name not in sel.defs[module]:
                    sel.defs[module].add(name)
                    node = transformed_def(module, info.defs[name], keep_methods)
                    require_deps_of(node, info, origin)
            elif name in info.assignments:
                if name not in sel.assignments[module]:
                    sel.assignments[module].add(name)
                    require_deps_of(info.assignments[name], info, origin)
            elif name in info.symbol_aliases:
                depmod, depsym = info.symbol_aliases[name]
                require(depmod, depsym, origin)
            elif name in info.module_aliases:
                require(info.module_aliases[name], WHOLE_MODULE, origin)
            else:
                provider = info.star_provider(name)
                if provider is not None:
                    require(provider, name, origin)
                else:
                    sel.missing[(module, name)].add(parent)
    return sel


def _expand_symbol(symbol, info, traced):
    """Concrete names for one requested symbol.

    For WHOLE_MODULE the runtime trace resolves dynamic lookups such as
    eval_in_module(name, Activators).  Without a usable trace the whole module
    (except *_bkup) is retained conservatively.
    """
    if symbol != WHOLE_MODULE:
        return [symbol]
    available = info.local_names
    narrowed = set(traced) & available
    return sorted(narrowed or {n for n in available if not n.endswith('_bkup')})


# ===========================================================================
# 5. Inference-only transformation
# ===========================================================================

_FORWARD_FUNCTION_SRC = '''
class Function:
    def __init__(self, log=False, log_file='log_file.txt', preserve_attr=False):
        self.inputs = None
        self.outputs = None
        self.generation = 0
        if not hasattr(self, 'config'):
            self.config = None
        self.preserve_attr = preserve_attr

    def forward(self, *inputs, **kwargs):
        return self.__forward__(*inputs, **kwargs)

    def __forward__(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
'''

_FORWARD_COMPOSIT_SRC = '''
class CompositFunction:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, *inputs, **kwargs):
        return self._forward(*inputs, **kwargs)

    def _forward(self, *inputs, **kwargs):
        raise NotImplementedError()

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)
'''

# common_function.import_parameters_recursive() uses hasattr(obj, 'update')
# to recognise a parameter-bearing layer.  This no-op keeps that marker.
_UPDATE_MARKER_SRC = '''
def update(self, *args, **kwargs):
    pass
'''

# (module, class) -> inference-only replacement source
CLASS_REPLACEMENTS = {
    ('nucleus', 'Function'): _FORWARD_FUNCTION_SRC,
    ('nucleus', 'CompositFunction'): _FORWARD_COMPOSIT_SRC,
}

# Classes whose constructor builds a regularizer that inference does not need.
REGULARIZER_CLASSES = {'AttentionUnit', 'AttentionUnit_bkup'}


def _parse_single(src):
    return ast.parse(src).body[0]


def _contains_name(node, names):
    return any(isinstance(x, ast.Name) and x.id in names for x in ast.walk(node))


def _self_attr_targets(node):
    """Attributes assigned as ``self.xxx = ...``."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [t.attr for t in targets
            if isinstance(t, ast.Attribute)
            and isinstance(t.value, ast.Name) and t.value.id == 'self']


class InferenceInitTransformer(ast.NodeTransformer):
    """Remove ``self.optimizer_* / self.OF* = ...Optimizers/cf...`` from __init__."""

    def _drop_optimizer_setup(self, node):
        node = self.generic_visit(node)
        attrs = _self_attr_targets(node)
        if (any(a.startswith(('optimizer_', 'OF')) for a in attrs)
                and _contains_name(node, {'Optimizers', 'cf'})):
            return None
        return node

    visit_Assign = _drop_optimizer_setup
    visit_AnnAssign = _drop_optimizer_setup

    def visit_If(self, node):
        node = self.generic_visit(node)
        node.body = [x for x in node.body if x is not None]
        node.orelse = [x for x in node.orelse if x is not None]
        if not node.body and not node.orelse:
            return None
        if not node.body:
            node.body = [ast.Pass()]
        return node


def _drop_regularizer_setup(method):
    """``regularizer = ...`` -> ``self.regularizer = None``; drop the following if-block."""
    body = []
    skip_next_regularizer_if = False
    for stmt in method.body:
        if (isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'regularizer' for t in stmt.targets)):
            body.append(ast.Assign(
                targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                       attr='regularizer', ctx=ast.Store())],
                value=ast.Constant(value=None)))
            skip_next_regularizer_if = True
            continue
        if (skip_next_regularizer_if and isinstance(stmt, ast.If)
                and _contains_name(stmt, {'regularizer'})):
            skip_next_regularizer_if = False
            continue
        skip_next_regularizer_if = False
        body.append(stmt)
    method.body = body or [ast.Pass()]


def strip_training_initialization(method, class_name):
    """Remove constructor state used only by optimizer/regularizer machinery."""
    if method.name != '__init__':
        return method
    method = InferenceInitTransformer().visit(copy.deepcopy(method))
    if class_name in REGULARIZER_CLASSES:
        _drop_regularizer_setup(method)
    ast.fix_missing_locations(method)
    return method


def _is_training_method(name, keep_methods):
    return (name in TRAINING_METHODS or name.endswith('_bkup')) and name not in keep_methods


def prune_class(module, node, keep_methods):
    """Inference-only version of a class definition."""
    replacement = CLASS_REPLACEMENTS.get((module, node.name))
    if replacement is not None:
        return _parse_single(replacement)

    method_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    original_methods = {item.name for item in node.body if isinstance(item, method_types)}

    body = []
    for item in node.body:
        if isinstance(item, method_types):
            if _is_training_method(item.name, keep_methods):
                continue
            if isinstance(item, ast.FunctionDef):
                item = strip_training_initialization(item, node.name)
        body.append(item)

    if 'update' in original_methods and 'update' not in keep_methods:
        body.append(_parse_single(_UPDATE_MARKER_SRC))

    new = ast.ClassDef(name=node.name, bases=node.bases, keywords=node.keywords,
                       body=body or [ast.Pass()], decorator_list=node.decorator_list)
    if hasattr(node, 'type_params'):
        new.type_params = node.type_params
    return ast.copy_location(new, node)


def transformed_def(module, node, keep_methods):
    if isinstance(node, ast.ClassDef):
        return prune_class(module, node, keep_methods)
    return node


# ===========================================================================
# 6. Emitting reduced modules
# ===========================================================================

def names_loaded(nodes):
    return {x.id for node in nodes for x in ast.walk(node)
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}


def import_bound_names(node):
    """Local names bound by an import statement ('*' for a star import)."""
    names = []
    for a in node.names:
        if a.name == '*':
            names.append('*')
        elif isinstance(node, ast.Import):
            # "import os.path" binds "os"; "import x.y as z" binds "z".
            names.append(a.asname or a.name.split('.')[0])
        else:
            names.append(a.asname or a.name)
    return names


def _with_aliases(node, kept):
    if isinstance(node, ast.Import):
        return ast.Import(names=kept)
    return ast.ImportFrom(module=node.module, names=kept, level=node.level)


def rewrite_plain_import(node, used_names):
    """Drop non-pyaino imports no longer referenced by the reduced module.

    Stripping training/plotting definitions can leave their imports behind,
    e.g. matplotlib in BigramLanguageModel once graph_plus() is removed.
    """
    bound = import_bound_names(node)
    if '*' in bound:
        return node                  # may provide names still relied upon
    kept = [a for a, local in zip(node.names, bound) if local in used_names]
    if not kept:
        return None
    if len(kept) == len(node.names):
        return node
    return _with_aliases(node, kept)


def rewrite_pyaino_import(node, used_names):
    """Keep only the pyaino import bindings the reduced module references."""
    if isinstance(node, ast.Import):
        kept = []
        for a in node.names:
            if _pyaino_submodule(a.name) is None:
                kept.append(a)
            else:
                local = a.asname or a.name.split('.')[-1]
                if local in used_names:
                    kept.append(a)
        return ast.Import(names=kept) if kept else None

    if _pyaino_submodule(node.module or '') is None:
        return node
    if any(a.name == '*' for a in node.names):
        return node
    kept = [a for a in node.names if (a.asname or a.name) in used_names]
    return _with_aliases(node, kept) if kept else None


def build_reduced_tree(module, info, sel, keep_methods):
    """Docstring + needed imports + selected definitions, in original order."""
    kept_defs = sel.defs.get(module, set())
    kept_assign = sel.assignments.get(module, set())

    selected = []
    for node in info.tree.body:
        if isinstance(node, DEF_NODES):
            if node.name in kept_defs:
                selected.append(transformed_def(module, node, keep_methods))
        elif isinstance(node, ASSIGN_NODES):
            if any(n in kept_assign for n in assigned_names(node)):
                selected.append(node)
    used = names_loaded(selected)

    body = []
    if info.tree.body and is_docstring(info.tree.body[0]):
        body.append(info.tree.body[0])
    body += [r for r in (rewrite_plain_import(n, used) for n in info.non_pyaino_import_nodes)
             if r is not None]
    body += [r for r in (rewrite_pyaino_import(n, used) for n in info.pyaino_import_nodes)
             if r is not None]
    body += selected

    tree = ast.Module(body=body or [ast.Pass()], type_ignores=[])
    ast.fix_missing_locations(tree)
    return tree


def write_safe_np(dst, requested):
    """Emit only the safe_np names required by this NumPy-only pack."""
    names = sorted(n for n in requested if n not in {WHOLE_MODULE, MODULE_ONLY})
    lines = [
        '# generated NumPy-only safe_np subset',
        'import numpy as np',
        'import math as _math',
        '',
    ]
    for name in names:
        if name == 'add_at':
            lines.append('add_at = np.add.at')
        elif name == 'erf':
            lines += ['def erf(x):',
                      '    return np.vectorize(_math.erf, otypes=[float])(x)']
        else:
            lines.append(f'{name} = np.{name}')
    if not names:
        lines.append('pass')
    dst.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_module(module, info, dst, sel, keep_methods):
    """Write one module of the pack.  Returns (source_size, output_size)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if module == SAFE_NP_MODULE:
        write_safe_np(dst, sel.required.get(module, set()))
    elif module in FULL_COPY_MODULES:
        shutil.copy2(info.path, dst)
    else:
        tree = build_reduced_tree(module, info, sel, keep_methods)
        dst.write_text(ast.unparse(tree) + '\n', encoding='utf-8')
    return info.path.stat().st_size, dst.stat().st_size


# ===========================================================================
# 7. Runtime resources
# ===========================================================================
# Two ways to specify the data files (tokenizer, checkpoint, ...):
#   * discover_resources(): read the target's load calls (target is NOT run)
#   * resources=...       : given explicitly to build_numpy_min

_UNKNOWN = object()


def _static_value(node, env):
    """Evaluate a simple, side-effect-free expression; _UNKNOWN otherwise.

    Supported: constants, names in ``env``, f-strings, ``+``, Path ``/``,
    ``.parent``, Path()/str()/os.fspath(), ``.resolve()``, os.path.join().
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, _UNKNOWN)

    if isinstance(node, ast.JoinedStr):
        parts = [_static_value(x, env) for x in node.values]
        return _UNKNOWN if _UNKNOWN in parts else ''.join(map(str, parts))
    if isinstance(node, ast.FormattedValue):
        v = _static_value(node.value, env)
        if v is _UNKNOWN:
            return _UNKNOWN
        if node.format_spec is None:
            return str(v)
        spec = _static_value(node.format_spec, env)
        return _UNKNOWN if spec is _UNKNOWN else format(v, spec)

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        a = _static_value(node.left, env)
        b = _static_value(node.right, env)
        if a is _UNKNOWN or b is _UNKNOWN:
            return _UNKNOWN
        try:
            if isinstance(node.op, ast.Div) and isinstance(a, Path):
                return a / b
            if isinstance(node.op, ast.Add) and isinstance(a, (str, int, float)):
                return a + b
        except (TypeError, ValueError):
            pass
        return _UNKNOWN

    if isinstance(node, ast.Attribute) and node.attr == 'parent':
        p = _static_value(node.value, env)
        return p.parent if isinstance(p, Path) else _UNKNOWN

    if isinstance(node, ast.Call):
        return _static_call_value(node, env)
    return _UNKNOWN


def _static_call_value(node, env):
    func = node.func
    # Path(x), str(x), os.fspath(x)
    if len(node.args) == 1 and not node.keywords:
        v = _static_value(node.args[0], env)
        if v is _UNKNOWN:
            return _UNKNOWN
        if isinstance(func, ast.Name) and func.id == 'Path':
            return Path(v)
        if isinstance(func, ast.Name) and func.id == 'str':
            return str(v)
        if isinstance(func, ast.Attribute) and func.attr == 'fspath':
            return os.fspath(v)
    # <path>.resolve()
    if (isinstance(func, ast.Attribute) and func.attr == 'resolve'
            and not node.args and not node.keywords):
        p = _static_value(func.value, env)
        return p.resolve() if isinstance(p, Path) else _UNKNOWN
    # os.path.join(...)
    if (isinstance(func, ast.Attribute) and func.attr == 'join'
            and isinstance(func.value, ast.Attribute) and func.value.attr == 'path'
            and isinstance(func.value.value, ast.Name) and func.value.value.id == 'os'):
        values = [_static_value(a, env) for a in node.args]
        if not values or _UNKNOWN in values:
            return _UNKNOWN
        return os.path.join(*(os.fspath(v) for v in values))
    return _UNKNOWN


def _module_scope_env(tree, target):
    """Values of simple module-level assignments of the target (e.g. BASE_DIR)."""
    env = {'__file__': str(target)}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            value = _static_value(stmt.value, env)
            for dst in stmt.targets:
                if isinstance(dst, ast.Name):
                    env[dst.id] = value
        elif (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
              and stmt.value is not None):
            env[stmt.target.id] = _static_value(stmt.value, env)
    return env


def _is_certain_resource_load(call):
    """An unresolvable argument is an error only for clearly resource-loading calls.

    ``load_parameters(...)`` always is; plain ``.load(...)`` only on an object
    whose name contains 'token' (``json.load(f)`` etc. must be ignored).
    """
    if call.func.attr != 'load':
        return True
    owner = call.func.value
    return isinstance(owner, ast.Name) and 'token' in owner.id.lower()


def relative_to_target(p, target):
    """Path relative to target.parent; the resource must live under it."""
    p = Path(p)
    if not p.is_absolute():
        p = target.parent / p
    p = p.resolve()
    try:
        return p.relative_to(target.parent)
    except ValueError as e:
        raise ValueError(f'Resource must be under the target program directory: {p}') from e


def discover_resources(target,
                       load_methods=DEFAULT_LOAD_METHODS,
                       resource_suffixes=DEFAULT_RESOURCE_SUFFIXES):
    """Return resource paths relative to target.parent, without running target.

    load_methods      : method names whose first argument is a resource path
    resource_suffixes : file suffixes treated as resources
    """
    target = Path(target).resolve()
    if not target.is_file():
        raise FileNotFoundError(f'Target program not found: {target}')
    tree = ast.parse(target.read_text(encoding='utf-8-sig'), filename=str(target))
    env = _module_scope_env(tree, target)
    suffixes = tuple(s.lower() for s in resource_suffixes)

    files = []
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        if call.func.attr not in load_methods or not call.args:
            continue
        value = _static_value(call.args[0], env)
        if not isinstance(value, (Path, str)):
            if _is_certain_resource_load(call):
                raise ValueError(
                    f'Cannot determine resource at {target.name}:{call.lineno}. '
                    'Use a literal path or one formed from BASE_DIR, '
                    'simple variables, string concatenation or f-strings, '
                    'or give it via extra_resources.')
            continue
        p = Path(value)
        if p.suffix.lower() not in suffixes:
            continue
        rel = relative_to_target(p, target)
        if rel not in files:
            files.append(rel)
    return files


def check_source_set(target, resources):
    """Check that every resource file exists before the output is overwritten."""
    target = Path(target).resolve()
    missing = [target.parent / rel for rel in resources
               if not (target.parent / rel).is_file()]
    if missing:
        lines = ['Required source files referenced by target are missing:']
        lines.extend(f'  {p}' for p in missing)
        raise FileNotFoundError('\n'.join(lines))


def _resolve_resources(resources, target_dir):
    """Normalise resource specs to ``[(absolute_source, relative_destination)]``.

    Each item is either ``path`` (destination keeps the path relative to the
    target directory) or ``(source, destination_relative_path)``.
    """
    target_dir = Path(target_dir).resolve()
    resolved = []
    for item in resources or ():
        if isinstance(item, (tuple, list)):
            if len(item) != 2:
                raise ValueError('resource tuple/list must be (source, destination_relative_path)')
            raw_src, raw_dst = Path(item[0]), Path(item[1])
            src = raw_src if raw_src.is_absolute() else target_dir / raw_src
            dst_rel = raw_dst
        else:
            raw_src = Path(item)
            src = raw_src if raw_src.is_absolute() else target_dir / raw_src
            if raw_src.is_absolute():
                try:
                    dst_rel = src.resolve().relative_to(target_dir)
                except ValueError:
                    dst_rel = Path(src.name)
            else:
                dst_rel = raw_src
        src = src.resolve()
        if not src.exists():
            raise FileNotFoundError(f'runtime resource not found: {src}')
        if dst_rel.is_absolute() or '..' in dst_rel.parts:
            raise ValueError(f'resource destination must stay inside output_dir: {dst_rel}')
        resolved.append((src, dst_rel))
    return resolved


def _path_is_inside(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    return path == directory or directory in path.parents


def _copy_path(src, dst):
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _stage_overlapping_resources(resolved, output_dir, stage_dir):
    """Save resources that live inside ``output_dir`` before it is replaced.

    This allows a previous numpy_package_min to serve as the source of its
    own tokenizer/checkpoint files when it is rebuilt.
    """
    staged = []
    for i, (src, dst_rel) in enumerate(resolved):
        if _path_is_inside(src, output_dir):
            tmp = Path(stage_dir) / f'{i:03d}_{src.name}'
            _copy_path(src, tmp)
            staged.append((tmp, dst_rel))
        else:
            staged.append((src, dst_rel))
    return staged


def _copy_resources(resolved, output_dir):
    output_dir = Path(output_dir).resolve()
    for src, dst_rel in resolved:
        _copy_path(Path(src).resolve(), output_dir / dst_rel)
    return list(resolved)


# ===========================================================================
# 8. Runtime trace
# ===========================================================================

class _UsageRecorder:
    """sys.setprofile() hook recording which pyaino classes/functions run.

    The hook is called for every Python call in the process, so the hot path
    is string-only and filesystem/AST work is cached per file.
    """

    def __init__(self, pyaino_root):
        self.used = defaultdict(set)
        self._root = os.path.normcase(os.path.abspath(str(pyaino_root)))
        self._root_prefix = self._root + os.sep
        self._inside_cache = {}
        self._defs_cache = {}

    def _inside_pyaino(self, filename):
        hit = self._inside_cache.get(filename)
        if hit is None:
            f = os.path.normcase(os.path.abspath(filename))
            hit = f == self._root or f.startswith(self._root_prefix)
            self._inside_cache[filename] = hit
        return hit

    def _top_defs(self, filename):
        """(classes, functions) defined at the top level of a file."""
        hit = self._defs_cache.get(filename)
        if hit is None:
            classes, functions = set(), set()
            try:
                tree = ast.parse(Path(filename).read_text(encoding='utf-8-sig'),
                                 filename=filename)
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        classes.add(node.name)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions.add(node.name)
            except Exception:
                pass
            hit = self._defs_cache[filename] = (classes, functions)
        return hit

    def _add(self, module_name, symbol):
        module = _pyaino_submodule(module_name)
        if module is not None and symbol and symbol not in {'<module>', '<lambda>'}:
            self.used[module].add(symbol)

    def _add_class(self, cls):
        self._add(getattr(cls, '__module__', ''), getattr(cls, '__name__', ''))

    def profile(self, frame, event, arg):
        if event != 'call':
            return
        filename = frame.f_code.co_filename
        if not self._inside_pyaino(filename):
            return
        module_name = frame.f_globals.get('__name__', '')
        if _pyaino_submodule(module_name) is None:
            return

        # Instance method (also inherited): record the concrete runtime class.
        self_obj = frame.f_locals.get('self')
        if self_obj is not None:
            self._add_class(self_obj.__class__)
            return
        # Class method.
        cls_obj = frame.f_locals.get('cls')
        if isinstance(cls_obj, type):
            self._add_class(cls_obj)
            return

        classes, functions = self._top_defs(filename)
        name = frame.f_code.co_name
        qualname = getattr(frame.f_code, 'co_qualname', name)

        # Executing "class Foo: ..." at import time is definition, not use.
        if name in classes and qualname == name:
            return
        # Static method: Class.method.
        head = qualname.split('.', 1)[0]
        if head in classes and '.' in qualname:
            self._add(module_name, head)
            return
        # Nested function: retaining its top-level owner is sufficient.
        if '.<locals>.' in qualname:
            outer = qualname.split('.<locals>.', 1)[0].split('.', 1)[0]
            if outer in functions or outer in classes:
                self._add(module_name, outer)
            return
        # Ordinary module-level function.
        if name in functions:
            self._add(module_name, name)


def _is_pyaino_module_name(name):
    return name == 'pyaino' or name.startswith('pyaino.')


@contextmanager
def _run_like_direct_execution(target, pyaino_root, working_dir):
    """Make execution resemble running the target directly.

    * relative files resolve from working_dir
    * target-local helper modules are importable
    * the selected pyaino tree wins over another installation
    Everything is restored afterwards.
    """
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    saved_pyaino = {n: m for n, m in sys.modules.items() if _is_pyaino_module_name(n)}
    for n in saved_pyaino:
        del sys.modules[n]
    front = [str(pyaino_root.parent), str(target.parent)]
    sys.path[:] = front + [x for x in old_path if x not in front]
    try:
        os.chdir(working_dir)
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for n in [n for n in sys.modules if _is_pyaino_module_name(n)]:
            del sys.modules[n]
        sys.modules.update(saved_pyaino)


def _write_trace(trace_output, target, working_dir, pyaino_root, used, completed):
    data = {
        'trace_format': TRACE_FORMAT,
        'completed': completed,
        'target': str(target),
        'working_dir': str(working_dir),
        'pyaino_root': str(pyaino_root),
        'modules': {m: sorted(v) for m, v in sorted(used.items())},
    }
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    trace_output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
    print(f'trace written: {trace_output}' if completed
          else f'incomplete trace written: {trace_output}')
    for module, symbols in data['modules'].items():
        print(f'  {module}: {", ".join(symbols)}')


def trace_usage(target, trace_output='PYAINO_TRACE.json', source_dir=None, working_dir=None):
    """Run an inference program and record the pyaino symbols actually used.

    source_dir  : pyaino tree to trace (default: the package containing this module)
    working_dir : current directory while the target runs (default: target directory).
                  Useful for legacy programs that open paths such as
                  ``corpus_params/model.pkl`` relative to the process cwd.

    The trace file is written even when the target fails; it is then marked
    ``completed: false`` and the exception is re-raised.
    """
    target = Path(target).resolve()
    trace_output = Path(trace_output).resolve()
    working_dir = target.parent if working_dir is None else Path(working_dir).resolve()
    pyaino_root = current_pyaino_dir() if source_dir is None else Path(source_dir).resolve()
    if not pyaino_root.exists():
        raise RuntimeError(f'pyaino package not found: {pyaino_root}')
    if not working_dir.exists():
        raise FileNotFoundError(f'working directory not found: {working_dir}')

    recorder = _UsageRecorder(pyaino_root)
    exc = None
    try:
        with _run_like_direct_execution(target, pyaino_root, working_dir):
            sys.setprofile(recorder.profile)
            try:
                runpy.run_path(str(target), run_name='__main__')
            finally:
                sys.setprofile(None)
    except BaseException as e:
        exc = e

    _write_trace(trace_output, target, working_dir, pyaino_root, recorder.used,
                 completed=exc is None)
    if exc is not None:
        raise exc


def _read_trace_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_trace(path):
    """``{module: set(symbols)}`` from a complete trace file."""
    data = _read_trace_json(path)
    if 'completed' in data and not data['completed']:
        raise RuntimeError(f'incomplete runtime trace: {Path(path).resolve()}')
    return {m: set(v) for m, v in data.get('modules', {}).items()}


def trace_is_complete(path, target):
    """True if ``path`` is a complete trace of ``target`` newer than the target."""
    path, target = Path(path), Path(target)
    if not path.is_file() or path.stat().st_mtime < target.stat().st_mtime:
        return False
    try:
        data = _read_trace_json(path)
        return (data.get('completed') is True
                and Path(data.get('target', '')).resolve() == target.resolve())
    except (OSError, ValueError, TypeError):
        return False


# ===========================================================================
# 9. Package build
# ===========================================================================

def _py_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*.py'))


def _emit_package(source_dir, package_dir, sel, keep_methods):
    """Write the reduced pyaino package.  Returns [(module, before, after)]."""
    # An empty initializer avoids importing unrelated pyaino modules.
    (package_dir / '__init__.py').write_text(
        '# generated inference-only pyaino subset; no installed pyaino required\n',
        encoding='utf-8')
    sizes = []
    for module in sorted(sel.modules):
        if module == PACKAGE_INIT:
            continue
        src = module_source(source_dir, module)
        if src is None:
            continue
        info = get_info(module, source_dir)
        dst = module_destination(package_dir, module, src)
        sizes.append((module, *write_module(module, info, dst, sel, keep_methods)))
    return sizes


def _manifest_lines(target, source_dir, package_dir, sel, sizes, traced_defs, keep_methods):
    lines = [f'target: {target}', f'source: {source_dir}', '', 'reduced modules:']
    for module, before, after in sizes:
        pct = 100.0 * after / before if before else 0.0
        lines.append(f'  {module}: {before:,} -> {after:,} bytes ({pct:5.1f}%)')
        if module not in FULL_COPY_MODULES:
            defs = ', '.join(sorted(sel.defs.get(module, ()))) or '-'
            lines.append(f'    defs: {defs}')

    before_total = sum(before for _, before, _ in sizes)
    after_total = _py_size(package_dir)
    lines += [
        '',
        f'pyaino source total: {before_total:,} -> {after_total:,} bytes',
        f'reduction: {(1 - after_total / before_total) * 100:.1f}%' if before_total
        else 'reduction: n/a',
    ]

    if sel.missing:
        lines += ['', 'unresolved symbols/modules:']
        for (module, symbol), parents in sorted(sel.missing.items()):
            lines.append(f'  {module}:{symbol} <- {", ".join(sorted(parents))}')
    else:
        lines += ['', 'all selected dependencies resolved.']

    if traced_defs:
        lines += ['', 'runtime trace assisted: yes']
        for module in sorted(traced_defs):
            lines.append(f'  {module}: {", ".join(sorted(traced_defs[module]))}')
    else:
        lines += ['', 'runtime trace assisted: no']

    lines += [
        '',
        'forward-only policy:',
        '  stripped methods: ' + ', '.join(sorted(TRAINING_METHODS - keep_methods)),
    ]
    if keep_methods:
        lines.append('  explicitly kept: ' + ', '.join(sorted(keep_methods)))
    lines += [
        '  update() is retained only as a no-op structural marker for parameter loading.',
        '  safe_np is replaced by a NumPy-only shim containing only referenced names.',
        '  nucleus.Function / CompositFunction replaced by inference-only dispatchers.',
    ]
    return lines


def _resource_lines(copied):
    lines = ['', 'runtime resources:']
    lines += [f'  {src} -> {dst_rel}' for src, dst_rel in copied] or ['  none']
    return lines


def _as_traced_defs(trace):
    if trace is None:
        return None
    if isinstance(trace, (str, Path)):
        return load_trace(trace)
    return {m: set(v) for m, v in trace.items()}


def build_numpy_min(target, output_dir=DEFAULT_OUTPUT_NAME, source_dir=None,
                    trace=None, keep_methods=(), resources=()):
    """Build a minimum NumPy-only pyaino inference package.

    target       : one executable Python program that uses pyaino
    output_dir   : destination directory; an existing one is replaced
    source_dir   : source pyaino package (default: the package containing this module)
    trace        : trace file from trace_usage, a ``{module: symbols}`` mapping,
                   or None for static analysis only
    keep_methods : training-side method names that must exceptionally be kept
    resources    : runtime data files/directories to copy; each item is
                   ``path`` (relative to the target directory) or ``(source, destination)``

    Returns the unresolved symbol/module report ``{(module, symbol): parents}``.
    """
    target = Path(target).resolve()
    output_dir = Path(output_dir).resolve()
    source_dir = current_pyaino_dir() if source_dir is None else Path(source_dir).resolve()
    if not target.exists():
        raise FileNotFoundError(f'target program not found: {target}')
    if not source_dir.exists():
        raise FileNotFoundError(f'pyaino source directory not found: {source_dir}')
    keep_methods = set(keep_methods)
    traced_defs = _as_traced_defs(trace)

    # Resolve resources before output_dir is replaced (they may live inside it).
    resolved = _resolve_resources(resources, target.parent)

    with tempfile.TemporaryDirectory(prefix='pyaino_numpy_min_') as tmpdir:
        resolved = _stage_overlapping_resources(resolved, output_dir, tmpdir)

        sel = collect_symbols(target, source_dir, keep_methods, traced_defs)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        package_dir = output_dir / 'pyaino'
        package_dir.mkdir(parents=True)
        shutil.copy2(target, output_dir / target.name)
        sizes = _emit_package(source_dir, package_dir, sel, keep_methods)

        copied = _copy_resources(resolved, output_dir)

    lines = _manifest_lines(target, source_dir, package_dir, sel, sizes,
                            traced_defs, keep_methods)
    lines += _resource_lines(copied)
    (output_dir / MANIFEST_NAME).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return sel.missing


# ===========================================================================
# 10. One-shot driver
# ===========================================================================

def make_numpy_min(target, work_dir=None, *,
                   trace=None, output=None, force_trace=False,
                   trace_message=DEFAULT_TRACE_MESSAGE,
                   extra_resources=(),
                   load_methods=DEFAULT_LOAD_METHODS,
                   resource_suffixes=DEFAULT_RESOURCE_SUFFIXES):
    """Trace target (if needed) and build numpy_package_min.

    The target program is the single source of truth: resources are found
    from its load calls, so no separate file list has to be maintained.

    target          : program to be made runnable with Python + NumPy only
    work_dir        : directory for trace and output (default: target.parent)
    trace           : trace json (default: work_dir/PYAINO_TRACE_<stem>.json)
    output          : output dir (default: work_dir/numpy_package_min)
    force_trace     : retrace even if a valid trace exists
    trace_message   : instruction shown before the runtime trace
    extra_resources : resources not found by static analysis
                      (relative to target.parent, or absolute under it)

    Returns the output directory as Path.
    """
    target = Path(target).resolve()
    work_dir = target.parent if work_dir is None else Path(work_dir)
    trace = work_dir / f'PYAINO_TRACE_{target.stem}.json' if trace is None else Path(trace)
    output = work_dir / DEFAULT_OUTPUT_NAME if output is None else Path(output)

    # 1. resources
    resources = discover_resources(target, load_methods, resource_suffixes)
    for p in extra_resources:
        rel = relative_to_target(p, target)
        if rel not in resources:
            resources.append(rel)
    if not resources:
        raise ValueError(f'No resource files ({", ".join(resource_suffixes)}) '
                         f'found in {target}')
    check_source_set(target, resources)
    work_dir.mkdir(parents=True, exist_ok=True)

    print('=== pyaino numpy_min: settings ===')
    print('target =', target)
    for rel in resources:
        print('resource =', rel)
    print('trace  =', trace)
    print('output =', output)
    print()

    # 2. runtime trace (reused while valid)
    if force_trace or not trace_is_complete(trace, target):
        if trace.exists():
            print('Existing trace is stale, incomplete, or for another target; retracing.')
        print('=== pyaino numpy_min: runtime trace ===')
        if trace_message:
            print(trace_message)
        trace_usage(target, trace)
        print()

    # 3. build
    print('=== pyaino numpy_min: build ===')
    build_numpy_min(target, output_dir=output, trace=trace, resources=resources)
    print()
    print('completed =>', output)
    print('entry     =>', output / target.name)
    print('manifest  =>', output / MANIFEST_NAME)
    return output


# ===========================================================================
# 11. Command-line interface
# ===========================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Make a pyaino program runnable with Python + NumPy only.')
    sub = parser.add_subparsers(dest='command', required=True)
    source_help = 'pyaino package directory (default: the package containing pyaino.numpy_min)'

    p = sub.add_parser('make', help='discover resources, trace if needed, and build (recommended)')
    p.add_argument('target', type=Path, help='Python inference program using pyaino')
    p.add_argument('--work-dir', type=Path, default=None,
                   help='directory for trace and output (default: target directory)')
    p.add_argument('--trace', type=Path, default=None)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--force-trace', action='store_true')
    p.add_argument('--extra-resource', action='append', default=[],
                   help='resource not found by static analysis; repeatable')

    p = sub.add_parser('trace', help='run a target and record used pyaino symbols')
    p.add_argument('target', type=Path, help='Python inference program using pyaino')
    p.add_argument('--source', type=Path, default=None, help=source_help)
    p.add_argument('--trace-output', type=Path, default=Path('PYAINO_TRACE.json'))
    p.add_argument('--working-dir', type=Path, default=None,
                   help='current directory while running the target (default: target directory)')

    p = sub.add_parser('build', help='build the minimized NumPy-only package')
    p.add_argument('target', type=Path, help='Python inference program using pyaino')
    p.add_argument('--source', type=Path, default=None, help=source_help)
    p.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT_NAME))
    p.add_argument('--trace', type=Path, default=None,
                   help='trace file created by the trace command')
    p.add_argument('--keep-method', action='append', default=[],
                   help='normally stripped method that this target still needs; repeatable')
    p.add_argument('--resource', action='append', default=[],
                   help='runtime file/directory relative to the target; repeatable')

    args = parser.parse_args(argv)
    if args.command == 'make':
        make_numpy_min(args.target, args.work_dir, trace=args.trace, output=args.output,
                       force_trace=args.force_trace, extra_resources=args.extra_resource)
    elif args.command == 'trace':
        trace_usage(args.target, args.trace_output, args.source, args.working_dir)
    else:
        build_numpy_min(args.target, output_dir=args.output, source_dir=args.source,
                        trace=args.trace, keep_methods=args.keep_method,
                        resources=args.resource)


if __name__ == '__main__':
    main()
