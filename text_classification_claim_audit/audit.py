"""Offline documentation and AST evidence auditing."""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import re
import sys

from . import __version__
from .phrases import (TEXTCLASSIFICATION, NOTEXTCLASSIFICATION,
                      GENERIC_NAMES, SKLEARN_CLASSIFIERS)


def normalize(text):
    return ' '.join(text.lower().replace("n't", ' not').replace('-', ' ').split())


def documentation(text):
    """Yield paragraphs outside Markdown fences, preserving paragraph boundaries."""
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
            continue
        if match:
            fence = match[1]
            lines.append('')
        else:
            lines.append(line)
    return [normalize(p) for p in re.split(r'\n\s*\n', '\n'.join(lines)) if p.strip()]


def claims(text):
    positive = negative = False
    for paragraph in documentation(text):
        for phrase in sorted(NOTEXTCLASSIFICATION, key=lambda p: (-len(p), p)):
            paragraph, count = re.subn(r'\b' + re.escape(phrase) + r'\b', ' ', paragraph)
            negative |= bool(count)
        for phrase in sorted(TEXTCLASSIFICATION, key=lambda p: (-len(p), p)):
            if re.search(r'\b' + re.escape(phrase) + r'\b', paragraph):
                positive = True
                break
    return positive, negative


def evidence(source):
    """Recognize route pairs within one module without importing or executing it."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    nodes = list(ast.walk(tree))
    shadows = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    aliases = {}
    libraries = set()
    for n in nodes:
        if isinstance(n, ast.Import):
            for a in n.names:
                aliases[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
                libraries.add(a.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            module = n.module or ''
            libraries.add(module.split('.')[0])
            for a in n.names:
                aliases[a.asname or a.name] = module + '.' + a.name
    bindings = {}
    for n in nodes:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name):
                    value = n.value
                    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                            and value.func.attr == 'fit' and isinstance(value.func.value, ast.Name)
                            and value.func.value.id == target.id and target.id in bindings):
                        value = copy.deepcopy(value)
                        value.func.value = bindings[target.id]
                    bindings[target.id] = value
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value:
            bindings[n.target.id] = n.value

    def name(node):
        if isinstance(node, ast.Name):
            if node.id in shadows:
                return ''
            resolved = aliases.get(node.id, node.id).split('.')[-1]
            return '' if resolved in shadows else resolved
        if isinstance(node, ast.Attribute):
            # A locally defined route or import base shadows all matching routes.
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if node.attr in shadows or isinstance(base, ast.Name) and base.id in shadows:
                return ''
            return node.attr
        return ''

    def bound(node, seen=None):
        seen = set() if seen is None else seen
        if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
            return bound(bindings[node.id], seen | {node.id})
        return node

    def literal(node, choices):
        return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in choices

    def construction(node, route, seen=None):
        seen = set() if seen is None else seen
        if id(node) in seen:
            return False
        seen = seen | {id(node)}
        if not isinstance(node, ast.Call):
            return False
        f = node.func
        if route == 'pipeline':
            task = node.args[:1] + [k.value for k in node.keywords if k.arg == 'task']
            return 'transformers' in libraries and name(f) == 'pipeline' and any(literal(t, {'text-classification'}) for t in task)
        if route in ('model', 'nltk'):
            lib, method, names = ('transformers', 'from_pretrained', {'AutoModelForSequenceClassification', 'TFAutoModelForSequenceClassification'}) if route == 'model' else ('nltk', 'train', {'NaiveBayesClassifier'})
            return lib in libraries and isinstance(f, ast.Attribute) and f.attr == method and name(f.value) in names
        if route == 'textcat':
            return 'spacy' in libraries and name(f) == 'TextCategorizer'
        if route == 'classifier':
            if name(f) in set(SKLEARN_CLASSIFIERS) | {'Pipeline'}:
                return True
            return isinstance(f, ast.Attribute) and f.attr == 'fit' and construction(bound(f.value), 'classifier', seen)
        return False

    calls = [n for n in nodes if isinstance(n, ast.Call)]
    vectorizer = 'sklearn' in libraries and any(name(n.func) in {'TfidfVectorizer', 'CountVectorizer'} for n in calls)
    pipes = set()
    if 'spacy' in libraries:
        for n in calls:
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in {'add_pipe', 'create_pipe'} and isinstance(f.value, ast.Name) and f.value.id not in shadows and n.args and literal(n.args[0], {'textcat', 'textcat_multilabel'}):
                pipes.add(f.value.id)
    for n in calls:
        f = n.func
        if isinstance(f, ast.Attribute) and name(f) in GENERIC_NAMES:
            return True
        if isinstance(f, ast.Name) and f.id in aliases and name(f) in GENERIC_NAMES:
            return True
        if (any(construction(bound(f), route) for route in ('pipeline', 'model'))
                or isinstance(f, ast.Name) and construction(bound(f), 'textcat')):
            return True
        if isinstance(f, ast.Name) and f.id in pipes:
            return True
        if isinstance(f, ast.Attribute):
            if f.attr == 'pipe' and isinstance(f.value, ast.Name) and f.value.id in pipes:
                return True
            if f.attr == 'classify' and construction(bound(f.value), 'nltk'):
                return True
            if (vectorizer and f.attr in {'predict', 'predict_proba'}
                    and (isinstance(f.value, ast.Name) or isinstance(f.value, ast.Call)
                         and isinstance(f.value.func, ast.Attribute) and f.value.func.attr == 'fit')
                    and construction(bound(f.value), 'classifier')):
                return True
    return False


def audit(root, ignores=()):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise OSError('root must be a readable directory')
    positives, negatives, production, tests = [], [], [], []
    prefixes = [Path(p).as_posix().strip('/') for p in ignores]
    def ignored(relative):
        value = relative.as_posix()
        return any(value == p or value.startswith(p + '/') for p in prefixes)
    def walk_error(error):
        raise error
    for directory, dirs, files in os.walk(root, onerror=walk_error, followlinks=False):
        base = Path(directory)
        dirs[:] = sorted(d for d in dirs if not (base / d).is_symlink() and not ignored((base / d).relative_to(root)))
        for filename in sorted(files):
            path = base / filename
            rel = path.relative_to(root)
            if path.is_symlink() or ignored(rel):
                continue
            if (rel == Path('README.md') or path.suffix == '.md' and 'docs' in rel.parts):
                pos, neg = claims(path.read_text(encoding='utf-8'))
                if pos:
                    positives.append(rel.as_posix())
                if neg:
                    negatives.append(rel.as_posix())
            if path.suffix == '.py' and not {'fixtures', 'docs', 'examples'}.intersection(rel.parts):
                is_test = 'tests' in rel.parts or filename.startswith('test_') or filename.endswith('_test.py')
                if evidence(path.read_text(encoding='utf-8')):
                    (tests if is_test else production).append(rel.as_posix())
    findings = []
    if positives and not production:
        test_only = bool(tests)
        findings.append({'rule': 'documented-text-classification-claim-test-only' if test_only else 'documented-text-classification-claim-false', 'severity': 'MEDIUM' if test_only else 'HIGH', 'message': 'Documented claim has evidence only in tests.' if test_only else 'Documented claim has no production evidence.', 'paths': positives})
    if negatives and production:
        findings.append({'rule': 'documented-no-text-classification-claim-false', 'severity': 'HIGH', 'message': 'Documented denial conflicts with production evidence.', 'paths': negatives})
    return {'verdict': 'FAIL' if findings else 'PASS', 'findings': findings, 'production_evidence': production, 'test_evidence': tests}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, self.prog + ': error: ' + message + '\n')


def main(argv=None):
    parser = Parser(prog='text-classification-claim-audit')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    command = commands.add_parser('audit')
    command.add_argument('repo', nargs='?', default='.')
    command.add_argument('--json', action='store_true')
    command.add_argument('--verbose', action='store_true')
    command.add_argument('--ignore', action='append', default=[])
    command.add_argument('--fail-on', type=str.lower, choices=('high', 'medium'), default='high')
    command.add_argument('--version', action='version', version=__version__)
    args = parser.parse_args(argv)
    try:
        result = audit(args.repo, args.ignore)
    except (OSError, UnicodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result['verdict'])
        for finding in result['findings']:
            print('{severity} {rule}: {message}'.format(**finding))
        if args.verbose:
            print('Production evidence: ' + ', '.join(result['production_evidence']))
            print('Test evidence: ' + ', '.join(result['test_evidence']))
    return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in result['findings']) else 0
