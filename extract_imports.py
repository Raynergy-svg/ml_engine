#!/usr/bin/env python3
"""
Extract all import statements from Python files for dependency analysis.
This script scans the codebase and extracts import statements with classification.
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re

# Standard library modules (for classification)
STANDARD_LIBRARY_MODULES = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
    'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect', 'builtins',
    'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs',
    'codeop', 'collections', 'colorsys', 'compileall', 'concurrent', 'configparser',
    'contextlib', 'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv', 'ctypes',
    'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils',
    'doctest', 'email', 'encodings', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp',
    'fileinput', 'fnmatch', 'formatter', 'fractions', 'ftplib', 'functools', 'gc',
    'getopt', 'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq',
    'hmac', 'html', 'http', 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io',
    'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3', 'linecache', 'locale',
    'logging', 'lzma', 'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes', 'mmap',
    'modulefinder', 'msilib', 'msvcrt', 'multiprocessing', 'netrc', 'nis', 'nntplib',
    'numbers', 'operator', 'optparse', 'os', 'ossaudiodev', 'pathlib', 'pdb', 'pickle',
    'pickletools', 'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posix', 'posixpath',
    'pprint', 'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc',
    'queue', 'quopri', 'random', 're', 'readline', 'reprlib', 'resource', 'rlcompleter',
    'runpy', 'sched', 'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil',
    'signal', 'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd',
    'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct', 'subprocess',
    'sunau', 'symbol', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile',
    'telnetlib', 'tempfile', 'termios', 'test', 'textwrap', 'threading', 'time', 'timeit',
    'tkinter', 'token', 'tokenize', 'tomllib', 'trace', 'traceback', 'tracemalloc', 'tty',
    'turtle', 'turtledemo', 'types', 'typing', 'typing_extensions', 'unicodedata', 'unittest',
    'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser', 'winreg',
    'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport',
    'zoneinfo', 'zlib'
}


def should_exclude_path(path: Path) -> bool:
    """Check if a path should be excluded from scanning."""
    path_str = str(path)
    
    # Exclude specific directories
    exclude_dirs = ['.venv', 'venv', 'env', '__pycache__', 'legacy_quarantine', 
                    'node_modules', '.git', '.idea', '.vscode']
    
    for exclude_dir in exclude_dirs:
        if exclude_dir in path.parts:
            return True
    
    return False


def get_python_files(root_dir: Path) -> List[Path]:
    """Get all Python files in the project, excluding certain directories."""
    python_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Filter out excluded directories in-place
        dirs[:] = [d for d in dirs if not should_exclude_path(Path(root) / d)]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                if not should_exclude_path(file_path):
                    python_files.append(file_path)
    
    return sorted(python_files)


def classify_import(module_name: str, file_path: Path, project_root: Path) -> str:
    """
    Classify an import as standard, third-party, local, or relative.
    
    Returns: 'standard', 'third_party', 'local', or 'relative'
    """
    # Check for relative imports
    if module_name.startswith('.') or module_name.startswith('..'):
        return 'relative'
    
    # Get the base module name (first part before any dots)
    base_module = module_name.split('.')[0]
    
    # Check if it's a standard library module
    if base_module in STANDARD_LIBRARY_MODULES:
        return 'standard'
    
    # Check if it's a local project import
    # Local imports typically start with 'src' or other project directories
    file_dir = file_path.parent
    relative_path = file_dir.relative_to(project_root)
    
    # Check if the module could be a local import
    # by seeing if it matches any top-level directory in the project
    project_dirs = [d.name for d in project_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if base_module in project_dirs:
        return 'local'
    
    # If it's not standard and not local, it's likely third-party
    return 'third_party'


def extract_imports_from_file(file_path: Path, project_root: Path) -> Dict:
    """
    Extract all import statements from a Python file.
    
    Returns a dictionary with file info and imports.
    """
    result = {
        'file_path': str(file_path.relative_to(project_root)),
        'full_path': str(file_path),
        'imports': [],
        'dynamic_imports': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content, filename=str(file_path))
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_type = classify_import(alias.name, file_path, project_root)
                    result['imports'].append({
                        'statement': f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                        'module': alias.name,
                        'alias': alias.asname,
                        'type': import_type,
                        'line': node.lineno
                    })
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                import_type = classify_import(module, file_path, project_root)
                
                for alias in node.names:
                    import_statement = f"from {module} import {alias.name}"
                    if alias.asname:
                        import_statement += f" as {alias.asname}"
                    
                    result['imports'].append({
                        'statement': import_statement,
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'type': import_type,
                        'line': node.lineno
                    })
            
            # Check for dynamic imports
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == '__import__':
                        result['dynamic_imports'].append({
                            'type': '__import__',
                            'line': node.lineno
                        })
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'import_module':
                        result['dynamic_imports'].append({
                            'type': 'importlib.import_module',
                            'line': node.lineno
                        })
        
        # Also check for importlib usage via regex
        importlib_pattern = r'importlib\.import_module\s*\('
        for match in re.finditer(importlib_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            if not any(d['line'] == line_num for d in result['dynamic_imports']):
                result['dynamic_imports'].append({
                    'type': 'importlib.import_module',
                    'line': line_num
                })
        
        # Check for __import__ via regex
        __import__pattern = r'__import__\s*\('
        for match in re.finditer(__import__pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            if not any(d['line'] == line_num for d in result['dynamic_imports']):
                result['dynamic_imports'].append({
                    'type': '__import__',
                    'line': line_num
                })
    
    except SyntaxError as e:
        result['error'] = f"Syntax error: {e}"
    except Exception as e:
        result['error'] = f"Error reading file: {e}"
    
    return result


def main():
    """Main function to extract imports from all Python files."""
    # Get project root
    project_root = Path(__file__).parent
    
    print(f"Scanning project at: {project_root}")
    print("=" * 80)
    
    # Get all Python files
    python_files = get_python_files(project_root)
    print(f"Found {len(python_files)} Python files to analyze")
    print()
    
    # Extract imports from each file
    all_imports = {}
    import_stats = {
        'standard': 0,
        'third_party': 0,
        'local': 0,
        'relative': 0,
        'dynamic': 0
    }
    
    for i, file_path in enumerate(python_files, 1):
        print(f"[{i}/{len(python_files)}] Processing: {file_path.relative_to(project_root)}")
        
        file_data = extract_imports_from_file(file_path, project_root)
        all_imports[file_data['file_path']] = file_data
        
        # Update statistics
        for imp in file_data['imports']:
            import_type = imp['type']
            if import_type in import_stats:
                import_stats[import_type] += 1
        
        import_stats['dynamic'] += len(file_data['dynamic_imports'])
    
    # Prepare output data
    output_data = {
        'project_root': str(project_root),
        'scan_date': '2026-01-27T19:15:00Z',
        'total_files': len(python_files),
        'total_imports': sum(import_stats.values()),
        'statistics': import_stats,
        'files': all_imports
    }
    
    # Save to JSON file
    output_path = project_root / 'plans' / 'dependency_import_data.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Output saved to: {output_path}")
    print()
    print("SUMMARY STATISTICS:")
    print(f"  Total Python files scanned: {output_data['total_files']}")
    print(f"  Total imports found: {output_data['total_imports']}")
    print()
    print("Import breakdown:")
    print(f"  Standard library: {import_stats['standard']}")
    print(f"  Third-party: {import_stats['third_party']}")
    print(f"  Local project: {import_stats['local']}")
    print(f"  Relative: {import_stats['relative']}")
    print(f"  Dynamic: {import_stats['dynamic']}")
    
    return output_data


if __name__ == '__main__':
    main()
