import ast
import sys

files = [
    'cli/io_utils.py',
    'cli/config.py',
    'cli/models.py',
    'cli/buddy_helpers.py',
    'main.py',
    'src/training/training_config.py',
]

ok = True
for f in files:
    try:
        ast.parse(open(f).read())
        print(f'{f}: OK')
    except SyntaxError as e:
        print(f'{f}: SYNTAX ERROR - {e}')
        ok = False

if ok:
    print('\nAll syntax checks passed!')
else:
    print('\nSome files have syntax errors!')
    sys.exit(1)
