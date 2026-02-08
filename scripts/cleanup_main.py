#!/usr/bin/env python3
"""Remove duplicate code from main.py that has been extracted to modules."""

import re
import sys
from pathlib import Path


def main():
    main_py = Path(__file__).parent.parent / "main.py"
    
    # Read file
    content = main_py.read_text()
    original_len = len(content)
    changes_made = 0
    
    # Remove VALID_OANDA_INSTRUMENTS and related functions
    # Pattern: from "# Valid OANDA FX instruments" to just before "def launch_buddy_repl_from_wizard"
    pattern1 = r'\n# Valid OANDA FX instruments.*?(?=\ndef launch_buddy_repl_from_wizard)'
    match1 = re.search(pattern1, content, re.DOTALL)
    if match1:
        print(f"Found instrument validation block: {len(match1.group(0))} chars")
        content = re.sub(pattern1, '\n', content, flags=re.DOTALL)
        changes_made += 1
    else:
        print("Instrument validation block not found")
    
    # Remove model builder functions
    # Pattern: from "def _build_buddy_model(" to just before "def train_buddy("
    pattern2 = r'\n\ndef _build_buddy_model\(.*?(?=\n\ndef train_buddy\()'
    match2 = re.search(pattern2, content, re.DOTALL)
    if match2:
        print(f"Found model builder block: {len(match2.group(0))} chars")
        content = re.sub(pattern2, '\n', content, flags=re.DOTALL)
        changes_made += 1
    else:
        print("Model builder block not found")
    
    if changes_made > 0:
        main_py.write_text(content)
        print(f"\nFile updated. Original: {original_len} chars, New: {len(content)} chars")
        print(f"Removed: {original_len - len(content)} chars")
    else:
        print("\nNo changes made")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
