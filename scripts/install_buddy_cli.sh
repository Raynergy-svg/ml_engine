#!/usr/bin/env bash
#===============================================================================
# Install Buddy CLI to PATH
# Creates a symlink in ~/.local/bin for global access
#===============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${HOME}/.local/bin"
TARGET="${TARGET_DIR}/buddy"

# Create directory if needed
mkdir -p "${TARGET_DIR}"

# Remove old version if exists
rm -f "${TARGET}"

# Create symlink to repo's buddy script
ln -s "${REPO_ROOT}/buddy" "${TARGET}"

echo "✓ Installed: ${TARGET}"
echo ""
echo "Make sure ~/.local/bin is in your PATH:"
echo '  export PATH="$HOME/.local/bin:$PATH"'
echo ""
echo "Then you can run:"
echo "  buddy help"
echo "  buddy status"
echo "  buddy predict"
echo "  buddy train --oanda-live"
