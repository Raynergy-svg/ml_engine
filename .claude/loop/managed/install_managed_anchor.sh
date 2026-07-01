#!/usr/bin/env bash
# install_managed_anchor.sh — idempotent installer for the managed trust anchor.
#
# The ONE privileged step is the OPERATOR'S: copying the prepared files into the SYSTEM managed dir
# (root-owned, outside the repo, where the worker can't write). This script does NOT sudo itself.
#   - Run WITHOUT sudo: prints the exact privileged command to run (and a dry-run summary).
#   - Run WITH sudo (operator): performs the copy, sets ownership/permissions, and points you at the
#     verifier.
# After install, run:  python3 .claude/loop/managed/verify_managed_anchor.py
set -uo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"

case "$(uname -s)" in
  Darwin) DEST="/Library/Application Support/ClaudeCode" ;;
  Linux)  DEST="/etc/claude-code" ;;
  *)      echo "Windows: copy (as Administrator) managed-settings.json + ml_engine_gate_wrapper.py to"
          echo "  C:\\Program Files\\ClaudeCode\\   (NOT the deprecated C:\\ProgramData\\ClaudeCode\\)"
          exit 0 ;;
esac

WRAPPER_NAME="ml_engine_gate_wrapper.py"
do_install() {
  mkdir -p "$DEST"
  cp "$SRC/managed-settings.json" "$DEST/managed-settings.json"
  cp "$SRC/$WRAPPER_NAME"          "$DEST/$WRAPPER_NAME"
  chown root:wheel "$DEST/managed-settings.json" "$DEST/$WRAPPER_NAME" 2>/dev/null || true
  chmod 0644 "$DEST/managed-settings.json"
  chmod 0755 "$DEST/$WRAPPER_NAME"
  echo "Installed (root-owned) to: $DEST"
  echo "Next: python3 \"$SRC/verify_managed_anchor.py\""
}

if [ "$(id -u)" = "0" ]; then
  do_install
else
  echo "Not root — nothing was changed. The operator's one privileged step:"
  echo
  echo "  sudo bash \"$SRC/install_managed_anchor.sh\""
  echo
  echo "…or run these three commands manually:"
  echo "  sudo mkdir -p \"$DEST\""
  echo "  sudo cp \"$SRC/managed-settings.json\" \"$DEST/\""
  echo "  sudo cp \"$SRC/$WRAPPER_NAME\" \"$DEST/\""
  echo
  echo "Then verify (no sudo needed):"
  echo "  python3 \"$SRC/verify_managed_anchor.py\""
fi
