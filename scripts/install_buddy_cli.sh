#!/usr/bin/env bash
set -euo pipefail

# Installs a `buddy` launcher into ~/.local/bin that runs this repo's Buddy CLI.
# This is useful when you previously had a stale/broken `buddy` script on PATH.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${HOME}/.local/bin/buddy"

mkdir -p "$(dirname "$target")"

cat >"$target" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="__REPO_ROOT__"

# Prefer running via conda env if available so dependencies resolve.
ENV_NAME="${BUDDY_CONDA_ENV:-ml_engine}"

# IMPORTANT: Use `--no-capture-output` so stdin/stdout stay TTY.
# Otherwise `conda run` captures streams and Buddy's interactive wizard
# won't trigger (it checks isatty()).


# Prefer explicit Miniforge binary when available (reliable in non-interactive shells)
if [[ -x "$HOME/miniforge3/bin/conda" ]]; then
  "$HOME/miniforge3/bin/conda" run -n "$ENV_NAME" --no-capture-output python "$REPO_ROOT/main.py" buddy "$@"
  exit $?
fi

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  "${CONDA_EXE}" run -n "$ENV_NAME" --no-capture-output python "$REPO_ROOT/main.py" buddy "$@"
  exit $?
fi

if command -v conda >/dev/null 2>&1; then
  conda run -n "$ENV_NAME" --no-capture-output python "$REPO_ROOT/main.py" buddy "$@"
  exit $?
fi

# Fall back to current python (requires you to be in the right env already).
python "$REPO_ROOT/main.py" buddy "$@"
EOF

# Replace placeholder safely.
python - <<PY
import pathlib
target = pathlib.Path(r"$target")
text = target.read_text()
text = text.replace("__REPO_ROOT__", r"$repo_root")
target.write_text(text)
PY

chmod +x "$target"

echo "Installed: $target"
echo "If you still hit the old script, ensure ~/.local/bin is early in PATH, or run: hash -r"
