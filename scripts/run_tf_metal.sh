#!/usr/bin/env bash
set -euo pipefail

# Always run using the pinned conda env to avoid using base Python by accident.
# Use activation (not `conda run`) to avoid hangs in some setups.
if [[ -f "/Users/mirelacertan/miniforge3/etc/profile.d/conda.sh" ]]; then
	# shellcheck disable=SC1091
	source "/Users/mirelacertan/miniforge3/etc/profile.d/conda.sh"
fi
conda activate tf-metal >/dev/null 2>&1 || true
exec python "$@"
