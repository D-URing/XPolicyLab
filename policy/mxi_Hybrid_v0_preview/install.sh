#!/bin/bash
set -euo pipefail
: "${MXI_WHEEL:?Set the path of the authorized MXI wheel}"
: "${MXI_WHEEL_SHA256:?Set its independently supplied SHA-256}"
python - "${MXI_WHEEL}" "${MXI_WHEEL_SHA256}" <<'PY'
import hashlib, pathlib, sys
path = pathlib.Path(sys.argv[1])
if hashlib.sha256(path.read_bytes()).hexdigest() != sys.argv[2]:
    raise SystemExit('Inference wheel checksum mismatch')
PY
python -m pip install --no-deps "${MXI_WHEEL}"
python -c 'from mxi.adapters.robodojo_online import Model'
