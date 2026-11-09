#!/usr/bin/env bash
# Create/install the author's vendored runtime in its own uv environment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEM_ROOT="${ROOT}/external/libero-mem"
EXPECTED_REV="0eee0defa4024e51511b6e15a79f30e9620209de"
VENV="${ROOT}/.venv_libero_mem_author_candidate"
UV="${UV:-$(command -v uv || true)}"
GIT_BIN="${MEM_EVAL_GIT:-$(command -v git.exe || command -v git)}"
if [[ "${MEM_ROOT}" == /mnt/* ]] && command -v git.exe >/dev/null 2>&1; then
  NATIVE_MEM_ROOT="$(wslpath -w "${MEM_ROOT}")"
else
  NATIVE_MEM_ROOT="${MEM_ROOT}"
fi

if [[ -z "${UV}" ]]; then
  echo "uv is required to create the isolated Python 3.10.22 environment." >&2
  exit 2
fi

actual_rev="$("${GIT_BIN}" -C "${NATIVE_MEM_ROOT}" rev-parse HEAD)"
if [[ "${actual_rev}" != "${EXPECTED_REV}" ]]; then
  echo "Pinned LIBERO-Mem revision mismatch: expected ${EXPECTED_REV}, found ${actual_rev}." >&2
  exit 2
fi

tracked_changes="$("${GIT_BIN}" -C "${NATIVE_MEM_ROOT}" status --porcelain --untracked-files=no)"
if [[ -n "${tracked_changes}" ]]; then
  echo "Tracked author source files are modified; refusing to install an altered source tree." >&2
  exit 2
fi

if [[ ! -x "${VENV}/bin/python" ]] || [[ "$("${VENV}/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)" != "3.10.22" ]]; then
  "${UV}" venv --clear --python 3.10.22 "${VENV}"
fi

PYTHON="${VENV}/bin/python"
"${PYTHON}" - <<'PY'
import sys
if sys.version_info[:3] != (3, 10, 22):
    raise SystemExit(f"Expected project-selected Python 3.10.22, found {sys.version.split()[0]}")
PY

# Keep policy and training packages in their existing environments. This venv
# holds the simulator, source task code, and physical-state diagnostics only.
"${UV}" pip install --python "${PYTHON}" -r "${ROOT}/envs/libero_mem_author_candidate.txt"

# Install the author's vendored dependency forks only if this environment does
# not already point at the pinned source. Building robosuite's editable wheel
# scans a large tree on /mnt/d, so avoid repeating that work on manifest refresh.
ensure_editable_source() {
  local package="$1" version="$2" source="$3"
  if ! "${PYTHON}" - "${package}" "${version}" "${source}" <<'PY' >/dev/null 2>&1
import importlib.metadata
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

name, expected_version, expected_source = sys.argv[1:]
try:
    dist = importlib.metadata.distribution(name)
    direct_url = json.loads(dist.read_text("direct_url.json"))
    installed_source = Path(unquote(urlparse(direct_url["url"]).path)).resolve()
    ok = dist.version == expected_version and installed_source == Path(expected_source).resolve()
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
  then
    "${UV}" pip install --python "${PYTHON}" --no-deps -e "${source}"
  fi
}

ensure_editable_source robomimic 0.3.0 "${MEM_ROOT}/thirdparty/robomimic"
ensure_editable_source robosuite 1.4.1 "${MEM_ROOT}/thirdparty/robosuite"

# The LIBERO source itself is imported from the pinned checkout by diagnostic
# scripts; avoid pip-installing it because that rewrites tracked egg-info.
if "${PYTHON}" -c 'import importlib.metadata as m; m.version("libero")' >/dev/null 2>&1; then
  "${UV}" pip uninstall --python "${PYTHON}" -y libero
fi

"${PYTHON}" - "${MEM_ROOT}/thirdparty/robosuite" <<'PY'
import importlib.metadata
import importlib.util
from pathlib import Path
import sys

expected = Path(sys.argv[1]).resolve()
actual = Path(importlib.util.find_spec("robosuite").origin).resolve()
try:
    actual.relative_to(expected)
except ValueError:
    raise SystemExit(f"robosuite was imported from outside the author fork: {actual}")
if importlib.metadata.version("robosuite") != "1.4.1":
    raise SystemExit(f"Expected vendored robosuite 1.4.1, found {importlib.metadata.version('robosuite')}")
if importlib.metadata.version("mujoco") != "3.2.2":
    raise SystemExit(f"Expected author-pinned MuJoCo 3.2.2, found {importlib.metadata.version('mujoco')}")
expected = {
    "bddl": "3.5.0",
    "cloudpickle": "3.0.0",
    "easydict": "1.13",
    "egl_probe": "1.0.2",
    "evdev": "1.7.1",
    "glfw": "2.7.0",
    "gym": "0.26.2",
    "h5py": "3.11.0",
    "imageio": "2.35.1",
    "imageio-ffmpeg": "0.5.1",
    "matplotlib": "3.9.2",
    "numba": "0.60.0",
    "numpy": "1.23.5",
    "opencv-python": "4.10.0.84",
    "Pillow": "10.4.0",
    "PyOpenGL": "3.1.7",
    "pyparsing": "3.1.2",
    "PyYAML": "6.0.2",
    "scipy": "1.14.1",
    "termcolor": "2.4.0",
}
for package, version in expected.items():
    actual_version = importlib.metadata.version(package)
    if actual_version != version:
        raise SystemExit(f"Expected author-pinned {package} {version}, found {actual_version}")
print("Author fork loaded:", actual)
print("robosuite:", importlib.metadata.version("robosuite"))
print("mujoco:", importlib.metadata.version("mujoco"))
PY

RUNTIME_DIR="${ROOT}/experiments/runtime/libero_mem_author_candidate"
mkdir -p "${RUNTIME_DIR}"
"${UV}" pip freeze --python "${PYTHON}" > "${RUNTIME_DIR}/pip-freeze.txt"
"${PYTHON}" - "${ROOT}" "${MEM_ROOT}" "${RUNTIME_DIR}" <<'PY'
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

root, mem_root, runtime_dir = map(Path, sys.argv[1:])
manifest = {
    "classification": "AUTHOR_SOURCE_WITH_AUTHOR_SIMULATOR_DEPENDENCY_SUBSET",
    "upstream_repository": "https://github.com/libero-mem/libero-mem",
    "upstream_revision": "0eee0defa4024e51511b6e15a79f30e9620209de",
    "python": sys.version.split()[0],
    "robosuite": importlib.metadata.version("robosuite"),
    "robosuite_import": str(Path(importlib.util.find_spec("robosuite").origin).resolve()),
    "mujoco": importlib.metadata.version("mujoco"),
    "robomimic": importlib.metadata.version("robomimic"),
    "runtime_note": "The environment pins the simulator-relevant dependency subset copied from the pinned author's scripts/requirements.txt; it is not the full policy/training stack. The source file pins MuJoCo 3.2.2. Python 3.10.22 is a project choice because the public author setup does not pin Python.",
}
(runtime_dir / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(runtime_dir / "runtime-manifest.json")
PY
