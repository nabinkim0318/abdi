#!/usr/bin/env bash
# Generate or verify requirements.txt from the Poetry lock.
#
# pyproject.toml is the declared dependency source.
# poetry.lock is the resolved graph.
# requirements.txt is the generated runtime install artifact — do not edit it
# by hand. Regenerate with: make requirements
set -euo pipefail

EXPORT_PLUGIN_VERSION="${ABDI_EXPORT_PLUGIN_VERSION:-1.9.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMITTED="${ROOT}/requirements.txt"

die() {
  echo "error: $*" >&2
  exit 1
}

ensure_poetry_export() {
  command -v poetry >/dev/null 2>&1 || die "poetry is required (Poetry 2.x)"
  if poetry export --help >/dev/null 2>&1; then
    return 0
  fi
  echo "poetry-plugin-export is not available; installing ${EXPORT_PLUGIN_VERSION}" >&2
  poetry self add "poetry-plugin-export==${EXPORT_PLUGIN_VERSION}"
  poetry export --help >/dev/null 2>&1 || die "poetry export is still unavailable after installing poetry-plugin-export==${EXPORT_PLUGIN_VERSION}"
}

poetry_export_to() {
  local out="$1"
  (cd "$ROOT" && poetry export \
    --only main \
    --without-hashes \
    --format requirements.txt \
    --output "$out")
}

generate_to() {
  local out="$1"
  local tmp
  tmp="$(mktemp)"
  # Always write to a temp path first so a failed export cannot clobber the
  # destination and so CI never treats a failed tool as a clean check.
  if [[ -n "${ABDI_EXPORT_CMD:-}" ]]; then
    ABDI_EXPORT_OUT="$tmp" eval "${ABDI_EXPORT_CMD}"
  else
    ensure_poetry_export
    poetry_export_to "$tmp"
  fi
  if [[ ! -s "$tmp" ]]; then
    rm -f "$tmp"
    die "requirements export produced an empty file"
  fi
  mv "$tmp" "$out"
}

cmd_export() {
  generate_to "$COMMITTED"
  echo "Wrote ${COMMITTED}"
}

cmd_check() {
  local tmpdir tmp status
  tmpdir="$(mktemp -d)"
  tmp="${tmpdir}/requirements.txt"
  generate_to "$tmp"
  status=0
  if ! diff -u "$COMMITTED" "$tmp"; then
    status=1
  fi
  rm -rf "$tmpdir"
  if [[ "$status" -ne 0 ]]; then
    cat >&2 <<'EOF'

requirements.txt is out of sync with pyproject.toml / poetry.lock.

Regenerate the runtime artifact locally (Poetry 2.x):

  make requirements

Then commit the updated requirements.txt.
EOF
    exit 1
  fi
  echo "requirements.txt matches the Poetry runtime export."
}

usage() {
  cat <<'EOF'
Usage: scripts/requirements-contract.sh <export|check>

  export  Write requirements.txt from poetry.lock (runtime/main only)
  check   Export to a temp file and fail if it differs from the committed file
EOF
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    export) cmd_export ;;
    check) cmd_check ;;
    *) usage >&2; exit 2 ;;
  esac
}

main "$@"
