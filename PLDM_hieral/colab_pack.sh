#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/PLDM_hieral}"
TAR_NAME="${2:-pldm_colab.tar.gz}"
TAR_PATH="${OUT_DIR}/${TAR_NAME}"

EXTRA_EXCLUDES=()
if [[ "${TAR_PATH}" == "${ROOT_DIR}"/* ]]; then
  REL_TAR_PATH="${TAR_PATH#${ROOT_DIR}/}"
  EXTRA_EXCLUDES+=(--exclude="${REL_TAR_PATH}")
fi

mkdir -p "${OUT_DIR}"

tar \
  "${EXTRA_EXCLUDES[@]}" \
  --exclude="PLDM_hieral/outputs" \
  -czf "${TAR_PATH}" \
  -C "${ROOT_DIR}" \
  pldm \
  pldm_envs \
  PLDM_hieral \
  requirements.txt \
  pyproject.toml \
  readme.md \
  LICENSE

echo "Created ${TAR_PATH}"
echo "Upload it to Colab and extract with: tar -xzf ${TAR_NAME}"
