#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/PLDM_hieral}"
TAR_NAME="${2:-pldm_colab_6m.tar.gz}"

bash "${ROOT_DIR}/PLDM_hieral/colab_pack.sh" "${OUT_DIR}" "${TAR_NAME}"
