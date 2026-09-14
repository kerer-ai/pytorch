#!/usr/bin/bash
# Install CANN toolkit for Ascend NPU.
# Usage: CANN_CHIP=A1 ./install_cann.sh
#   CANN_CHIP: A1 (Ascend 910), A2 (Ascend 910b), A3 (Ascend A3)
#
# CANN packages come from either of two sources:
#   1. Public repo (default): fixed version downloaded from ascend-repo.
#   2. OBS share link: set OBS_SHARE_URL to an e-share link
#      (https://e-share.obs-website.<region>.myhuaweicloud.com?v2token=...)
#      whose v2token works as an obsutil authorization code. Packages are
#      downloaded with obsutil share-cp. Required in this mode:
#        OBS_SHARE_URL     the e-share link
#        OBS_ACCESS_CODE   access code (提取码) for the share
#      Optional:
#        CANN_VERSION      expected version, e.g. 9.2.0-20260910200430;
#                          validated against the share, auto-discovered when unset
# Automatically detects architecture (x86_64 / aarch64).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANN_CHIP="${CANN_CHIP:-A1}"
ARCH=$(uname -m)

CANN_VERSION_PUBLIC="9.1.0-beta.3"
CANN_BASE_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.T6"

case "${ARCH}" in
  x86_64)  ARCH_TAG="x86_64" ;;
  aarch64) ARCH_TAG="aarch64" ;;
  *)       echo "Unsupported architecture: ${ARCH}"; exit 1 ;;
esac

case "${CANN_CHIP}" in
  A1) OPS_PACKAGE="Ascend-cann-910-ops"  OPS_GLOB="Ascend-cann-910-ops*"  ;;
  A2) OPS_PACKAGE="Ascend-cann-910b-ops" OPS_GLOB="Ascend-cann-910b-ops*" ;;
  A3) OPS_PACKAGE="Ascend-cann-A3-ops"   OPS_GLOB="Ascend-cann-A3-ops*"   ;;
  *)  echo "Unsupported CANN_CHIP: ${CANN_CHIP} (expected A1, A2 or A3)"; exit 1 ;;
esac

echo "Installing CANN ${CANN_CHIP} for ${ARCH}..."

# ---------------------------------------------------------------- share mode
if [[ -n "${OBS_SHARE_URL:-}" ]]; then
  if [[ -z "${OBS_ACCESS_CODE:-}" ]]; then
    echo "ERROR: OBS_ACCESS_CODE (share access code) is required when OBS_SHARE_URL is set."
    exit 1
  fi
  # shellcheck source=obs_share.sh
  source "${SCRIPT_DIR}/obs_share.sh"

  if ! command -v obsutil >/dev/null 2>&1; then
    echo "obsutil not found, installing via install_obs.sh ..."
    bash "${SCRIPT_DIR}/install_obs.sh"
  fi

  V2TOKEN=$(obs_share_extract_token "${OBS_SHARE_URL}")
  if [[ -z "${V2TOKEN}" ]]; then
    echo "ERROR: no v2token found in OBS_SHARE_URL."
    exit 1
  fi
  AUTH_FILE=$(obs_share_write_auth_file "${V2TOKEN}")
  trap 'rm -f "${AUTH_FILE}"' EXIT

  if ! obs_share_discover "${AUTH_FILE}" "${OBS_ACCESS_CODE}"; then
    exit 1
  fi
  if [[ -n "${CANN_VERSION:-}" && "${CANN_VERSION}" != "${OBS_SHARE_VERSION}" ]]; then
    echo "ERROR: CANN_VERSION [${CANN_VERSION}] does not match the share version [${OBS_SHARE_VERSION}]."
    exit 1
  fi
  CANN_VERSION="${OBS_SHARE_VERSION}"
  echo "Using CANN ${CANN_VERSION} from OBS share (prefix: ${OBS_SHARE_PREFIX})"
  SHARE_RUN_OBJECTS=$(obs_share_list_run_objects "${AUTH_FILE}" "${OBS_ACCESS_CODE}" "${OBS_SHARE_PREFIX}" "${ARCH}")
  SHARE_MODE=1
else
  # -------------------------------------------------------------- public mode
  CANN_VERSION="${CANN_VERSION_PUBLIC}"
  TOOLKIT_URL="${CANN_BASE_URL}/Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH_TAG}.run"
  OPS_URL="${CANN_BASE_URL}/${OPS_PACKAGE}_${CANN_VERSION}_linux-${ARCH_TAG}.run"
  NNAL_URL="${CANN_BASE_URL}/Ascend-cann-nnal_${CANN_VERSION}_linux-${ARCH_TAG}.run"
  SHARE_MODE=0
fi

# ------------------------------------------------------------------- install
echo "=== Creating HwHiAiUser user and group ==="
groupadd -f HwHiAiUser
id -u HwHiAiUser >/dev/null 2>&1 || useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

rm -rf cann
mkdir -p cann && cd cann

echo "=== Downloading CANN packages ==="
if [[ "${SHARE_MODE}" == "1" ]]; then
  obs_share_download_pkg "${AUTH_FILE}" "${OBS_ACCESS_CODE}" "${SHARE_RUN_OBJECTS}" \
    "Ascend-cann-toolkit_*.run" >/dev/null
  obs_share_download_pkg "${AUTH_FILE}" "${OBS_ACCESS_CODE}" "${SHARE_RUN_OBJECTS}" \
    "${OPS_PACKAGE}_*.run" >/dev/null
  obs_share_download_pkg "${AUTH_FILE}" "${OBS_ACCESS_CODE}" "${SHARE_RUN_OBJECTS}" \
    "Ascend-cann-nnal_*.run" >/dev/null
else
  curl -O "${TOOLKIT_URL}"
  curl -O "${OPS_URL}"
  curl -O "${NNAL_URL}"
fi
echo "Download complete."

chmod +x Ascend-cann*.run

echo "=== Installing CANN toolkit ==="
./Ascend-cann-toolkit*.run --full --quiet --install-path=/usr/local/Ascend

# Some CANN versions install to versioned paths (e.g. cann-9.0.0-beta.2)
# instead of /usr/local/Ascend/cann/. Fix broken symlinks so that sourcing
# set_env.sh works both during the rest of this install and at runtime.
if [ ! -f /usr/local/Ascend/cann/set_env.sh ]; then
  CANN_REAL_DIR=$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | head -1)
  if [ -n "${CANN_REAL_DIR}" ]; then
    ln -sf "${CANN_REAL_DIR}" /usr/local/Ascend/cann
    echo "Fixed: linked ${CANN_REAL_DIR} -> /usr/local/Ascend/cann"
  fi
fi

source /usr/local/Ascend/cann/set_env.sh
echo "toolkit install success"

echo "=== Installing CANN ops ==="
./${OPS_GLOB}.run --install --quiet --install-path=/usr/local/Ascend
echo "ops install success"

echo "=== Installing CANN nnal ==="
./Ascend-cann-nnal*.run --install --quiet --install-path=/usr/local/Ascend
source /usr/local/Ascend/nnal/atb/set_env.sh
echo "nnal install success"

rm -rf *
echo "CANN ${CANN_CHIP} installation complete."
