#!/usr/bin/bash
# Build torch-npu CI Docker images.
#
# Usage:
#   ./docker_build.sh <TAG>
#
# Builder: torch-npu-builder-<ARCH>-torch<PYTORCH_VERSION>
# Test:    torch-npu-test-<ARCH>-cann<CHIP>-py<PYTHON_VERSION>-torch<PYTORCH_VERSION>-cann<CANN_VERSION>
#          (test tags always carry the CANN version, e.g.
#           ...-torch-master-cann9.1.0-beta.3 for the public repo, or
#           ...-torch-master-cann9.2.0-20260910200430 for an OBS share link)
#
# Examples:
#   ./docker_build.sh torch-npu-builder-x86_64-torch2.13.0
#   ./docker_build.sh torch-npu-builder-aarch64-torch-master
#   ./docker_build.sh torch-npu-test-aarch64-cann-a2-py3.10-torch2.13.0
#   ./docker_build.sh torch-npu-test-x86_64-cann-a1-py3.10-torch-master
#   ./docker_build.sh torch-npu-test-aarch64-cann-a3-py3.10-torch-master
#
# Environment variables (CANN from OBS share link, test images only):
#   OBS_SHARE_URL    e-share link (with v2token); requires obsutil on PATH
#   OBS_ACCESS_CODE  access code (提取码) for the share, required with the URL
#   CANN_VERSION     optional expected version, validated against the share
#   DRY_RUN=1        print resolved configuration and tag, skip docker build
#   TAG_OUT=<file>   write the final image tag to this file

set -euo pipefail

BASE_TAG="${1:?Usage: $0 <TAG>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT="${SCRIPT_DIR}"

case "$BASE_TAG" in
  # --- v2.13.0 builder ---
  torch-npu-builder-x86_64-torch2.13.0)
    IMAGE_TYPE=builder
    ARCH=x86_64
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-builder-aarch64-torch2.13.0)
    IMAGE_TYPE=builder
    ARCH=aarch64
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  # --- v2.13.0 test ---
  torch-npu-test-x86_64-cann-a1-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A1
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-test-x86_64-cann-a2-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A2
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-test-x86_64-cann-a3-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A3
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-test-aarch64-cann-a1-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A1
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-test-aarch64-cann-a2-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A2
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-test-aarch64-cann-a3-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A3
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  # --- master (nightly) builder ---
  torch-npu-builder-x86_64-torch-master)
    IMAGE_TYPE=builder
    ARCH=x86_64
    PYTORCH_VERSION=2.14.0.dev20260708
    VERSION_DIR=master
    ;;
  torch-npu-builder-aarch64-torch-master)
    IMAGE_TYPE=builder
    ARCH=aarch64
    PYTORCH_VERSION=2.14.0.dev20260708
    VERSION_DIR=master
    ;;
  # --- master (nightly) test ---
  torch-npu-test-x86_64-cann-a1-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A1
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  torch-npu-test-x86_64-cann-a2-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A2
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  torch-npu-test-x86_64-cann-a3-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A3
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  torch-npu-test-aarch64-cann-a1-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A1
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  torch-npu-test-aarch64-cann-a2-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A2
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  torch-npu-test-aarch64-cann-a3-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A3
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  *)
    echo "ERROR: Unknown image tag: ${BASE_TAG}"
    echo ""
    echo "Supported tags:"
    echo "  Builder: torch-npu-builder-<x86_64|aarch64>-torch<2.13.0|master>"
    echo "  Test:    torch-npu-test-<x86_64|aarch64>-cann-<a1|a2|a3>-py3.10-torch<2.13.0|master>"
    exit 1
    ;;
esac

# --- CANN version for the image tag (test images only) ---
# Test image tags always carry the CANN version:
#   public repo: pinned version from common/install_cann.sh (single source of truth)
#   OBS share:   version discovered from the share via obsutil
CANN_TAG_SUFFIX=""
SECRET_ARGS=()
SHARE_BUILD=0
if [[ -n "${OBS_SHARE_URL:-}" && "${IMAGE_TYPE}" != "test" ]]; then
  echo "NOTE: OBS_SHARE_URL is set but this is a ${IMAGE_TYPE} image (no CANN inside); ignoring share parameters."
fi
if [[ "${IMAGE_TYPE}" == "test" ]]; then
  if [[ -n "${OBS_SHARE_URL:-}" ]]; then
    SHARE_BUILD=1
    if [[ -z "${OBS_ACCESS_CODE:-}" ]]; then
      echo "ERROR: OBS_ACCESS_CODE is required when OBS_SHARE_URL is set."
      exit 1
    fi
    if ! command -v obsutil >/dev/null 2>&1; then
      echo "ERROR: obsutil not found on the build machine."
      echo "       Install it first: sudo bash .ci/docker/common/install_obs.sh"
      exit 1
    fi
    # shellcheck source=common/obs_share.sh
    source "${SCRIPT_DIR}/common/obs_share.sh"

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

    # Pass share URL / access code via a BuildKit secret so they never end up
    # in image layers or docker history.
    SECRET_FILE=$(mktemp /tmp/obs_share_secret.XXXXXX)
    chmod 600 "${SECRET_FILE}"
    printf 'OBS_SHARE_URL="%s"\nOBS_ACCESS_CODE="%s"\n' "${OBS_SHARE_URL}" "${OBS_ACCESS_CODE}" > "${SECRET_FILE}"
    trap 'rm -f "${AUTH_FILE}" "${SECRET_FILE}"' EXIT
    SECRET_ARGS=(--secret "id=obs_share,src=${SECRET_FILE}")
  else
    CANN_VERSION=$(sed -n 's/^CANN_VERSION_PUBLIC="\([^"]*\)"$/\1/p' "${SCRIPT_DIR}/common/install_cann.sh")
    if [[ -z "${CANN_VERSION}" ]]; then
      echo "ERROR: cannot read CANN_VERSION_PUBLIC from ${SCRIPT_DIR}/common/install_cann.sh"
      exit 1
    fi
  fi
  CANN_TAG_SUFFIX="-cann${CANN_VERSION}"
fi

TIMESTAMP="${TIMESTAMP:-$(TZ=Asia/Shanghai date +%Y%m%d%H%M)}"
COMMIT_ID="${COMMIT_ID:-$(git -C "${SCRIPT_DIR}" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)}"
TAG="${BASE_TAG}${CANN_TAG_SUFFIX}-${TIMESTAMP}-${COMMIT_ID}"

DOCKERFILE="${SCRIPT_DIR}/${VERSION_DIR}/${IMAGE_TYPE}/Dockerfile.${ARCH}"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "ERROR: Dockerfile not found: ${DOCKERFILE}"
  exit 1
fi

BUILD_ARGS=()
if [[ -n "${CANN_CHIP:-}" ]]; then
  BUILD_ARGS+=(--build-arg CANN_CHIP="${CANN_CHIP}")
fi
if [[ -n "${PYTHON_VERSION:-}" ]]; then
  BUILD_ARGS+=(--build-arg PYTHON_VERSION="${PYTHON_VERSION}")
fi
if [[ -n "${PYTORCH_VERSION:-}" ]]; then
  BUILD_ARGS+=(--build-arg PYTORCH_VERSION="${PYTORCH_VERSION}")
fi
if [[ "${SHARE_BUILD}" == "1" ]]; then
  BUILD_ARGS+=(--build-arg CANN_VERSION="${CANN_VERSION}")
fi

echo "=== Image Configuration ==="
echo "  Image Type:   ${IMAGE_TYPE}"
echo "  Architecture: ${ARCH}"
echo "  Version Dir:  ${VERSION_DIR}"
echo "  CANN Chip:    ${CANN_CHIP:--}"
if [[ "${IMAGE_TYPE}" == "test" ]]; then
  if [[ "${SHARE_BUILD}" == "1" ]]; then
    echo "  CANN Source:  OBS share ${OBS_SHARE_URL%%\?*}"
  else
    echo "  CANN Source:  public repo"
  fi
  echo "  CANN Version: ${CANN_VERSION}"
else
  echo "  CANN Source:  none (no CANN in builder images)"
fi
echo "  Python:       ${PYTHON_VERSION:--}"
echo "  PyTorch:      ${PYTORCH_VERSION:-nightly (built in CI)}"
echo "  Full Tag:     ${TAG}"
echo "  Dockerfile:   ${DOCKERFILE}"

if [[ -n "${TAG_OUT:-}" ]]; then
  printf '%s\n' "${TAG}" > "${TAG_OUT}"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "=== DRY_RUN=1: skipping docker build ==="
  exit 0
fi

echo "=== Building ${IMAGE_TYPE} image: ${TAG} ==="
# BuildKit is required for --secret support.
export DOCKER_BUILDKIT=1
docker build \
  "${BUILD_ARGS[@]}" \
  "${SECRET_ARGS[@]}" \
  --tag "${TAG}" \
  --file "${DOCKERFILE}" \
  "${BUILD_CONTEXT}"

echo "=== Image built successfully: ${TAG} ==="
