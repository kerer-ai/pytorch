#!/usr/bin/bash
# Install triton-ascend for NPU.
# Usage: ./install_triton.sh <PYTHON_VERSION>
#   PYTHON_VERSION: e.g. 3.10, 3.11, 3.12, 3.13
#
# Wheels come from a fixed snapshot on Huawei Cloud OBS
# (ascend-triton-open bucket, per-build timestamped directory).

set -e

TRITON_VERSION="${TRITON_VERSION:-3.6.0}"
PYTHON_VERSION="${1:?Usage: $0 <PYTHON_VERSION> (e.g. 3.10)}"

ARCH=$(uname -m)
PY_SHORT=$(echo "${PYTHON_VERSION}" | tr -d '.')

TRITON_WHL="triton_ascend-${TRITON_VERSION}-cp${PY_SHORT}-cp${PY_SHORT}-manylinux_2_27_${ARCH}.manylinux_2_28_${ARCH}.whl"
# 20261015213503 is the OBS build snapshot directory for triton-ascend 3.6.0.
TRITON_URL="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261015213503/${TRITON_WHL}"

echo "Installing triton-ascend ${TRITON_VERSION} for Python ${PYTHON_VERSION} (${ARCH})..."
pip3 install --no-cache-dir "${TRITON_URL}"
echo "triton-ascend installed."
