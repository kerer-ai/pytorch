#!/usr/bin/bash
# Install Huawei OBS util for object storage access.
# Downloads obsutil into /usr/local/obsutil and symlinks it into /usr/local/bin.
# The upstream tarball stores root-only modes (dirs dr-x------, binary
# r-x------), so permissions are normalized to make obsutil usable by
# non-root users (e.g. CI runners where only this script runs under sudo).
# Self-verifies by running the binary via its resolved path (PATH-independent).

set -e

ARCH=$(uname -m)
case "${ARCH}" in
  x86_64)  OBS_ARCH="amd64" ;;
  aarch64) OBS_ARCH="arm64" ;;
  *)       echo "Unsupported architecture: ${ARCH}"; exit 1 ;;
esac

OBS_URL="https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_${OBS_ARCH}.tar.gz"

WORKDIR=$(mktemp -d)
trap 'rm -rf "${WORKDIR}"' EXIT
wget -q -O "${WORKDIR}/obsutil.tar.gz" "${OBS_URL}"
mkdir -p /usr/local/obsutil
tar -zxf "${WORKDIR}/obsutil.tar.gz" -C /usr/local/obsutil/

# Locate the obsutil binary regardless of the archive's inner layout
# (versioned dir like obsutil_linux_amd64_5.8.3/ or a flat one).
OBS_BIN=$(find /usr/local/obsutil -maxdepth 2 -type f -name obsutil | sort -V | tail -1)
if [ -z "${OBS_BIN}" ]; then
  echo "ERROR: obsutil binary not found after extracting ${OBS_URL}; archive layout:"
  ls -laR /usr/local/obsutil >&2
  exit 1
fi

# Normalize the root-only archive modes: make the install dirs traversable
# and the binary executable/readable by all users.
chmod 755 /usr/local/obsutil "$(dirname "${OBS_BIN}")"
chmod 755 "${OBS_BIN}"
ln -sf "${OBS_BIN}" /usr/local/bin/obsutil

# Self-verify via the resolved path so failures surface here, not later.
"${OBS_BIN}" version

echo "OBS util installed: ${OBS_BIN} -> /usr/local/bin/obsutil"
