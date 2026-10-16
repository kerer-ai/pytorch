#!/usr/bin/bash
# Generate the image version report (default: /image_report.txt).
# Run as the LAST step of a Dockerfile so it captures the final image state.
#
# Usage:
#   ./gen_image_report.sh test <PYTHON_VERSION>   # test images (conda env + CANN + triton)
#   ./gen_image_report.sh builder                 # builder images (python runtimes + tools)
#
# Environment inputs (Dockerfile ARGs, or inline in the RUN step):
#   IMAGE_TAG      full image tag (docker_build.sh passes it via --build-arg)
#   BASE_IMAGE     base image of this Dockerfile (inline in the RUN step)
#   CANN_CHIP      CANN chip variant, e.g. A1 (test images)
#   CANN_VERSION   CANN version fallback (the stamp from install_cann.sh wins)
#   REPORT_PATH    output path override (default /image_report.txt)
#
# Every probe degrades to "-" instead of failing: a missing component must
# never break the image build.

set -u

MODE="${1:?Usage: $0 test <PYTHON_VERSION> | $0 builder}"
REPORT_PATH="${REPORT_PATH:-/image_report.txt}"

# --- helpers ---------------------------------------------------------------

# First line of a command's stdout, or "-" when the command is unavailable.
first_line() {
  local out
  out=$("$@" 2>/dev/null | head -1)
  printf '%s\n' "${out:--}"
}

# Version token of "<cmd> --version" output, or "-" when unavailable.
pyver_of() {
  local out
  out=$("$1" --version 2>/dev/null | head -1 | awk '{print $2}')
  printf '%s\n' "${out:--}"
}

os_pretty_name() {
  local name=""
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    name=$(. /etc/os-release && echo "${PRETTY_NAME:-}")
  fi
  printf '%s\n' "${name:--}"
}

# --- header ----------------------------------------------------------------

cat > "${REPORT_PATH}" <<EOF
===== torch-npu ${MODE} image report =====
Generated (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)
Image tag:       ${IMAGE_TAG:--}
Base image:      ${BASE_IMAGE:--}
Architecture:    $(uname -m)
OS:              $(os_pretty_name)
EOF

# --- test images -----------------------------------------------------------

gen_test_report() {
  local py_ver="${1:?Usage: $0 test <PYTHON_VERSION>}"
  local env_name="py_${py_ver}"

  # CANN: the stamp written by install_cann.sh is the installed truth;
  # the CANN_VERSION build-arg is only a fallback.
  local cann_ver=""
  cann_ver=$(cat /usr/local/Ascend/cann/.installed_version 2>/dev/null)
  cann_ver="${cann_ver:-${CANN_VERSION:-}}"

  local python_ver="" triton_ver=""
  python_ver=$(conda run -n "${env_name}" python --version 2>/dev/null | awk '{print $2}')
  triton_ver=$(conda run -n "${env_name}" pip show triton-ascend 2>/dev/null | awk '/^Version:/{print $2}')

  {
    echo ""
    echo "--- Core versions ---"
    echo "CANN chip:       ${CANN_CHIP:--}"
    echo "CANN version:    ${cann_ver:--}"
    echo "Python:          ${python_ver:--} (conda env ${env_name})"
    echo "triton-ascend:   ${triton_ver:--}"
    echo ""
    echo "--- Tools ---"
    echo "cmake:           $(first_line cmake --version)"
    echo "gcc:             $(first_line gcc --version)"
    echo "g++:             $(first_line g++ --version)"
    echo "obsutil:         $(first_line obsutil version)"
    echo ""
    echo "--- pip freeze (conda env ${env_name}) ---"
    conda run -n "${env_name}" pip freeze 2>/dev/null || echo "(unavailable)"
  } >> "${REPORT_PATH}"
}

# --- builder images --------------------------------------------------------

gen_builder_report() {
  local versions=(3.10 3.11 3.12 3.13 3.14)

  local cmake_ver
  cmake_ver=$(first_line cmake --version)
  [ "${cmake_ver}" = "-" ] && cmake_ver=$(first_line cmake3 --version)

  {
    echo ""
    echo "--- Python runtimes ---"
    local v
    for v in "${versions[@]}"; do
      printf '%-16s %s\n' "python${v}:" "$(pyver_of "python${v}")"
    done
    echo ""
    echo "--- Tools ---"
    echo "cmake:           ${cmake_ver}"
    echo "ninja:           $(first_line ninja --version)"
    echo "mold:            $(first_line mold --version)"
    echo "ccache:          $(first_line ccache --version)"
    echo "auditwheel:      $(first_line auditwheel --version)"
    echo "gcc:             $(first_line gcc --version)"
    echo "obsutil:         $(first_line obsutil version)"
    for v in "${versions[@]}"; do
      echo ""
      echo "--- pip freeze (python${v}) ---"
      "pip${v}" freeze 2>/dev/null || echo "(unavailable)"
    done
  } >> "${REPORT_PATH}"
}

case "${MODE}" in
  test)
    gen_test_report "${2:?Usage: $0 test <PYTHON_VERSION>}"
    ;;
  builder)
    gen_builder_report
    ;;
  *)
    echo "ERROR: unknown mode '${MODE}' (expected 'test' or 'builder')" >&2
    exit 1
    ;;
esac

echo "Image report written to ${REPORT_PATH}"
