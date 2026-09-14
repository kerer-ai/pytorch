#!/usr/bin/bash
# Shared helpers for downloading CANN packages from a Huawei OBS share link
# (the e-share link whose v2token works as an obsutil authorization code).
#
# Sourced by install_cann.sh (inside Docker) and docker_build.sh (on the
# build runner). Requires obsutil to be installed and on PATH.

# Print the v2token extracted from an e-share URL.
# The token is everything after "v2token=" up to the next "&" (if any).
obs_share_extract_token() {
  local token="${1#*v2token=}"
  printf '%s' "${token%%&*}"
}

# Write a share token into a temp authorization-code file and print its path.
# The caller is responsible for cleaning it up.
obs_share_write_auth_file() {
  local auth_file
  auth_file=$(mktemp /tmp/obs_share_auth.XXXXXX) || return 1
  chmod 600 "${auth_file}" || { rm -f "${auth_file}"; return 1; }
  printf '%s' "$1" > "${auth_file}" || { rm -f "${auth_file}"; return 1; }
  printf '%s' "${auth_file}"
}

# Query a share and set globals OBS_SHARE_PREFIX and OBS_SHARE_VERSION:
#   OBS_SHARE_PREFIX  authorized object-key prefix, e.g.
#                     "version_combo_snapshot/CANN 9.2.0-20260910200430/"
#   OBS_SHARE_VERSION CANN version derived from the prefix, e.g.
#                     "9.2.0-20260910200430"
# Usage: obs_share_discover <auth_file> <access_code>
obs_share_discover() {
  local auth_file="$1" access_code="$2" out prefix version
  out=$(obsutil share-ls "file://${auth_file}" -ac="${access_code}" -limit=1 2>&1) || {
    echo "ERROR: obsutil share-ls failed (invalid share URL or access code?):" >&2
    printf '%s\n' "${out}" >&2
    return 1
  }
  prefix=$(printf '%s\n' "${out}" | sed -n 's/^The authorized prefix is \[\(.*\)\][[:space:]]*$/\1/p' | head -1)
  if [[ -z "${prefix}" ]]; then
    echo "ERROR: cannot parse 'The authorized prefix is [...]' from obsutil output:" >&2
    printf '%s\n' "${out}" >&2
    return 1
  fi
  version="${prefix%/}"
  version="${version##*/}"
  version="${version#CANN }"
  if [[ -z "${version}" ]]; then
    echo "ERROR: cannot derive CANN version from share prefix [${prefix}]" >&2
    return 1
  fi
  OBS_SHARE_PREFIX="${prefix}"
  OBS_SHARE_VERSION="${version}"
}

# List .run package objects for the given architecture under the share and
# print their obs:// URLs, one per line.
# Usage: obs_share_list_run_objects <auth_file> <access_code> <prefix> <arch>
#   <arch>: uname -m style, e.g. x86_64 or aarch64
obs_share_list_run_objects() {
  local auth_file="$1" access_code="$2" prefix="$3" arch="$4" out
  out=$(obsutil share-ls "file://${auth_file}" -ac="${access_code}" \
    -prefix="${prefix}run/${arch}-linux/" -s 2>&1) || {
    echo "ERROR: obsutil share-ls failed for prefix [${prefix}run/${arch}-linux/]:" >&2
    printf '%s\n' "${out}" >&2
    return 1
  }
  # Brief mode prints one path per line; folders end with "/".
  printf '%s\n' "${out}" | grep '^obs://' | grep -v '/$'
}

# Download one package whose basename matches <pattern> (shell glob) from the
# share into the current directory. Prints the local filename on success.
# Usage: obs_share_download_pkg <auth_file> <access_code> <objects> <pattern>
#   <objects>: output of obs_share_list_run_objects
obs_share_download_pkg() {
  local auth_file="$1" access_code="$2" objects="$3" pattern="$4" obj key base
  while IFS= read -r obj; do
    [[ -z "${obj}" ]] && continue
    base="${obj##*/}"
    case "${base}" in
      ${pattern})
        key="${obj#obs://*/}"
        echo "Downloading ${base} ..." >&2
        obsutil share-cp "file://${auth_file}" "./${base}" -key="${key}" -ac="${access_code}" || return 1
        printf '%s' "${base}"
        return 0
        ;;
    esac
  done <<< "${objects}"
  echo "ERROR: no package matching [${pattern}] in the share." >&2
  echo "Available packages:" >&2
  printf '%s\n' "${objects}" | sed 's|.*/||;s|^|  |' >&2
  return 1
}
