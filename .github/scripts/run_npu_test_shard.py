#!/usr/bin/env python3
"""
Run a shard of patched upstream PyTorch tests via direct pytest execution.

Test types:
- Distributed tests: NPU distributed tests (run with per-file isolation for crash safety)
- Regular tests: All other tests (run with parallel workers for speed)

Each shard applies whitelist/blacklist filtering from case_paths_ci.yml
and item-level deselection from disabled_testcases.json.

Tests are executed using direct pytest with pytest-xdist for file-level
parallel execution (--dist=loadfile, -n=parallel). Files in the
SERIAL_TEST_FILES set are run one-at-a-time to avoid resource contention.
"""

import argparse
import fnmatch
import json
import os
import re
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Dict, List, Tuple
import threading


# Test files that should never be run in parallel with other test files.
# Borrowed from PyTorch's run_test.py RUN_PARALLEL_BLOCKLIST + CI_SERIAL_LIST.
# These tests are run one-at-a-time via individual pytest invocations (no -n flag).
SERIAL_TEST_FILES = {
    # From RUN_PARALLEL_BLOCKLIST: tests inside these files should never run in parallel
    "test_extension_utils",
    "test_cpp_extensions_jit",
    "test_cpp_extensions_open_device_registration",
    "test_cpp_extensions_stream_and_event",
    "test_cpp_extensions_mtia_backend",
    "test_jit_disabled",
    "test_mobile_optimizer",
    "test_multiprocessing",
    "test_multiprocessing_spawn",
    "test_namedtuple_return_api",
    "test_overrides",
    "test_show_pickle",
    "test_tensorexpr",
    "test_cuda_primary_ctx",
    "test_cuda_trace",
    "test_cuda_nvml_based_avail",
    "test_autograd_fallback",
    # From FSDP_TEST: dynamically generated in run_test.py as
    # [test for test in TESTS if test.startswith("distributed/fsdp")]
    "distributed/fsdp/test_checkpoint_wrapper",
    "distributed/fsdp/test_distributed_checkpoint",
    "distributed/fsdp/test_fsdp_apply",
    "distributed/fsdp/test_fsdp_backward_prefetch",
    "distributed/fsdp/test_fsdp_checkpoint",
    "distributed/fsdp/test_fsdp_clip_grad_norm",
    "distributed/fsdp/test_fsdp_comm",
    "distributed/fsdp/test_fsdp_comm_hooks",
    "distributed/fsdp/test_fsdp_core",
    "distributed/fsdp/test_fsdp_dtensor_state_dict",
    "distributed/fsdp/test_fsdp_exec_order",
    "distributed/fsdp/test_fsdp_fine_tune",
    "distributed/fsdp/test_fsdp_flatten_params",
    "distributed/fsdp/test_fsdp_freezing_weights",
    "distributed/fsdp/test_fsdp_fx",
    "distributed/fsdp/test_fsdp_grad_acc",
    "distributed/fsdp/test_fsdp_hybrid_shard",
    "distributed/fsdp/test_fsdp_ignored_modules",
    "distributed/fsdp/test_fsdp_input",
    "distributed/fsdp/test_fsdp_memory",
    "distributed/fsdp/test_fsdp_meta",
    "distributed/fsdp/test_fsdp_misc",
    "distributed/fsdp/test_fsdp_mixed_precision",
    "distributed/fsdp/test_fsdp_multiple_forward",
    "distributed/fsdp/test_fsdp_multiple_wrapping",
    "distributed/fsdp/test_fsdp_optim_state",
    "distributed/fsdp/test_fsdp_overlap",
    "distributed/fsdp/test_fsdp_pure_fp16",
    "distributed/fsdp/test_fsdp_sharded_grad_scaler",
    "distributed/fsdp/test_fsdp_state_dict",
    "distributed/fsdp/test_fsdp_tp_integration",
    "distributed/fsdp/test_fsdp_traversal",
    "distributed/fsdp/test_fsdp_uneven",
    "distributed/fsdp/test_fsdp_unshard_params",
    "distributed/fsdp/test_fsdp_use_orig_params",
    "distributed/fsdp/test_hsdp_dtensor_state_dict",
    "distributed/fsdp/test_shard_utils",
    "distributed/fsdp/test_utils",
    "distributed/fsdp/test_wrap",
    # From CI_SERIAL_LIST: file-level serial (often due to OOM or resource contention)
    "test_nn",
    "test_fake_tensor",
    "test_cpp_api_parity",
    "test_reductions",
    "test_fx_backends",
    "test_torch",
    "test_tensor_creation_ops",
    "test_dispatch",
    "test_python_dispatch",
    "test_spectral_ops",
    "nn/test_pooling",
    "nn/test_convolution",
    "distributions/test_distributions",
    "test_fx",
    "test_utils",
    "test_sort_and_select",
    "test_backward_compatible_arguments",
    "test_autocast",
    "test_native_mha",
    "test_module_hooks",
}

# Excluded shard special tests: directories and files excluded by discover_tests.py
# (blocklisted_patterns and blocklisted_tests), but should be tested on NPU.
# Excluded shard dynamically scans these directories/files, then applies
# whitelist/blacklist from case_paths_ci.yml before execution.
EXCLUDED_TESTS_PATTERNS = [
    # blocklisted_patterns from discover_tests.py (directories)
    "test/custom_backend",      # all test files in this directory
    "test/custom_operator",     # all test files in this directory
    "test/fx",                  # all test files in this directory
    "test/mobile",              # all test files in this directory
    "test/quantization",        # all test files in this directory
    # blocklisted_tests from discover_tests.py (individual files)
    "test/test_bundled_images.py",
    "test/test_cpp_extensions_aot.py",
    "test/test_determination.py",
    "test/test_jit_string.py",
    "test/test_kernel_launch_checks.py",
    "test/test_nnapi.py",
    "test/test_static_runtime.py",
    "test/test_throughput_benchmark.py",
]


def path_matches_excluded_pattern(path: str) -> bool:
    """
    Check if a test file path matches any excluded test pattern.

    Args:
        path: Test file path with 'test/' prefix (e.g., 'test/fx/test_common_passes.py')

    Returns:
        True if path matches any pattern in EXCLUDED_TESTS_PATTERNS
    """
    for pattern in EXCLUDED_TESTS_PATTERNS:
        if pattern.endswith('.py'):
            # Exact file match
            if path == pattern:
                return True
        else:
            # Directory match: path starts with pattern + '/'
            if path.startswith(pattern + '/'):
                return True
    return False


def get_shard_type_prefix(shard_type: str) -> str:
    """
    Convert shard type to short prefix for file naming.

    Args:
        shard_type: "distributed" or "regular"

    Returns:
        "dist" for distributed, "reg" for regular
    """
    return "dist" if shard_type == "distributed" else "reg"


def parse_args():
    parser = argparse.ArgumentParser(description="Run PyTorch NPU tests for a shard via direct pytest")
    parser.add_argument("--shard", type=int, required=True, help="Shard number (1-indexed within test-type)")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards for this test-type")
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["distributed", "regular"],
        default="regular",
        help="Test type: 'distributed' for distributed tests, 'regular' for other tests",
    )
    parser.add_argument("--test-dir", type=str, required=True, help="Path to the PyTorch test directory")
    parser.add_argument("--disabled-testcases", type=str, help="Path to disabled_testcases.json")
    parser.add_argument(
        "--case-paths-config",
        type=str,
        help="Path to case_paths_ci.yml for file-level whitelist/blacklist control",
    )
    parser.add_argument(
        "--crashed-files-config",
        type=str,
        help="Path to CRASHED.yml for crashed test files blacklist",
    )
    parser.add_argument("--report-dir", type=str, default="test-reports", help="Directory for test reports")
    parser.add_argument("--timeout", type=int, default=600, help="Per-test timeout passed to pytest")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", type=int, default=2, help="Number of parallel workers (NUM_PARALLEL_PROCS)")
    parser.add_argument(
        "--use-tests-list",
        action="store_true",
        default=True,
        help="Use TESTS list from discover_tests.py (same as run_test.py --help). Default: True",
    )
    parser.add_argument(
        "--use-raw-discovery",
        action="store_true",
        default=False,
        help="Use raw file discovery (scan all test_*.py) instead of TESTS list. Default: False",
    )
    parser.add_argument(
        "--per-file-isolation",
        action="store_true",
        default=False,
        help=(
            "Run each test file in its own pytest subprocess. "
            "This prevents NPU kernel crashes from affecting other test files. "
            "If a file crashes, it won't generate a JUnit XML report, and the summary "
            "script will show 'Missing' for that file. Slower but more resilient. "
            "Default: False"
        ),
    )
    return parser.parse_args()


def normalize_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def normalize_rule_path(rule: str) -> str:
    normalized = normalize_path(rule)
    if not normalized:
        return ""
    if normalized == "test" or normalized.startswith("test/"):
        return normalized.rstrip("/")
    return f"test/{normalized}".rstrip("/")


def load_disabled_testcases_count(json_file: str) -> int:
    if not json_file or not os.path.exists(json_file):
        return 0

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, (dict, list)):
        return len(data)
    return 0


def parse_simple_yaml_lists(raw_text: str) -> Dict[str, List[str]]:
    parsed = {"whitelist": [], "blacklist": []}
    current_key = None

    for raw_line in raw_text.splitlines():
        without_comment = raw_line.split("#", 1)[0].rstrip()
        if not without_comment.strip():
            continue

        stripped = without_comment.lstrip()
        if not raw_line.startswith((" ", "\t")) and stripped.endswith(":"):
            key = stripped[:-1].strip()
            current_key = key if key in parsed else None
            continue

        if current_key and stripped.startswith("- "):
            value = stripped[2:].strip().strip("\"'")
            if value:
                parsed[current_key].append(value)

    return parsed


def coerce_rule_list(value, key: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected '{key}' to be a list, got {type(value).__name__}")

    normalized_values = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Expected every '{key}' entry to be a string, got {type(item).__name__}")
        normalized = normalize_rule_path(item)
        if normalized:
            normalized_values.append(normalized)
    return normalized_values


def load_case_path_rules(config_file: str) -> Tuple[str, List[str], List[str]]:
    if not config_file:
        return "", [], []

    config_path = Path(config_file).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"case_paths_ci config not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        payload = parse_simple_yaml_lists(raw_text)
    else:
        payload = yaml.safe_load(raw_text) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML object in {config_path}, got {type(payload).__name__}")

    whitelist = coerce_rule_list(payload.get("whitelist"), "whitelist")
    blacklist = coerce_rule_list(payload.get("blacklist"), "blacklist")
    return str(config_path), whitelist, blacklist


def load_crashed_files_list(config_file: str) -> Tuple[str, List[str]]:
    """
    Load crashed test files blacklist from CRASHED.yml.

    This file contains test files that cause segmentation fault or process crash
    on NPU, which would break CI execution. These files are excluded before
    the whitelist/blacklist rules are applied.

    Args:
        config_file: Path to CRASHED.yml file

    Returns:
        Tuple of (config_file_path, list_of_crashed_files)
    """
    if not config_file:
        return "", []

    config_path = Path(config_file).resolve()
    if not config_path.exists():
        # File doesn't exist, return empty list (not an error)
        return "", []

    raw_text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        # Parse manually - CRASHED.yml has simpler format
        crashed_files = []
        for line in raw_text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("- "):
                value = stripped[2:].strip()
                if value:
                    crashed_files.append(value)
    else:
        payload = yaml.safe_load(raw_text) or {}
        if not isinstance(payload, dict):
            return str(config_path), []
        crashed_files = payload.get("crashed_files", [])
        if not isinstance(crashed_files, list):
            crashed_files = []

    # Normalize the paths
    normalized_files = coerce_rule_list(crashed_files, "crashed_files")
    return str(config_path), normalized_files


def discover_raw_test_files(test_dir: Path) -> List[str]:
    """Fallback: scan all test_*.py files in test directory."""
    files = []
    for test_file in test_dir.rglob("test_*.py"):
        rel_path = test_file.relative_to(test_dir).as_posix()
        files.append(f"test/{rel_path}")
    return sorted(files)


def get_tests_list_from_discover_tests(test_dir: Path) -> List[str]:
    """
    Get TESTS list from tools/testing/discover_tests.py (same as run_test.py --help).

    This provides the official test list that run_test.py recognizes, which includes:
    - Blocklisted patterns (e.g., "ao", "custom_backend", "fx", "jit")
    - Blocklisted tests (specific test files to exclude)
    - Extra tests (additional tests not discovered by file scanning)

    Returns:
        List of test paths with 'test/' prefix and '.py' suffix (e.g., 'test/test_autograd.py')
    """
    repo_root = test_dir.parent
    discover_tests_path = repo_root / "tools" / "testing" / "discover_tests.py"

    if not discover_tests_path.exists():
        print(f"Warning: discover_tests.py not found at {discover_tests_path}, falling back to raw file scan")
        return discover_raw_test_files(test_dir)

    # Import TESTS list from discover_tests.py
    # We need to temporarily add repo_root to sys.path
    original_path = sys.path.copy()
    sys.path.insert(0, str(repo_root))

    try:
        from tools.testing.discover_tests import TESTS

        # TESTS list contains test names without 'test/' prefix and without '.py' suffix
        # e.g., 'test_autograd', 'distributed/test_c10d'
        # We need to convert to full paths with 'test/' prefix and '.py' suffix
        tests_with_prefix = []
        for test in TESTS:
            if test.startswith("cpp/"):
                # C++ tests - skip for now as we focus on Python tests
                continue
            # Add 'test/' prefix if not already present
            if not test.startswith("test/"):
                test_path = f"test/{test}"
            else:
                test_path = test
            # Add '.py' suffix for all Python test files (they all end with test_*)
            # Check if it's a directory-like entry (no test_ prefix in the last component)
            last_component = test.split("/")[-1]
            if not test_path.endswith(".py") and last_component.startswith("test_"):
                test_path = f"{test_path}.py"
            tests_with_prefix.append(test_path)

        return sorted(tests_with_prefix)
    except ImportError as e:
        print(f"Warning: Failed to import TESTS from discover_tests.py: {e}, falling back to raw file scan")
        return discover_raw_test_files(test_dir)
    finally:
        sys.path = original_path


def path_matches_rule(test_path: str, rule: str) -> bool:
    normalized_path = normalize_path(test_path)
    normalized_rule = normalize_rule_path(rule)
    if not normalized_rule:
        return False

    if any(char in normalized_rule for char in "*?[]"):
        return fnmatch.fnmatch(normalized_path, normalized_rule)

    return normalized_path == normalized_rule or normalized_path.startswith(f"{normalized_rule}/")


def apply_case_path_rules(
    test_files: List[str], whitelist: List[str], blacklist: List[str]
) -> Tuple[List[str], List[str]]:
    if whitelist:
        selected = [path for path in test_files if any(path_matches_rule(path, rule) for rule in whitelist)]
    else:
        selected = list(test_files)

    if blacklist:
        selected = [path for path in selected if not any(path_matches_rule(path, rule) for rule in blacklist)]

    selected_set = set(selected)
    excluded = [path for path in test_files if path not in selected_set]
    return selected, excluded


def select_shard_files(test_files: List[str], shard: int, num_shards: int) -> List[str]:
    """
    Select test files for a shard using contiguous range-based selection.

    This approach assigns consecutive files to each shard (sorted alphabetically),
    ensuring that files from the same directory end up in the same or adjacent shards.
    This is different from round-robin (modulo) distribution which spreads adjacent
    files across different shards.

    Args:
        test_files: List of test file paths, already sorted alphabetically
        shard: Shard number (1-indexed, 1 <= shard <= num_shards)
        num_shards: Total number of shards (max 100)

    Returns:
        List of test files assigned to this shard
    """
    if not test_files:
        return []

    shard_index = shard - 1  # Convert to 0-indexed
    total_files = len(test_files)

    # Calculate base size and remainder for even distribution
    base_size = total_files // num_shards
    remainder = total_files % num_shards

    # Shards with index < remainder get one extra file
    # This ensures files are distributed as evenly as possible
    if shard_index < remainder:
        start = shard_index * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (shard_index - remainder) * base_size
        end = start + base_size

    return test_files[start:end]


def filter_tests_by_type(test_files: List[str], shard_type: str) -> Tuple[List[str], List[str]]:
    """
    Filter test files by shard type.

    Args:
        test_files: List of test file paths (with test/ prefix)
        shard_type: "distributed", "excluded", or "regular"

    Returns:
        Tuple of (selected_files, excluded_files)
    """
    if shard_type == "distributed":
        # Distributed tests: files starting with test/distributed/
        selected = [f for f in test_files if f.startswith("test/distributed/")]
        excluded = [f for f in test_files if not f.startswith("test/distributed/")]
    elif shard_type == "excluded":
        # Excluded shard: dynamically match files from EXCLUDED_TESTS_PATTERNS
        # These are directories/files excluded by discover_tests.py but should run on NPU
        selected = [f for f in test_files if path_matches_excluded_pattern(f)]
        excluded = [f for f in test_files if not path_matches_excluded_pattern(f)]
    else:
        # Regular tests: exclude distributed and excluded pattern files
        selected = [
            f for f in test_files
            if not f.startswith("test/distributed/") and not path_matches_excluded_pattern(f)
        ]
        excluded = [
            f for f in test_files
            if f.startswith("test/distributed/") or path_matches_excluded_pattern(f)
        ]

    return selected, excluded


def strip_test_prefix_and_suffix(test_path: str) -> str:
    """
    Remove 'test/' prefix and '.py' suffix from path.

    Example: 'test/test_autograd.py' -> 'test_autograd'
             'test/distributed/test_c10d.py' -> 'distributed/test_c10d'
    """
    path = test_path
    if path.startswith("test/"):
        path = path[5:]  # Remove 'test/' prefix
    if path.endswith(".py"):
        path = path[:-3]  # Remove '.py' suffix
    return path


def is_serial_test(test_path: str) -> bool:
    """
    Check if a test file should be run serially (not in parallel with other files).

    Tests in SERIAL_TEST_FILES are run one-at-a-time via individual pytest
    invocations without the -n flag, to avoid resource contention and OOM.
    """
    name = strip_test_prefix_and_suffix(test_path)
    return name in SERIAL_TEST_FILES


def parse_junit_xml(xml_file: str) -> Dict:
    stats = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
    }

    if not os.path.exists(xml_file):
        return stats

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for testsuite in root.iter("testsuite"):
            stats["total"] += int(testsuite.get("tests", 0))
            stats["failed"] += int(testsuite.get("failures", 0))
            stats["skipped"] += int(testsuite.get("skipped", 0))
            stats["errors"] += int(testsuite.get("errors", 0))
            stats["duration"] += float(testsuite.get("time", 0))
        stats["passed"] = stats["total"] - stats["failed"] - stats["skipped"] - stats["errors"]
    except Exception as exc:
        print(f"Warning: Failed to parse XML report {xml_file}: {exc}")

    return stats


def aggregate_junit_stats(report_roots: List[Path]) -> Dict:
    totals = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
    }

    seen_files = set()
    for report_root in report_roots:
        if not report_root.exists():
            continue
        for xml_file in report_root.rglob("*.xml"):
            try:
                resolved = str(xml_file.resolve())
            except OSError:
                resolved = str(xml_file)
            if resolved in seen_files:
                continue
            seen_files.add(resolved)

            stats = parse_junit_xml(str(xml_file))
            for key in totals:
                totals[key] += stats[key]
    return totals


def create_empty_stats() -> Dict:
    return {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
        "junit_generated": False,
        "junit_xml_files": 0,
        "zero_item_test_files": 0,
        "startup_failures": 0,
        "import_failures": 0,
        "test_failures": 0,
    }


def create_shard_info(shard: int, num_shards: int, timestamp: str) -> Dict:
    return {
        "shard": shard,
        "num_shards": num_shards,
        "selection_mode": "pytest_direct",
        "total_files": 0,
        "selected_test_entries": 0,
        "selected_test_files": 0,
        "shard_files": 0,
        "path_filtered_out_files": 0,
        "excluded_test_files": 0,
        "disabled_count": 0,
        "disabled_count_matched": 0,
        "disabled_count_deselected": 0,
        "whitelist_entries": 0,
        "blacklist_entries": 0,
        "junit_generated": False,
        "junit_xml_files": 0,
        "zero_item_test_files": 0,
        "startup_failures": 0,
        "import_failures": 0,
        "test_failures": 0,
        "timestamp": timestamp,
    }


def finalize_stats(base_stats: Dict, returncode: int, duration: float, error_message: str = "") -> Dict:
    stats = dict(base_stats)
    stats["duration"] = max(float(stats.get("duration", 0.0)), duration)
    if returncode != 0:
        stats["returncode"] = returncode
        if returncode < 0:
            signal_num = abs(returncode)
            try:
                signal_name = signal.Signals(signal_num).name
            except ValueError:
                signal_name = f"SIG{signal_num}"
            stats["crashed"] = True
            stats["crash_signal"] = signal_name
        if stats.get("total", 0) == 0:
            stats["errors"] = max(stats.get("errors", 0), 1)
            stats["incomplete"] = True
        if error_message:
            stats["error_message"] = error_message
    else:
        stats["returncode"] = 0
    return stats


def save_stats_file(report_dir: str, shard: int, stats: Dict, shard_type: str = "regular") -> str:
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    stats_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats_file


def save_info_file(report_dir: str, shard: int, info: Dict, shard_type: str = "regular") -> str:
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    info_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_info.json")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info_file


def save_test_plan_file(report_dir: str, shard: int, planned_tests: List[str], shard_type: str = "regular") -> str:
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    plan_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_planned_test_files.txt")
    with open(plan_file, "w", encoding="utf-8") as f:
        for target in planned_tests:
            f.write(f"{target}\n")
    return plan_file


def save_excluded_test_files_file(report_dir: str, shard: int, test_targets: List[str], shard_type: str = "regular") -> str:
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    excluded_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_excluded_test_files.txt")
    with open(excluded_file, "w", encoding="utf-8") as f:
        for target in test_targets:
            f.write(f"{target}\n")
    return excluded_file


def save_unhandled_upstream_tests_file(report_dir: str, shard: int, test_targets: List[str], shard_type: str = "regular") -> str:
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    unhandled_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_unhandled_upstream_tests.txt")
    with open(unhandled_file, "w", encoding="utf-8") as f:
        for target in test_targets:
            f.write(f"{target}\n")
    return unhandled_file


def get_disabled_testcases_report_file(report_dir: str, shard: int, shard_type: str = "regular") -> str:
    prefix = get_shard_type_prefix(shard_type)
    return os.path.join(report_dir, f"shard_{prefix}-{shard}_disabled_testcases.json")


def load_disabled_testcases_report(report_dir: str, shard: int, shard_type: str = "regular") -> Dict:
    report_file = get_disabled_testcases_report_file(report_dir, shard, shard_type)
    if not os.path.exists(report_file):
        return {
            "disabled_count_matched": 0,
            "disabled_count_deselected": 0,
        }

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "disabled_count_matched": data.get("disabled_count_matched", 0),
            "disabled_count_deselected": data.get("disabled_count_deselected", 0),
        }
    except Exception as exc:
        print(f"Warning: Failed to read disabled testcase report: {exc}")
        return {
            "disabled_count_matched": 0,
            "disabled_count_deselected": 0,
        }


def print_stats_summary(shard: int, stats: Dict, shard_type: str = "regular") -> None:
    prefix = get_shard_type_prefix(shard_type)
    print(f"\n{'=' * 60}")
    print(f"Test Results for Shard {prefix}-{shard}")
    print(f"{'=' * 60}")
    print(f"Total:  {stats['total']}")
    print(f"Passed: {stats['passed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"Duration: {stats['duration']:.2f}s")
    print(f"{'=' * 60}")


def load_installed_torch_root() -> str:
    try:
        import torch
    except Exception as exc:
        print(f"Warning: Failed to import installed torch while preparing PYTHONPATH: {exc}")
        return ""

    return str(Path(torch.__file__).resolve().parent.parent)


def build_execution_env(
    test_dir: Path,
    script_dir: Path,
    disabled_testcases_file: str,
    report_dir: str,
    shard: int,
    shard_type: str = "regular",
) -> Dict[str, str]:
    repo_root = test_dir.parent
    pythonpath_parts = [str(script_dir)]

    torch_path = load_installed_torch_root()
    if torch_path:
        pythonpath_parts.append(torch_path)

    pythonpath_parts.extend([str(repo_root), str(test_dir)])

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    updates = {
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "PYTORCH_TEST_NPU": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
        "NO_TD": "1",
        "PYTEST_ADDOPTS": os.environ.get("PYTEST_ADDOPTS", ""),
        "PYTHONUNBUFFERED": "1",
        # Enable CI mode for slow/disabled test import behavior
        "CI": "true",
    }

    if disabled_testcases_file:
        updates["NPU_DISABLED_TESTCASES_JSON"] = os.path.abspath(disabled_testcases_file)
        updates["NPU_DISABLED_TESTCASES_REPORT"] = os.path.abspath(
            get_disabled_testcases_report_file(report_dir, shard, shard_type)
        )

    return updates


def clean_existing_junit_xml(report_dir: Path) -> None:
    if not report_dir.exists():
        return
    for xml_file in report_dir.rglob("*.xml"):
        xml_file.unlink(missing_ok=True)


def remove_existing_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def get_shard_log_file(report_dir: Path, shard: int, shard_type: str = "regular") -> Path:
    prefix = get_shard_type_prefix(shard_type)
    return report_dir / f"test_shard_{prefix}-{shard}.log"


def run_command_with_tee(command: List[str], cwd: Path, env: Dict[str, str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    merged_env.update(env)

    with log_file.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_handle.write(line)
            return process.wait()
        except BaseException:
            process.kill()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()


def analyze_pytest_log(log_file: Path, returncode: int) -> Dict:
    metrics = {
        "zero_item_test_files": 0,
        "startup_failures": 0,
        "import_failures": 0,
        "test_failures": 0,
    }

    if not log_file.exists():
        return metrics

    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return metrics

    if returncode == 5 or "collected 0 items" in content or "no tests ran" in content:
        metrics["zero_item_test_files"] = 1

    metrics["import_failures"] = len(
        re.findall(r"^ImportError while importing test module", content, flags=re.MULTILINE)
    )
    collection_errors = len(re.findall(r"^ERROR collecting ", content, flags=re.MULTILINE))
    metrics["startup_failures"] = max(collection_errors - metrics["import_failures"], 0)
    return metrics


def build_pytest_command(
    planned_tests: List[str],
    report_dir: Path,
    shard: int,
    timeout: int,
    verbose: bool,
    parallel: int,
    shard_type: str,
    xml_suffix: str = "",
) -> List[str]:
    """
    Build pytest command for direct test execution.

    All shard types (distributed, excluded, regular) use this unified command
    builder. Tests can run with pytest-xdist for file-level parallelism
    (when parallel > 0) or sequentially (when parallel == 0, for serial tests).

    Args:
        planned_tests: List of test file paths (with 'test/' prefix)
        report_dir: Directory for test reports
        shard: Shard number
        timeout: Per-test timeout
        verbose: Verbose output flag
        parallel: Number of parallel workers (0 = sequential, >0 = pytest-xdist)
        shard_type: "distributed", "excluded", or "regular"
        xml_suffix: Optional suffix appended to the XML filename
                    (e.g. "_test_nn" for serial per-file runs)

    Returns:
        Command list for subprocess execution
    """
    prefix = get_shard_type_prefix(shard_type)
    xml_file = report_dir / f"shard_{prefix}-{shard}_pytest{xml_suffix}.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--color=no",
        "-ra",
        "--tb=short",
        "--continue-on-collection-errors",
        f"--junitxml={xml_file}",
        "-p",
        "pytest_disabled_testcases_plugin",
    ]

    if parallel > 0:
        command.extend([
            f"-n={parallel}",       # pytest-xdist for parallel execution
            "--dist=loadfile",      # Distribute by file for locality
        ])

    if timeout > 0:
        command.append(f"--timeout={timeout}")

    if verbose:
        command.append("-vv")
    else:
        command.append("-v")

    # Add test files (strip only 'test/' prefix, keep '.py' suffix for pytest)
    # pytest needs actual file paths with .py extension
    for test in planned_tests:
        if test.startswith("test/"):
            test_stripped = test[5:]  # Remove only 'test/' prefix, keep '.py'
        else:
            test_stripped = test
        command.append(test_stripped)

    return command


def run_tests_via_pytest(
    planned_tests: List[str],
    shard: int,
    test_dir: Path,
    report_dir: Path,
    env_updates: Dict[str, str],
    timeout: int,
    verbose: bool,
    parallel: int,
    shard_type: str,
    per_file_isolation: bool = False,
) -> Tuple[int, Dict, Dict, List[str]]:
    """
    Run tests directly via pytest with serial/parallel split or per-file isolation.

    Execution modes:

    1. Per-file isolation mode (per_file_isolation=True):
       Each test file runs in its own pytest subprocess. This prevents NPU kernel
       crashes from affecting other test files. If a file crashes, it won't generate
       a JUnit XML report, and the summary script will show "Missing" for that file.

    2. Normal mode (per_file_isolation=False):
       - Serial group: Files in SERIAL_TEST_FILES are run one-at-a-time
       - Parallel group: Remaining files run together with pytest-xdist

    Args:
        planned_tests: List of test file paths (with 'test/' prefix)
        shard: Shard number
        test_dir: Path to the test directory (working directory)
        report_dir: Directory for test reports
        env_updates: Environment variable updates
        timeout: Per-test timeout
        verbose: Verbose output flag
        parallel: Number of parallel workers (pytest-xdist)
        shard_type: "distributed", "excluded", or "regular"
        per_file_isolation: If True, each file runs in isolated subprocess

    Returns:
        Tuple of (returncode, stats, log_metrics, missing_files)
        missing_files: List of test files that crashed and didn't generate XML report
    """
    start = monotonic()
    prefix = get_shard_type_prefix(shard_type)
    log_file = get_shard_log_file(report_dir, shard, shard_type)

    merged_env = os.environ.copy()
    merged_env.update(env_updates)

    # Track which files generated XML reports
    executed_files = {}  # test_file -> {"xml_expected": Path, "xml_generated": bool, "returncode": int}

    # Split tests into serial and parallel groups
    serial_tests = [t for t in planned_tests if is_serial_test(t)]
    parallel_tests = [t for t in planned_tests if not is_serial_test(t)]

    worst_returncode = 0

    with log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write("=" * 60 + "\n")
        log_handle.write(f"Direct pytest execution ({shard_type} shard)\n")
        log_handle.write("=" * 60 + "\n")
        if per_file_isolation:
            log_handle.write("Execution mode: PER-FILE ISOLATION (each file in separate subprocess)\n")
        log_handle.write(f"Total test files: {len(planned_tests)}\n")
        log_handle.write(f"Serial test files: {len(serial_tests)}\n")
        log_handle.write(f"Parallel test files: {len(parallel_tests)}\n")
        log_handle.write(f"Parallel workers: {parallel}\n")
        log_handle.write("=" * 60 + "\n\n")
        log_handle.flush()

        if per_file_isolation:
            # --- Per-file isolation mode: each file runs in separate subprocess ---
            # Use ThreadPoolExecutor for parallel execution with crash isolation
            isolation_parallel = parallel if parallel > 0 else 2  # Default to 2 workers if not specified
            log_handle.write(f"--- Per-file isolation mode: {len(planned_tests)} files ({isolation_parallel} parallel workers) ---\n\n")
            log_handle.flush()
            print(f"\n[Per-file isolation mode] Running {len(planned_tests)} files with {isolation_parallel} parallel workers:")
            print("      Each file runs in its own subprocess for crash isolation")

            # Thread lock for synchronized log writing
            log_lock = threading.Lock()

            def run_single_file_isolated(test_file: str, idx: int) -> Tuple[str, int, bool, str]:
                """
                Run a single test file in isolated subprocess.

                Returns: (test_file, returncode, xml_generated, test_name)
                """
                test_name = strip_test_prefix_and_suffix(test_file)
                safe_name = test_name.replace("/", "_")
                expected_xml = report_dir / f"shard_{prefix}-{shard}_pytest_{safe_name}.xml"

                command = build_pytest_command(
                    [test_file],
                    report_dir,
                    shard,
                    timeout,
                    verbose,
                    parallel=0,  # No xdist within the file
                    shard_type=shard_type,
                    xml_suffix=f"_{safe_name}",
                )

                # Write per-file log to separate file for isolation
                file_log_path = report_dir / f"shard_{prefix}-{shard}_log_{safe_name}.txt"

                with log_lock:
                    log_handle.write(f"\n[File {idx}/{len(planned_tests)}] {test_name}\n")
                    log_handle.write(f"  Expected XML: {expected_xml.name}\n")
                    log_handle.write(f"  Command: {' '.join(command)}\n")
                    log_handle.flush()

                print(f"  [{idx}/{len(planned_tests)}] {test_name} (parallel worker)")

                # Run pytest subprocess, capture output to per-file log
                rc = _run_pytest_subprocess_to_file(command, test_dir, merged_env, file_log_path)

                # Check if XML was generated
                xml_generated = expected_xml.exists()

                # Append per-file log to main log
                if file_log_path.exists():
                    with log_lock:
                        try:
                            file_log_content = file_log_path.read_text(encoding="utf-8", errors="replace")
                            log_handle.write(file_log_content)
                            if xml_generated:
                                log_handle.write(f"  Result: XML generated (returncode={rc})\n")
                            else:
                                log_handle.write(f"  Result: NO XML (returncode={rc}) - file may have crashed\n")
                            log_handle.flush()
                        except Exception:
                            pass

                if not xml_generated:
                    print(f"    Warning: No XML report generated for {test_name}")

                return (test_file, rc, xml_generated, test_name)

            # Execute files in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=isolation_parallel) as executor:
                futures = {}
                for idx, test_file in enumerate(planned_tests, 1):
                    future = executor.submit(run_single_file_isolated, test_file, idx)
                    futures[future] = test_file

                # Collect results as they complete
                for future in as_completed(futures):
                    test_file, rc, xml_generated, test_name = future.result()
                    executed_files[test_file] = {
                        "xml_expected": report_dir / f"shard_{prefix}-{shard}_pytest_{test_name.replace('/', '_')}.xml",
                        "xml_generated": xml_generated,
                        "returncode": rc,
                        "test_name": test_name,
                    }
                    if rc != 0 and rc != 5:
                        worst_returncode = rc if worst_returncode == 0 else worst_returncode

        else:
            # --- Normal mode: serial/parallel split ---
            # --- Phase 1: Serial tests (one-at-a-time) ---
            if serial_tests:
                log_handle.write(f"--- Serial phase: {len(serial_tests)} files ---\n\n")
                log_handle.flush()
                print(f"\n[Serial phase] Running {len(serial_tests)} files one-at-a-time:")

                for idx, test_file in enumerate(serial_tests, 1):
                    test_name = strip_test_prefix_and_suffix(test_file)
                    # Sanitize test_name for use in filename (replace / with _)
                    safe_name = test_name.replace("/", "_")
                    expected_xml = report_dir / f"shard_{prefix}-{shard}_pytest_{safe_name}.xml"

                    executed_files[test_file] = {
                        "xml_expected": expected_xml,
                        "xml_generated": False,
                        "returncode": None,
                        "test_name": test_name,
                    }

                    command = build_pytest_command(
                        [test_file],
                        report_dir,
                        shard,
                        timeout,
                        verbose,
                        parallel=0,
                        shard_type=shard_type,
                        xml_suffix=f"_{safe_name}",
                    )

                    print(f"  [{idx}/{len(serial_tests)}] {test_name}")
                    log_handle.write(f"[Serial {idx}/{len(serial_tests)}] {test_name}\n")
                    log_handle.write(f"  Command: {' '.join(command)}\n")
                    log_handle.flush()

                    rc = _run_pytest_subprocess(command, test_dir, merged_env, log_handle)
                    executed_files[test_file]["returncode"] = rc
                    executed_files[test_file]["xml_generated"] = expected_xml.exists()

                    if rc != 0 and rc != 5:
                        worst_returncode = rc if worst_returncode == 0 else worst_returncode

            # --- Phase 2: Parallel tests (all at once with -n=parallel) ---
            if parallel_tests:
                log_handle.write(f"\n--- Parallel phase: {len(parallel_tests)} files (workers: {parallel}) ---\n\n")
                log_handle.flush()
                print(f"\n[Parallel phase] Running {len(parallel_tests)} files with {parallel} workers:")

                command = build_pytest_command(
                    parallel_tests,
                    report_dir,
                    shard,
                    timeout,
                    verbose,
                    parallel=parallel,
                    shard_type=shard_type,
                )

                print("  " + " ".join(command))
                log_handle.write(f"  Command: {' '.join(command)}\n")
                log_handle.flush()

                rc = _run_pytest_subprocess(command, test_dir, merged_env, log_handle)
                if rc != 0 and rc != 5:
                    worst_returncode = rc if worst_returncode == 0 else worst_returncode

                # For parallel tests, we don't have per-file XML tracking
                # The single XML covers all parallel tests
                parallel_xml = report_dir / f"shard_{prefix}-{shard}_pytest.xml"
                for test_file in parallel_tests:
                    executed_files[test_file] = {
                        "xml_expected": parallel_xml,
                        "xml_generated": parallel_xml.exists(),
                        "returncode": rc,
                        "test_name": strip_test_prefix_and_suffix(test_file),
                    }

    # --- Identify missing files (crashed without generating XML) ---
    missing_files = []
    for test_file, info in executed_files.items():
        if not info["xml_generated"]:
            missing_files.append(test_file)

    # Save missing files list for summary script
    if missing_files:
        missing_file_path = report_dir / f"shard_{prefix}-{shard}_missing_files.txt"
        with missing_file_path.open("w", encoding="utf-8") as f:
            for test_file in missing_files:
                f.write(f"{test_file}\n")
        print(f"\nWarning: {len(missing_files)} files did not generate XML reports (likely crashed)")

    # --- Aggregate stats from all generated XML files ---
    xml_files = sorted(report_dir.glob(f"shard_{prefix}-{shard}_pytest*.xml"))
    stats = aggregate_junit_stats([report_dir])
    stats["junit_generated"] = bool(xml_files)
    stats["junit_xml_files"] = len(xml_files)
    stats["serial_test_files"] = len(serial_tests)
    stats["parallel_test_files"] = len(parallel_tests)
    stats["per_file_isolation"] = per_file_isolation
    stats["missing_files_count"] = len(missing_files)

    # Handle returncode=5 (no tests collected) as success when no real failures
    returncode = worst_returncode
    if worst_returncode == 5 and stats.get("total", 0) == 0 and stats.get("failed", 0) == 0 and stats.get("errors", 0) == 0:
        returncode = 0
        print("Tests collected no items after file filtering and testcase deselection.")

    elapsed = monotonic() - start
    stats = finalize_stats(stats, returncode, elapsed)

    log_metrics = analyze_pytest_log(log_file, worst_returncode)
    log_metrics["test_failures"] = stats.get("failed", 0) + stats.get("errors", 0)
    log_metrics["missing_files_count"] = len(missing_files)
    stats.update(log_metrics)

    if returncode != 0:
        print(f"\n{shard_type.capitalize()} shard tests completed with errors (returncode: {returncode})")
    else:
        print(f"\n{shard_type.capitalize()} shard tests completed successfully")

    return returncode, stats, log_metrics, missing_files


def _run_pytest_subprocess(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_handle,
) -> int:
    """
    Run a pytest subprocess, streaming output to both stdout and log_handle.

    Returns the process return code.
    """
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    raw_returncode = 0
    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
        raw_returncode = process.wait()
    except BaseException:
        process.kill()
        raw_returncode = 1
    finally:
        if process.stdout is not None:
            process.stdout.close()

    return raw_returncode


def _run_pytest_subprocess_to_file(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_file: Path,
) -> int:
    """
    Run a pytest subprocess, capturing output to a file (for parallel execution).

    This function is used in per-file isolation mode where multiple files run
    concurrently. Each file's output is captured to a separate file, then merged
    into the main log after completion.

    Returns the process return code.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    raw_returncode = 0
    try:
        with log_file.open("w", encoding="utf-8") as f:
            assert process.stdout is not None
            for line in process.stdout:
                f.write(line)
                # Also print to stdout for real-time visibility
                sys.stdout.write(line)
                sys.stdout.flush()
        raw_returncode = process.wait()
    except BaseException:
        process.kill()
        raw_returncode = 1
    finally:
        if process.stdout is not None:
            process.stdout.close()

    return raw_returncode


def main():
    args = parse_args()

    # Use test_type from args directly instead of deriving from shard number
    shard_type = args.test_type
    shard_index = args.shard
    shard_total = args.num_shards

    if args.shard < 1 or args.shard > args.num_shards:
        raise ValueError(f"Invalid shard {args.shard}; expected 1 <= shard <= {args.num_shards}")

    timestamp = datetime.now().isoformat()
    info = create_shard_info(args.shard, args.num_shards, timestamp)
    info["shard_type"] = shard_type
    info["shard_index"] = shard_index
    info["shard_total"] = shard_total
    info["disabled_count"] = load_disabled_testcases_count(args.disabled_testcases)
    info["parallel_procs"] = args.parallel  # Record parallel worker count

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    repo_root = test_dir.parent  # Parent of test directory
    script_dir = Path(__file__).resolve().parent
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load crashed files blacklist (applied first to prevent CI crashes)
    crashed_config_file, crashed_files_list = load_crashed_files_list(args.crashed_files_config)
    info["crashed_files_count"] = len(crashed_files_list)
    if crashed_config_file:
        info["crashed_files_config"] = crashed_config_file

    case_paths_file, whitelist, blacklist = load_case_path_rules(args.case_paths_config)
    info["whitelist_entries"] = len(whitelist)
    info["blacklist_entries"] = len(blacklist)
    if case_paths_file:
        info["path_rules_file"] = case_paths_file

    # Get test file list - use TESTS list from discover_tests.py (same as run_test.py --help)
    # unless --use-raw-discovery is specified
    if args.use_tests_list and not args.use_raw_discovery:
        raw_test_files = get_tests_list_from_discover_tests(test_dir)
        info["test_discovery_mode"] = "TESTS_list"
    else:
        raw_test_files = discover_raw_test_files(test_dir)
        info["test_discovery_mode"] = "raw_file_scan"

    # First exclude crashed files to prevent process crashes
    if crashed_files_list:
        crashed_excluded = [path for path in raw_test_files if any(path_matches_rule(path, rule) for rule in crashed_files_list)]
        raw_test_files = [path for path in raw_test_files if not any(path_matches_rule(path, rule) for rule in crashed_files_list)]
        info["crashed_excluded_files"] = len(crashed_excluded)
    else:
        info["crashed_excluded_files"] = 0

    # Filter tests by shard type (distributed or regular) based on --test-type argument
    type_selected_files, type_excluded_files = filter_tests_by_type(raw_test_files, shard_type)
    info["type_selected_files"] = len(type_selected_files)
    info["type_excluded_files"] = len(type_excluded_files)

    # Apply whitelist/blacklist rules to type-selected files
    selected_test_files, excluded_test_files = apply_case_path_rules(type_selected_files, whitelist, blacklist)
    info["whitelist_blacklist_selected"] = len(selected_test_files)
    info["whitelist_blacklist_excluded"] = len(excluded_test_files)

    # Select files for this shard (within the shard type group)
    planned_tests = select_shard_files(selected_test_files, shard_index, shard_total)

    # Record counts
    info["total_files"] = len(raw_test_files)
    info["selected_test_files"] = len(selected_test_files)
    info["path_filtered_out_files"] = len(excluded_test_files)
    info["excluded_test_files"] = len(excluded_test_files)
    info["shard_files"] = len(planned_tests)

    save_test_plan_file(str(report_dir), args.shard, planned_tests, shard_type)
    save_excluded_test_files_file(str(report_dir), args.shard, excluded_test_files, shard_type)
    save_unhandled_upstream_tests_file(str(report_dir), args.shard, [], shard_type)

    print(f"\n{'=' * 60}")
    print("PyTorch NPU Test Runner (direct pytest)")
    print(f"{'=' * 60}")
    print(f"Shard: {args.shard}/{args.num_shards}")
    print(f"Shard type: {shard_type} (shard {shard_index}/{shard_total} within type)")
    print(f"Repository root: {repo_root}")
    print(f"Test directory: {test_dir}")
    # Show appropriate description based on discovery mode
    discovery_desc = {
        "TESTS_list": "TESTS list from discover_tests.py",
        "raw_file_scan": "raw file scan (all test_*.py)",
        "raw_file_scan_for_excluded_tests": "raw file scan for excluded tests (blocklisted patterns/tests)",
    }
    print(f"Test discovery mode: {info['test_discovery_mode']} ({discovery_desc.get(info['test_discovery_mode'], 'unknown')})")
    print(f"Parallel workers requested: {args.parallel}")
    if shard_type == "distributed" and not args.per_file_isolation:
        print("  (distributed tests will auto-switch to per-file isolation for crash safety)")
    if crashed_config_file:
        print(f"Crashed files config: {crashed_config_file}")
        print(f"Crashed files excluded: {info['crashed_excluded_files']}")
    if case_paths_file:
        print(f"Case path rules: {case_paths_file}")
        print(f"Whitelist entries: {info['whitelist_entries']}")
        print(f"Blacklist entries: {info['blacklist_entries']}")
    print(f"Total test files discovered: {info['total_files']}")
    print(f"Files for shard type '{shard_type}': {info['type_selected_files']}")
    print(f"After whitelist/blacklist: {info['whitelist_blacklist_selected']}")
    print(f"Filtered out by blacklist: {info['whitelist_blacklist_excluded']}")
    print(f"Tests in this shard ({shard_type}): {len(planned_tests)}")
    print(f"Disabled testcase entries: {info['disabled_count']}")
    print(f"{'=' * 60}\n")

    for index, target in enumerate(planned_tests, 1):
        # Show test name without 'test/' prefix for clarity
        display_name = strip_test_prefix_and_suffix(target)
        print(f"  [{index:03d}] {display_name}")

    clean_existing_junit_xml(report_dir)
    remove_existing_file(Path(get_disabled_testcases_report_file(str(report_dir), args.shard, shard_type)))
    remove_existing_file(get_shard_log_file(report_dir, args.shard, shard_type))

    env_updates = build_execution_env(test_dir, script_dir, args.disabled_testcases, str(report_dir), args.shard, shard_type)

    missing_files = []
    if planned_tests:
        # Auto-select execution strategy based on test type:
        # - distributed: per-file isolation for crash safety (unless user explicitly disables)
        # - regular: normal execution with parallel workers for speed
        # User can override with --per-file-isolation and --parallel flags
        auto_per_file_isolation = args.per_file_isolation
        auto_parallel = args.parallel

        # Default strategy: distributed tests use serial per-file isolation, regular tests use parallel
        if shard_type == "distributed":
            # Distributed tests often cause NPU kernel crashes, use per-file isolation for safety
            # unless user explicitly sets --per-file-isolation=False (which is not possible via CLI,
            # so we always enable it for distributed tests for crash safety)
            if not args.per_file_isolation:
                auto_per_file_isolation = True
                # Use user's parallel setting for concurrent isolated execution,
                # or default to serial (parallel=1) for maximum safety
                if args.parallel <= 1:
                    auto_parallel = 1  # Serial execution (one file at a time)
                    print(f"Note: Distributed tests auto-switched to PER-FILE ISOLATION (serial execution)")
                    print("      This prevents NPU kernel crashes from affecting other test files")
                else:
                    # User specified parallel > 1, use it for concurrent isolated execution
                    auto_parallel = args.parallel
                    print(f"Note: Distributed tests auto-switched to PER-FILE ISOLATION mode")
                    print(f"      Running {auto_parallel} files concurrently in isolation")
                print("      Use --parallel 1 to force serial execution, or --per-file-isolation is always enabled for distributed")
            elif args.per_file_isolation and args.parallel > 1:
                print(f"Note: Running {len(planned_tests)} distributed tests in PER-FILE ISOLATION mode")
                print(f"      with {auto_parallel} parallel workers (crash isolation + concurrency)")
            else:
                print(f"Note: Running {len(planned_tests)} distributed tests via direct pytest with per-file isolation")
        else:
            # Regular tests: also use per-file isolation for crash safety
            # Each file runs in its own subprocess, preventing crashes from affecting other files
            # This ensures that even if one test file causes a segfault, other files continue
            # and each file generates its own XML report (no MISSING status due to crash)
            if not args.per_file_isolation:
                auto_per_file_isolation = True
                # Use user's parallel setting for concurrent isolated execution,
                # or default to serial (parallel=1) for maximum safety
                if args.parallel <= 1:
                    auto_parallel = 1  # Serial execution (one file at a time)
                    print(f"Note: Regular tests auto-switched to PER-FILE ISOLATION (serial execution)")
                    print("      Each file runs in separate subprocess for crash isolation")
                else:
                    # User specified parallel > 1, use it for concurrent isolated execution
                    auto_parallel = args.parallel
                    print(f"Note: Regular tests auto-switched to PER-FILE ISOLATION mode")
                    print(f"      Running {auto_parallel} files concurrently in isolation")
                print("      This prevents test crashes from affecting other files or losing all XML reports")
            elif args.per_file_isolation and args.parallel > 1:
                print(f"Note: Running {len(planned_tests)} regular tests in PER-FILE ISOLATION mode")
                print(f"      with {auto_parallel} parallel workers (crash isolation + concurrency)")
            else:
                print(f"Note: Running {len(planned_tests)} regular tests via per-file isolation")

        _, stats, log_metrics, missing_files = run_tests_via_pytest(
            planned_tests,
            args.shard,
            test_dir,
            report_dir,
            env_updates,
            args.timeout,
            args.verbose,
            auto_parallel,
            shard_type,
            per_file_isolation=auto_per_file_isolation,
        )
        info["per_file_isolation"] = auto_per_file_isolation
        info["effective_parallel"] = auto_parallel
        info["missing_files_count"] = len(missing_files)
    else:
        print("No test files assigned to this shard after file-level filtering.")
        stats = finalize_stats(create_empty_stats(), 0, 0.0)
        log_metrics = {
            "zero_item_test_files": 0,
            "startup_failures": 0,
            "import_failures": 0,
            "test_failures": 0,
            "missing_files_count": 0,
        }

    info["junit_generated"] = bool(stats.get("junit_generated", False))
    info["junit_xml_files"] = int(stats.get("junit_xml_files", 0))
    info["zero_item_test_files"] = int(log_metrics.get("zero_item_test_files", 0))
    info["startup_failures"] = int(log_metrics.get("startup_failures", 0))
    info["import_failures"] = int(log_metrics.get("import_failures", 0))
    info["test_failures"] = int(log_metrics.get("test_failures", 0))
    info.update(load_disabled_testcases_report(str(report_dir), args.shard, shard_type))

    save_info_file(str(report_dir), args.shard, info, shard_type)
    save_stats_file(str(report_dir), args.shard, stats, shard_type)
    print_stats_summary(args.shard, stats, shard_type)
    sys.exit(stats.get("returncode", 1))


if __name__ == "__main__":
    main()
