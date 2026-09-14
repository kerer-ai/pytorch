# torch-npu CI Docker Images

本目录管理 torch-npu 项目的 CI Docker 镜像，包括**构建镜像 (builder)** 和**测试镜像 (test)** 两类，每类分别支持 x86_64 和 aarch64 架构。

当前仅维护 master (nightly) 版本：

| 版本目录 | PyTorch 版本 | 镜像基座 (x86_64) | 镜像基座 (aarch64) |
|---------|------------|-------------------|---------------------|
| `master/` | 2.14.0.dev20260708 (nightly) | `pytorch/manylinux2_28-builder:cpu` | `cpu-aarch64` |

## 镜像类型

| 类型 | 基座 | 用途 |
|------|------|------|
| **builder (x86_64)** | manylinux2_28-builder | 编译构建 torch-npu wheel 包，包含完整编译工具链 |
| **builder (aarch64)** | manylinux2_28_aarch64-builder | 编译构建 torch-npu wheel 包，包含完整编译工具链 |
| **test** | `ubuntu:22.04` | CI 单元测试运行环境，包含 CANN runtime、triton-ascend 和测试框架（master 版本不含 PyTorch，由 CI 运行时单独构建安装） |

## 目录结构

```text
.ci/docker/
├── README.md                      # 本文档
├── docker_build.sh                # 构建入口脚本
├── common/                        # 公共共享脚本
│   ├── install_cann.sh            # 安装 CANN toolkit (支持 A1/A2/A3，公开仓库或 OBS 分享链接)
│   ├── install_triton.sh          # 安装 triton-ascend (需传 Python 版本)
│   ├── install_obs.sh             # 安装华为 OBS util (obsutil)
│   └── obs_share.sh               # OBS 分享链接下载助手 (share-ls / share-cp 封装)
└── master/                        # master (nightly) 版本特定
    ├── requirements-test.txt      # Test 镜像依赖 (torch nightly)
    ├── builder/
    │   ├── Dockerfile.x86_64
    │   └── Dockerfile.aarch64
    └── test/
        ├── Dockerfile.x86_64
        └── Dockerfile.aarch64
```

## 快速构建

```bash
# Builder 镜像 (不含 CANN)
./docker_build.sh torch-npu-builder-x86_64-torch-master
./docker_build.sh torch-npu-builder-aarch64-torch-master

# Test 镜像 (含 CANN)
./docker_build.sh torch-npu-test-x86_64-cann-a1-py3.10-torch-master
./docker_build.sh torch-npu-test-aarch64-cann-a2-py3.10-torch-master
```

## Tag 命名规范

参考上游 PyTorch `pytorch-linux-jammy-cuda12.4-cudnn9-py3-gcc11` 模式，tag 即为最终镜像名：

**Builder**（不含 CANN）：

```text
torch-npu-builder-<ARCH>-torch<PYTORCH_VERSION>
```

**Test**（含 CANN runtime）：

```text
torch-npu-test-<ARCH>-cann<CHIP>-py<PYTHON_VERSION>-torch<PYTORCH_VERSION>-cann<CANN_VERSION>
```

| 字段 | 可选值 |
|------|--------|
| IMAGE_TYPE | builder, test |
| ARCH | x86_64, aarch64 |
| CHIP | A1 (Ascend 910), A2 (Ascend 910b), A3 (仅 test) |
| PYTHON_VERSION | 3.10 (仅 test) |
| PYTORCH_VERSION | master (nightly) |
| CANN_VERSION | 镜像内 CANN 版本，始终携带：公开仓库为固定版本（如 `cann9.1.0-beta.3`），OBS 分享链接为分享实际版本（如 `cann9.2.0-20260910200430`） |

完整 tag 还会追加构建时间戳与 commit ID：`<BASE_TAG>-<TIMESTAMP>-<COMMIT_ID>`。

## Python 版本支持

Builder 镜像支持以下 Python 版本（由基座镜像提供）：

- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13
- Python 3.14

Test 镜像使用 Miniforge3 创建 conda 环境 `py_${PYTHON_VERSION}`（默认 Python 3.10）。

## CANN 安装来源

| 来源 | 条件 | CANN 版本 (tag 中体现) |
|------|------|----------|
| 公开仓库 (默认) | 不设置 OBS_SHARE_URL | 9.1.0-beta.3 (ascend-repo 固定版本，定义于 `common/install_cann.sh`) |
| OBS 分享链接 | 设置 OBS_SHARE_URL + OBS_ACCESS_CODE | 由分享链接自动发现 (如 9.2.0-20260910200430) |

## 从 OBS 分享链接构建（CANN 私享包）

当 CANN 包通过华为云 OBS e-share 链接分享（`https://e-share.obs-website.<region>.myhuaweicloud.com?v2token=...`）时，
可用 obsutil 的目录分享能力下载安装。链接中的 `v2token` 即 obsutil 授权码，配合提取码使用。

### 在 workflow 中使用（推荐）

在 `Build build/test Docker Images` workflow（`build-docker-images.yml`）手动触发时填写：

| Input | 说明 |
|-------|------|
| `cann_share_url` | e-share 链接（含 v2token） |
| `cann_share_code` | 提取码 |

workflow 会在 runner 上预装 obsutil（复用 `common/install_obs.sh`），并把链接与提取码通过 BuildKit
`--secret`（id=`obs_share`）传入 `docker build`，不会残留在镜像层或 `docker history` 中。

注意：`workflow_dispatch` 的输入值对能查看该 workflow run 的人可见；如需保密可改用 repo secret。

### 本地构建

```bash
# 构建机器需先安装 obsutil（需 sudo）
sudo bash .ci/docker/common/install_obs.sh

OBS_SHARE_URL='<e-share 链接>' \
OBS_ACCESS_CODE='<提取码>' \
  ./docker_build.sh torch-npu-test-x86_64-cann-a1-py3.10-torch-master
```

### 相关环境变量

| 变量 | 用途 |
|------|------|
| `OBS_SHARE_URL` | e-share 链接；设置后 `install_cann.sh` 切换为分享链接下载模式 |
| `OBS_ACCESS_CODE` | 分享链接提取码（分享模式必填） |
| `CANN_VERSION` | 期望的 CANN 版本（可选；设置后会与分享实际版本校验，不一致报错；docker_build.sh 中缺省时自动发现） |
| `DRY_RUN=1` | docker_build.sh 仅打印解析出的配置与最终 tag，不执行构建 |
| `TAG_OUT=<file>` | docker_build.sh 将最终镜像 tag 写入该文件（workflow 用于回传） |

分享目录约定为 `version_combo_snapshot/CANN <版本>/run/<arch>-linux/`，脚本按 `Ascend-cann-toolkit_*`、
`Ascend-cann-{910\|910b\|A3}-ops_*`（按 CANN_CHIP）、`Ascend-cann-nnal_*` 三个模式挑选 run 包下载安装。
