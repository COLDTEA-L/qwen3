# Qwen3 30B A3B 多路径接入部署与冲突验证手册

本实验把 CAM All-to-All Detour 接入 SGLang 的 Qwen3 Attention TP 通信，保留原生 MoE Dispatch 与 Combine 及 PD 的 Ascend KV 传输。目标是在两台各 8 卡的 A2 服务器上，部署 P4 加 D8 加 KV Pool4，并测量多路径、MoE、KV 并发时的资源竞争。当前完成的是源码接入、配置和实验工具；开发服务器无卡，未完成 A2 编译、正确性或性能验证。

先跑原生服务，再跑算子正确性，最后测并发。任何一道检查未通过，都不要把后面的数据当成有效性能结果。没有设备时间重叠或没有显著退化，也应记录为未发现冲突。

## 1 版本和交付范围

SGLang 基于 f620d73，分支为 feat/qwen3-cam-detour-conflict。UMDK 基于 all_gather_4_scheduler 的 bf74c6db，配套构建分支为 feat/qwen3-cam-detour-build。后续不要直接换成最新 main。两机的 SGLang、UMDK、CANN、TorchNPU、DeepEP NPU 适配和 MemFabric 版本应一致。

本实验使用原版 Qwen/Qwen3-30B-A3B BF16。它有 48 层、128 个专家、每 token 激活 8 个专家、4 个 KV heads 和 128 维 head。权重只在有卡的部署服务器下载；开发服务器不下载权重。

已交付的功能包括 Attention TP AllReduce 和 ReduceScatter 接入、三份角色配置、KV Pool 与中继窗口常驻进程、真实 MemCache put get 探测、算子冒烟测试、三类流量的受控回放和延迟分析。没有实现统一 Fabric 控制器、任意路径动态仲裁、生产图模式或对任意模型通用的替换。

代码入口位于 python/sglang/srt/distributed/device_communicators/cam_detour.py。Qwen3 模型构造时显式初始化；Attention 通信入口和层边界归约入口接入。变量 SGLANG_CAM_DETOUR_MODE 默认 off，只有显式启用才生效。

## 2 两机布局和通信域

| 机器 | 物理卡 | 角色 | 模型并行设置 | 同机绕行内存提供者 |
| --- | --- | --- | --- | --- |
| A | 0 至 3 | P | 启动 TP4 Attention TP4 EP4 | A4 至 7 的 D worker |
| A | 4 至 7 | D 的前半 | D 全局启动 TP8 Attention DP2 EP8 | A0 至 3 的 P worker |
| B | 0 至 3 | D 的后半 | D 全局启动 TP8 Attention DP2 EP8 | B4 至 7 的 Pool sidecar |
| B | 4 至 7 | KV Pool | 不加载模型 每卡初始 8GB HBM pool | 自身提供 CAM 窗口 |

D 只有一套跨两机的 EP8，Attention 则分为两个同机 TP4 组。SGLang 的启动参数 TP8 表示 8 个模型 worker，不表示 Attention 必须做 TP8。EP 是专家分布维度，不再额外乘一遍卡数；D 的专家内部 TP 为 1。

P 和 D 的 MoE 通信域彼此独立。两者共同加入的是同机 CAM 的物理内存资源域，不是把两个模型的专家集合通信混成一个组。CAM 共 8 个物理 rank，某次多路径通信的 commRankIds 只有 4 个，其余 4 个提供中继内存。P 与 D 的 CAM rank 以物理卡号编号，D 的逻辑 rank 另有映射。

| 调用方 | 引擎逻辑 rank | CAM 物理 rank | 配置 |
| --- | --- | --- | --- |
| A 的 P | 0 至 3 | 0 至 3 | configs/p_a.json |
| A 的 D | 0 至 3 | 4 至 7 | configs/d_a.json |
| B 的 D | 4 至 7 | 0 至 3 | configs/d_b.json |
| B 的 Pool | sidecar 本地 0 至 3 | 4 至 7 | 使用同机 d_b.json 的窗口参数 |

本实验要求每个进程能访问同机其余卡的 IPC 内存。不要仅向 P 容器映射 4 张卡或把可见设备重编号。两机容器均映射全部 8 个 davinci 设备、davinci_manager、hisi_hdc、driver 和 firmware；仅由 base-gpu-id 决定模型运行卡。保留共享 IPC 命名空间与运行时所需权限。模型卡和 Pool 卡可以被同机通信访问，不等于另一进程会在这些卡上加载模型。

## 3 算子接入的含义和对照

Attention 原本的 SUM AllReduce 被实现为先按目标 rank 切成 4 份，经 All-to-All 交换；每个 rank 对来自 4 个源的同一分片用 FP32 求和并转回输入精度；最后用原生 HCCL AllGather 收集各个归约分片。ReduceScatter 则保留交换和求和，不执行最后的 AllGather。

CAM exchange 沿用原有 All-to-All Detour 的直连加中继路径，未调用 HCCL All-to-All。最后的 AllGather 仍是 HCCL，因此不能把整个 AllReduce 称为纯 CAM。MoE 的 native DeepEP NPU 实现保持不变，不把均匀 All-to-All 当成 token 路由。

| 模式 | 交换和求和逻辑 | 用途 |
| --- | --- | --- |
| off | 原生 Attention 集合通信 | 服务正确性和生产实现基线 |
| hccl-a2a | 相同分块 用 HCCL All-to-All 交换 再 FP32 求和 | 控制分块 求和 额外 launch 开销 |
| cam-detour | CAM All-to-All Detour 交换 同样求和 | 多路径实验组 |

SUM 的数学语义不变，但浮点归约顺序变了，不能要求 BF16 输出逐 bit 相同。需要算子容差检查和模型质量检查。首次实验关闭图模式、torch compile、TBO、单批重叠；原生 baseline 也使用同样设置。多路径的 padding、缓冲和多次 launch 可能本身增加耗时，不能把 cam-detour 比 off 慢的全部差值归因于拥塞。

每个物理 rank 的 CAM 窗口默认 512MB。窗口先除以 8 乘 8 个源目标槽位，再分为两个 ping pong slot，每个 slot 前 2MiB 留给 flag。适配器检查 payload 是否能放下，exchange 总输入默认不超过 8MiB，按 5 条路径和 128 元素对齐分片。不要随意把 buffer 改成很小。CAM 窗口、Pool HBM、模型 KV 和临时张量的 HBM 用量需要分别预算。

AllReduce 每个 chunk 的原生 AllGather 提供组内进度约束。ReduceScatter 每个 chunk 额外设置组内 barrier，保护 ping pong 复用。它是正确性优先的实验开销，不是 Fabric 优化；hccl-a2a 也保留同样 barrier。多进程独立域的 collective 顺序必须一致，本版不允许自由跨 stream 重排同一组的 CAM 调用。

## 4 按顺序准备：代码、私有配置、权重、镜像

本节更新于 2026-10-10。下文是一套可执行的部署与验收流程，不是已完成的 A2 实测报告。先在无卡开发服务器推送本次分支，再在目标机器拉取。除非明确写“容器内”，命令在宿主机执行。每个代码块可逐行执行，不需要输入续行反斜杠。

### 4.0 已确定的环境与剩余运行条件

| 项目 | 已确认的信息 / 处理方式 |
| --- | --- |
| A、B 宿主机 | 两机各 8 张 Ascend 910B4，每卡 32GB；aarch64，EulerOS 2.0 SP10，driver 25.5.2 |
| 拓扑与接口 | 同机 FullMesh/HCCS；两机主机控制网接口均为 enp67s0f5 |
| 工作目录 | 两机都是 /ssd_2；无卡开发服务器仍为 /home/liuyuanwen |
| 复用的基础镜像 | B 已有 ImageID：sha256:39c97f336132e02b9e7066428ddde439e763623e4024c1f405049f79b5be4ad6 |
| 基础镜像用户态 | Ubuntu 22.04.5、Python 3.11.15、CANN 9.0.0 |
| 基础镜像 Python 包 | SGLang 0.5.17.dev386+gc5bd3d7dc；Torch 2.10.0+cpu；torch_npu 2.10.0；sgl_kernel_npu 2026.6.1；DeepEP 1.0.0+48942e37.cann.9.0.0.b250；MemFabric 1.0.8；Transformers 5.12.1 |
| 当前缺包 | MemCache 不存在；基础镜像不能直接启动真实 Pool |
| 磁盘 | B Docker 目录最新剩 166GB，A 此前剩 617GB；权重和镜像导出文件放 /ssd_2，不放 Docker 根目录 |
| 设备窗口 | 上次两机全部卡均有 Python 进程；模型启动前仍需所有者确认释放/授权使用整套 16 卡 |
| 尚未实测 | 当前源码与旧 kernel 兼容性、CAM 编译/IPC、DeepEP EP8、PD RDMA、MemCache Pool 和性能 |

不再要求复制全量 listen 端口、接口、镜像或硬件清单。可以先拉代码、下载权重、构建不占 NPU 的镜像。运行第 5 节以后要有 16 卡实验窗口：与现有任务所有者约好时段，退出自己的旧任务后，两机分别检查 `npu-smi info`。AICore=0 不等于卡已释放，不停止、删除或杀掉他人的进程。

基础镜像版本查询留作复查，在 B 宿主机运行：

```bash
docker run --rm --network=none --runtime=runc --entrypoint python sha256:39c97f336132e02b9e7066428ddde439e763623e4024c1f405049f79b5be4ad6 -m pip show sglang torch torch-npu sgl-kernel-npu deep-ep memfabric-hybrid memcache-hybrid transformers | awk '/^(Name|Version):/ {print}'
```

这里的 sha256 是本机 ImageID，不是可供其他机器从镜像仓库拉取的 RepoDigest。`torch 2.10.0+cpu` 配合 `torch_npu` 是本次观测到的组合，不要因此装 CUDA Torch。

### 4.1 两台宿主机拉实验源码

先执行第 14 节 push。然后 A、B 各执行一次：

```bash
test -w /ssd_2
git clone --branch feat/qwen3-cam-detour-conflict --single-branch https://github.com/COLDTEA-L/qwen3.git /ssd_2/sglang
cd /ssd_2/sglang
git status --short --branch
git rev-parse HEAD
```

如果 /ssd_2/sglang 已存在，不再次 clone，不覆盖目录。先查看状态；仅在工作区干净且分支正确时更新：

```bash
cd /ssd_2/sglang
git status --short --branch
git remote -v
git switch feat/qwen3-cam-detour-conflict
git pull --ff-only
git rev-parse HEAD
```

如果有本地改动、origin 不是自己的 qwen3 Fork、或 ff-only 失败，停在这里核对，不 reset。两机 HEAD 必须相同。镜像中的旧 SGLang 仅提供基础依赖，运行时通过 PYTHONPATH 使用 /ssd_2/sglang/python 的实验源码。

UMDK 是独立仓库，不能只拉 qwen3 就认为已有 CAM。无卡开发服务器执行：

```bash
cd /home/liuyuanwen/umdk
git status --short --branch
git switch feat/qwen3-cam-detour-build
git bundle create /home/liuyuanwen/umdk-qwen3-cam.bundle feat/qwen3-cam-detour-build
```

A、B 宿主机分别复制、克隆：

```bash
scp root@8.146.225.22:/home/liuyuanwen/umdk-qwen3-cam.bundle /ssd_2/umdk-qwen3-cam.bundle
git clone --branch feat/qwen3-cam-detour-build /ssd_2/umdk-qwen3-cam.bundle /ssd_2/umdk
git -C /ssd_2/umdk rev-parse HEAD
```

开发服务器登录账户如不是 root，改为自己的真实账户。配套 UMDK 预期 HEAD 为 9c0bbdadfcfe882426cabdce5cdf2d5d2fce92ef。已有 /ssd_2/umdk 时先检查改动/分支，再从 bundle fetch 和 ff-only 合并；不覆盖用户目录。

### 4.2 一次性生成两机私有配置

只在 B 宿主机生成；脚本会交互询问 A 地址、B 地址、SSH 用户。填写已确认的两个登录/主机通信地址和 yuanwenliu；这些信息只保存在 Git 忽略的 runtime-config 中。主机地址不是各 NPU 的 RoCE 地址。

```bash
cd /ssd_2/sglang
python3 experiments/qwen3_cam/configure_runtime.py
source experiments/qwen3_cam/runtime-config/deployment.env
ssh "${A2_SSH_USER}@${A_IP}" 'mkdir -p /ssd_2/sglang/experiments/qwen3_cam/runtime-config && chmod 700 /ssd_2/sglang/experiments/qwen3_cam/runtime-config'
scp experiments/qwen3_cam/runtime-config/deployment.env experiments/qwen3_cam/runtime-config/meta.json experiments/qwen3_cam/runtime-config/pool.json experiments/qwen3_cam/runtime-config/client.json "${A2_SSH_USER}@${A_IP}:/ssd_2/sglang/experiments/qwen3_cam/runtime-config/"
```

如果文件已存在，脚本拒绝覆盖。先查看旧配置，保留正确值或用编辑器逐项修订；不要为方便而直接重写已有配置。生成结果：

- deployment.env：两机地址、接口、模型路径、SSH 账户。
- meta.json：B 的 MemCache 元数据域；5000、6000、8100。
- pool.json：B4–7 每卡 8GB HBM，DRAM=0，DEVICE_RDMA。
- client.json：相同元数据域，HBM=0、DRAM=0；客户端不能悄悄成为额外存储池。

以后每个新容器终端都执行 `source experiments/qwen3_cam/env.sh A` 或 `... B`。它初始化 CANN、源码路径、控制网和 PD store，清除旧 OpenSora 的 RANK/WORLD_SIZE 等变量；容器能看到全部八卡，不重编号。不要给 SGLang 服务套 16 进程 torchrun。

### 4.3 在 B 构建两个隔离镜像

不修改或 commit 旧 yuanwen_sglang_a2。新镜像复用已存在的基础层，apt/pip 仅作用于新镜像；构建前后检查 Docker 盘余量：

```bash
df -h /ssd_1/docker /ssd_2
docker system df
docker image inspect sha256:39c97f336132e02b9e7066428ddde439e763623e4024c1f405049f79b5be4ad6 --format '{{.Id}} {{.Architecture}}'
docker tag sha256:39c97f336132e02b9e7066428ddde439e763623e4024c1f405049f79b5be4ad6 qwen3-cam:a2-base-observed
cd /ssd_2/sglang
cp python/pyproject_npu.toml experiments/qwen3_cam/docker/source-pyproject_npu.toml
docker build --progress=plain -f experiments/qwen3_cam/docker/Dockerfile.a2-core -t qwen3-cam:a2-core-v1 experiments/qwen3_cam/docker
```

执行 tag/build 前，先确认 a2-base-observed、a2-core-v1、a2-pool-v1 尚不是他人/旧实验在用的标签；若已存在，先核对 Id，或使用一个新的标签并修改后续引用，不覆盖。先给已知 ImageID 创建本地名称，避免构建器把 FROM sha256:... 当作远端仓库名。core 镜像：

- 保留 Torch/TorchNPU、sgl_kernel_npu、DeepEP、MemFabric 1.0.8 的观测版本。
- 按当前源码的 NPU pyproject 解析 Python 依赖，Transformers 固定为源码要求的 5.19.0；不是普通 CUDA 安装方式。
- 安装编译工具和 Router 0.3.2；要求原镜像自带当前 Python 3.11 对应的 Python.h。
- 若依赖解算/headers 失败，停止并保存构建错误，不解除约束去装另一套 Torch/kernel，不用 Ubuntu 的 Python3.10 headers 顶替。

真实 Pool 需要另一组版本。MemCache 1.0.x 不提供本实验使用的 LocalConfig/MetaConfig 配置方式；这里采用候选组合 MemCache 1.1.0 + MemFabric 1.1.0，固定 aarch64/cp311 wheel 和 SHA256。升级 MemFabric 后必须重新验收 PD，不能沿用 1.0.8 的通过结论。

```bash
docker build --progress=plain -f experiments/qwen3_cam/docker/Dockerfile.a2-pool --build-arg A2_CORE_IMAGE=qwen3-cam:a2-core-v1 -t qwen3-cam:a2-pool-v1 experiments/qwen3_cam/docker
docker run --rm --network=none --runtime=runc --entrypoint python qwen3-cam:a2-core-v1 -m pip show torch torch-npu sgl-kernel-npu deep-ep memfabric-hybrid transformers sglang-router | awk '/^(Name|Version):/ {print}'
docker run --rm --network=none --runtime=runc --entrypoint python qwen3-cam:a2-pool-v1 -m pip show memfabric-hybrid memcache-hybrid | awk '/^(Name|Version):/ {print}'
df -h /ssd_1/docker
```

core 预期 MemFabric=1.0.8、Transformers=5.19.0；pool 预期 MemFabric=1.1.0、MemCache=1.1.0。构建成功只证明安装/版本检查成功，不证明 native ABI、RDMA 或模型已通过。若空间不够，不删他人镜像/容器，不执行 docker system prune；先停止新的构建并核对本次增量。

### 4.4 把同一镜像复制给 A，创建新容器

B 宿主机执行；tar 放 /ssd_2，导出不会占 Docker 根目录第二份 tar 空间，但 A 导入层仍需 Docker 盘空间：

```bash
docker save -o /ssd_2/qwen3-cam-a2-v1.tar qwen3-cam:a2-core-v1 qwen3-cam:a2-pool-v1
sha256sum /ssd_2/qwen3-cam-a2-v1.tar
source /ssd_2/sglang/experiments/qwen3_cam/runtime-config/deployment.env
scp /ssd_2/qwen3-cam-a2-v1.tar "${A2_SSH_USER}@${A_IP}:/ssd_2/"
```

A 宿主机执行，核对 tar SHA256 与 B 一致，再导入：

```bash
sha256sum /ssd_2/qwen3-cam-a2-v1.tar
df -h /ssd_1/docker
docker load -i /ssd_2/qwen3-cam-a2-v1.tar
```

两机各执行；对比 image inspect 的 Id 必须相同，确认新容器名未被占用：

```bash
docker image inspect qwen3-cam:a2-core-v1 qwen3-cam:a2-pool-v1 --format '{{.Id}} {{.Architecture}}'
bash /ssd_2/sglang/experiments/qwen3_cam/docker/create_container.sh qwen3-cam:a2-core-v1 qwen3-cam-core
bash /ssd_2/sglang/experiments/qwen3_cam/docker/create_container.sh qwen3-cam:a2-pool-v1 qwen3-cam-pool
docker ps --filter name=qwen3-cam --format '{{.Names}} {{.Image}} {{.Status}}'
```

脚本映射所有 8 个 davinci、manager/devmm_svm/hisi_hdc，挂载 host driver/firmware/npu-smi（只读）及 /ssd_2（读写），使用 host 网络/IPC 和锁页权限，不开 privileged。它拒绝替换已有容器，缺设备/目录即停止。容器先只运行 sleep，不占模型 HBM；不要让两个容器同时启动实验 worker。新容器显式使用 Bash entrypoint，由 env.sh 初始化 CANN，不依赖旧容器的历史启动脚本。

以 A 为例进入 core：

```bash
docker exec -it qwen3-cam-core /bin/bash
source /ssd_2/sglang/experiments/qwen3_cam/env.sh A
```

B 用同样 docker exec，但 source 最后一项为 B。每个后续终端都要 source；宿主机 export 不会自动传入 docker exec。

### 4.5 在 B 下载 BF16 权重，再复制给 A

在 B 的新 core 容器内使用已确认的 Python3.11创建独立下载环境，不依赖宿主机未知的Python版本，不向旧 SGLang 容器装包。约需60GB权重空间及下载临时余量；不要下载AWQ/GPTQ/FP8变体。宿主机执行：

```bash
docker exec -it qwen3-cam-core /bin/bash
```

进入后执行（下载阶段不加载模型、不占NPU）：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python -m venv /ssd_2/qwen3-download-venv
/ssd_2/qwen3-download-venv/bin/python -m pip install modelscope
mkdir -p /ssd_2/models
/ssd_2/qwen3-download-venv/bin/python -c 'from modelscope import snapshot_download; print(snapshot_download("Qwen/Qwen3-30B-A3B", local_dir="/ssd_2/models/Qwen3-30B-A3B"))'
/ssd_2/qwen3-download-venv/bin/python -m pip freeze > /ssd_2/qwen3-download-versions.txt
```

不要先source开启offline的env.sh；如果已经source，先unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE。下载完成后检查配置、索引与权重完整性（仍在B的core容器）：

```bash
/ssd_2/qwen3-download-venv/bin/python -c 'import json; from pathlib import Path; p=Path("/ssd_2/models/Qwen3-30B-A3B"); c=json.loads((p/"config.json").read_text()); expected={"model_type":"qwen3_moe","num_hidden_layers":48,"num_experts":128,"num_experts_per_tok":8,"num_key_value_heads":4,"head_dim":128}; assert all(c.get(k)==v for k,v in expected.items()),c; i=json.loads((p/"model.safetensors.index.json").read_text()); shards=set(i["weight_map"].values()); assert all((p/n).is_file() and (p/n).stat().st_size>0 for n in shards); print("MODEL_FILES_PASS",len(shards),expected)'
```

然后exit返回B宿主机，复制到A：

```bash
source /ssd_2/sglang/experiments/qwen3_cam/runtime-config/deployment.env
ssh "${A2_SSH_USER}@${A_IP}" 'mkdir -p /ssd_2/models/Qwen3-30B-A3B'
rsync -a --info=progress2 /ssd_2/models/Qwen3-30B-A3B/ "${A2_SSH_USER}@${A_IP}:/ssd_2/models/Qwen3-30B-A3B/"
```

两机使用这一份下载快照，不各自下载漂移的 main。两机各执行以下校验清单，比较清单内容；这比文件数量相同更可靠：

```bash
cd /ssd_2/models/Qwen3-30B-A3B
find . -type f \( -name '*.safetensors' -o -name '*.json' \) -print0 | sort -z | xargs -0 sha256sum > /ssd_2/qwen3-model-sha256.txt
```

下载内容和清单不进入 Git。模型源码来自 SGLang 实验分支，权重里的配置/tokenizer 来自 Qwen，不再单独安装另一份 transformers 源码仓。

### 4.6 启动前仅检查本实验资源

两机宿主机检查设备归属、控制接口和本实验端口：

```bash
npu-smi info
ip -br addr show enp67s0f5
ss -lnt | awk 'NR==1 || $4 ~ /:(5000|6000|7100|8100|8998|10067|29500|29610|29611|29612|29620|30000|30001|8000)$/'
```

只需要这些端口的结果，不要复制全量 listen 清单。角色用途：A 的 7100=PD store、8998=P bootstrap、29500=D 建组、30000/30001=P/D HTTP、8000=router；B 的 5000/6000/8100=Pool meta/config/metrics；两机 10067=CAM 同机地址交换；296xx=测试/sidecar torchrun。启动前端口应空闲，启动后属于本实验是正常的。占用时先查所有者，不直接杀进程或随意更换单边端口。

HCCL/Gloo 的 enp67s0f5 是主机建连接口，不代表 RoCE 数据路径已经通过。若后续 EP/PD/RDMA 建连失败，才逐卡定向查询：

```bash
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i "$i" -ip -g; done
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i "$i" -net_health -g; done
```

如果工具不在 PATH，先找 /usr/local/Ascend/driver/tools/hccn_tool；没有权限请管理员读取，不盲目修改 IP/路由/交换机。历史 OpenSora 16 卡成功仅证明那套历史环境能通信，不是本实验后端的验收。

## 5 第一阶段：先跑通原生 PD，不装 CAM/Pool

以下命令都在新 core 容器内。A、B 先各加载 env.sh，CPU 测试与只导入模型代码的检查：

```bash
python experiments/qwen3_cam/test_cpu.py
python -c 'import sglang; print(sglang.__file__); assert "/ssd_2/sglang/python/" in sglang.__file__'
python experiments/qwen3_cam/preflight.py --config experiments/qwen3_cam/configs/p_a.json --model-path "$MODEL_PATH" --mode off --import-model
```

B 的 preflight 改用 configs/d_b.json。要求 CPU 测试 OK、MODEL_IMPORT_OK、DEVICES_OK 8 和 PREFLIGHT_PASS。不加载权重就导入 Qwen3，可较早发现源码要求的新 kernel API 与 2026.6.1 不兼容。若失败，保存完整错误并停下；当前已知版本足够制定流程，不等于已经证明旧 kernel 支持当前源码。不要用别的 A3/950 wheel 强行补包。

现在分别准备三个前台终端，每个都先 docker exec 进入 core 并 source 正确 A/B。首次用 4K 上下文、最大活跃请求 4，避免一开始大批量占满 32GB：

**A：P 终端**

```bash
export SGLANG_CAM_DETOUR_MODE=off
bash experiments/qwen3_cam/launch.sh p-a --context-length 4096 --max-running-requests 4 --chunked-prefill-size 1024
```

**A：D 主节点终端**

```bash
export SGLANG_CAM_DETOUR_MODE=off
bash experiments/qwen3_cam/launch.sh d-a --context-length 4096 --max-running-requests 4
```

**B：D 第二节点终端**

```bash
export SGLANG_CAM_DETOUR_MODE=off
bash experiments/qwen3_cam/launch.sh d-b --context-length 4096 --max-running-requests 4
```

A/B 的 D 必须共同启动才会完成跨机建组。SGLang 自己启动模型 worker，以上命令不套 torchrun。看日志中的 loaded weights、模型就绪、DeepEP/PD 初始化；同时查看 npu-smi 卡映射符合第 2 节。OOM 时先确认其他进程、上下文/临时内存，不立即修改并行布局。

**A：第四个 core 终端，待 P/D 就绪后启动 Router**

```bash
python -m sglang_router.launch_router --pd-disaggregation --prefill "http://${A_IP}:30000" --decode "http://${A_IP}:30001" --host 0.0.0.0 --port 8000
```

**A：第五个 core 终端发送请求**

```bash
curl --fail-with-body http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d '{"text":"Explain tensor parallelism briefly.","sampling_params":{"max_new_tokens":32,"temperature":0}}'
```

要求返回正常文本、请求成功、P→D 的 KV 传输完成。日志/trace 要确认 D 的 EP8 是 A4–7+B0–3，Attention 是两组同机 TP4；不能将未知失败静默改为 EP1 或假 KV 传输。HTTP成功还不是通信性能验收，后面会收集时间线。

本阶段结束：在自己打开的服务终端依次 Ctrl+C 停 Router、P/D；检查两机自己的子进程均退出、卡内存释放。不要用 pkill python，也不要 stop 原 yuanwen_sglang_a2。保留第一阶段日志，第二阶段 MemFabric 变更后还要重跑。

## 6 两机分别编译 CAM，做 8 卡正确性测试

core 容器内，模型/Pool 全部停下后，A、B 分别执行：

```bash
command -v msopgen
command -v cmake
command -v g++
bash experiments/qwen3_cam/install_cam.sh build
```

前三条缺任何工具都停下。msopgen应来自该CANN工具链；先确认env.sh加载了正确CANN9环境，不随意安装与CANN版本无关的生成器。

该脚本调用配套 UMDK 的 ascend910b4 构建，按当前 Torch CXX ABI 编译，安装生成的 .run 和确切 wheel；输出在 /ssd_2/umdk/output/cam/qwen3-all2all 和 comm_operator/run。它不删除原 src/cam/comm_operator/build；仅更新本实验生成目录及新容器内 CAM vendor。若找到多份 wheel/.run 会停止，不猜名字、不删除旧输出。

编译后重新 source env.sh A/B，确认 CANN vendor 环境，再每机各执行：

```bash
torchrun --nproc_per_node=8 --master_port=29610 experiments/qwen3_cam/smoke_collectives.py --mode hccl-a2a --iterations 100
torchrun --nproc_per_node=8 --master_port=29611 experiments/qwen3_cam/smoke_collectives.py --mode cam-detour --iterations 100
```

两次都要求本机 8 个 rank 打印 SMOKE_PASS。覆盖 FP16/BF16/FP32、不整除长度和跨 chunk 输入。超时、少 rank、输出容差失败都停在本节。CPU 测试通过不能替代此检查。

启动服务前，可再做 CAM 导入/配置检查：

```bash
python experiments/qwen3_cam/preflight.py --config experiments/qwen3_cam/configs/p_a.json --model-path "$MODEL_PATH" --mode cam-detour --import-model
```

B 改 configs/d_b.json。这里只检查依赖与配置；模型质量仍需第 9 节验证。

## 7 第二阶段：固定 Pool 依赖，并重新验收原生 PD

两机进入新 pool 容器（不改 core）：

```bash
docker exec -it qwen3-cam-pool /bin/bash
source /ssd_2/sglang/experiments/qwen3_cam/env.sh A
```

B 的 source 改为 B。core 容器内安装的 CAM 不自动出现在 pool 镜像内；两机的源码输出目录共享，Python/Torch/CANN ABI 一致后，在各自 pool 容器安装生成的 .run/wheel：

```bash
bash experiments/qwen3_cam/install_cam.sh install
```

再次 source 正确的 env.sh，然后执行：

```bash
python experiments/qwen3_cam/preflight.py --config experiments/qwen3_cam/configs/p_a.json --model-path "$MODEL_PATH" --mode cam-detour --require-pool --import-model
```

B 改 configs/d_b.json。要求 MemCache/MemFabric=1.1.0、POOL_API_OK、CAM_IMPORT_OK 和 PREFLIGHT_PASS；本检查不启动元数据域、不验证零容量客户端。native import 若出现找不到 libascendcl/libhcomm 等错误，先核对 CANN 环境和 wheel ABI，不能用 API名称存在掩盖失败。

按第 5 节完整重跑 PD（三个模型终端+Router+短请求），这次所有终端都在 pool 容器，模式 off；先不启 HiCache/sidecar。要求升级到 MemFabric 1.1.0 后原生 PD 仍成功，再结束自己的模型/Router。所有正式 off/hccl-a2a/cam-detour 对照都在同一 pool 环境运行，不能把第一阶段 1.0.8 与第二阶段 1.1.0 的差异归因于多路径。

Pool 容器还应重复第 6 节两个 8 卡 smoke，以排除运行时环境变化。任何一道失败都保留日志，不继续宣称 P4+D8+Pool4 部署成功。

## 8 真实 KV Pool 验收与 P 的 HiCache 接入

### 8.1 先独立验证远端存取

P/D/Router 停止，确保 A 的 probe 卡0不与模型争用。在 B 的 pool 容器另开元数据终端，source env.sh B 后：

```bash
python -m sglang.srt.mem_cache.storage.npu_memcache.start_meta_service --config_path experiments/qwen3_cam/runtime-config/meta.json
```

B 的第二个 pool 终端，source env.sh B 后启动四个 Pool owner，暂不开 CAM：

```bash
torchrun --nproc_per_node=4 --master_port=29612 experiments/qwen3_cam/pool_sidecar.py --config experiments/qwen3_cam/configs/d_b.json --pool-config experiments/qwen3_cam/runtime-config/pool.json --mode off
```

要求 physical_device=4、5、6、7 分别 KV_POOL_READY 和 RELAY_READY。只这四个服务有正 HBM 容量。A 的 pool 容器，source env.sh A 后：

```bash
python experiments/qwen3_cam/pool_roundtrip.py --client-config experiments/qwen3_cam/runtime-config/client.json --device 0 --mib 16 --iterations 20
```

要求 20 次 checksum PASS，B4–7 Pool容量/占用有相应变化，Meta/Pool日志或metrics确认远端读写：

```bash
curl --fail-with-body "http://${B_IP}:8100/metrics"
```

只看 checksum 不足以证明数据真的放在 B4–7；要补远端 placement/读取证据。如果 MemCache 1.1.0 的实际 A2 native 实现不支持零容量客户端或 DEVICE_RDMA 初始化失败，停止、保留错误，后续需针对该版本适配；不把客户端改成正容量后仍声称 Pool4。没有保证这组候选已在你的 CANN9/A2 上验证。

### 8.2 P 启用已有 HiCache 后端

保留 B 的 Meta 和 mode off sidecar。重启三个模型终端，全部在 pool 容器，模式 off。A 的 P 不使用第 5 节的裸启动行，改成：

```bash
export SGLANG_CAM_DETOUR_MODE=off
CLIENT_JSON=$(tr -d '\n' < experiments/qwen3_cam/runtime-config/client.json)
bash experiments/qwen3_cam/launch.sh p-a --context-length 4096 --max-running-requests 4 --chunked-prefill-size 1024 --enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-storage-backend npu_memcache --hicache-mem-layout page_first_kv_split --hicache-storage-backend-extra-config "$CLIENT_JSON"
```

D 的 d-a/d-b 和 Router 用第 5 节命令。先短请求通过，再两次提交相同约 2K 历史前缀。第一次检查存储写入；第二次必须有远端 Pool read/bytes 或 hit 证据，而不是只看 cached_tokens。本地 Radix 命中不证明远端恢复；可用不同前缀施加本地缓存容量压力，或仅在 off 模式下重启 P 清空本地缓存，再读同一前缀（须证明 L3 键/域保持有效）。

这个版本接入的是 P 的 HiCache：Pool→host staging→P 的模型 KV，P 再迁移给 D。没有实现 D 直接从 Pool 恢复请求，也不能保证 Pool读写与 CAM 共享某条特定 HCCS边。若写回策略/存储命名空间导致重启不能命中，记录实际行为并调整压力/策略，不伪造 Pool hit。

### 8.3 扩到综合负载

小请求与远端恢复通过后，再把 P/D 的 context-length 从 4096 恢复为 16384，P max-running-requests 从4逐步扫到16，D从4逐步扫到64；可先 P=8、D=16。不要第一轮同时增加所有变量。8K 前缀实验必须先提高上下文；16K前缀加输出会超 16384，需另增上下文或改15360前缀。

每卡32GB的容量需要同时覆盖权重、静态KV、临时张量、CAM512MiB；Pool卡为8GB HBM池加CAM窗口。OOM先根据实际占用调上下文/并发/内存预算，记录更改并让所有对照保持一致。

## 9 打开多路径接入，并做同环境三模式对照

所有进程都在 pool 容器，保留相同权重、并行、HiCache和负载设置。先比较 off、hccl-a2a，再 cam-detour；每次完整重启相应域，不能只在线切一部分 worker。

### 9.1 CAM 模式启动顺序与终端

先在自己的终端停 Router，再停 P/D，最后停 B sidecar；Meta 服务可保持。核对本实验 worker 已退出后，启动 B sidecar：

```bash
export SGLANG_CAM_DETOUR_MODE=cam-detour
torchrun --nproc_per_node=4 --master_port=29612 experiments/qwen3_cam/pool_sidecar.py --config experiments/qwen3_cam/configs/d_b.json --pool-config experiments/qwen3_cam/runtime-config/pool.json --mode cam-detour
```

B sidecar 建立 CAM 域时可能等待 B0–3 的 D，它不是一定能先独立打印最后就绪。接着在 A 的 P、A 的 D、B 的 D 终端（各先 source A/B）设置：

```bash
export SGLANG_CAM_DETOUR_MODE=cam-detour
export SGLANG_CAM_TRACE_MOE=1
```

为避免 A 的 P/D 逻辑 rank 相同而写到同一文件，各终端分别设置提交日志路径。P 终端：

```bash
export SGLANG_CAM_DETOUR_TRACE='experiments/qwen3_cam/results/cam-p-{rank}.jsonl'
CLIENT_JSON=$(tr -d '\n' < experiments/qwen3_cam/runtime-config/client.json)
bash experiments/qwen3_cam/launch.sh p-a --context-length 4096 --max-running-requests 4 --chunked-prefill-size 1024 --enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-storage-backend npu_memcache --hicache-mem-layout page_first_kv_split --hicache-storage-backend-extra-config "$CLIENT_JSON"
```

A 的 D 终端：

```bash
export SGLANG_CAM_DETOUR_TRACE='experiments/qwen3_cam/results/cam-d-a-{rank}.jsonl'
bash experiments/qwen3_cam/launch.sh d-a --context-length 4096 --max-running-requests 4
```

B 的 D 终端：

```bash
export SGLANG_CAM_DETOUR_TRACE='experiments/qwen3_cam/results/cam-d-b-{rank}.jsonl'
bash experiments/qwen3_cam/launch.sh d-b --context-length 4096 --max-running-requests 4
```

这里不要复制第5/8节的export MODE=off；上述三个启动终端都应保持cam-detour。初次仍用小上下文/小并发，四类 owner 全启动后才按第5节启动 Router并发送请求。A 的 P4+D4、B 的 D4+Pool4各构成完整 CAM8 域；模式启用时每个物理 rank 都要在线，不能让 Pool卡独占另一个隐藏设备域。

检查各 worker 的 CAM_READY、CAM_CALL 与实际 device trace。JSONL是host提交日志，不是device执行时间；仅看到日志不代表正确使用了预期路径，需第 11 节的设备时间线。

### 9.2 公平对照和关闭顺序

- off：P/D 的 MODE=off，sidecar mode off，真实 Pool仍在线。
- hccl-a2a：P/D 的 MODE=hccl-a2a，sidecar mode hccl-a2a；相同分块/归约，HCCL交换，CAM窗口不启用。
- cam-detour：P/D和sidecar都用cam-detour，全部8个物理owner在线。

每轮先短请求与质量检查，再压测。固定输入、temperature=0、seed，比较输出token/logprob和代表性准确率；BF16归约顺序变化不要求逐bit一致，但明显错误不能归入性能结果。所有模式保持 eager、禁图、禁单批重叠，不额外启TBO或EPLB。

停止顺序是 Router→P/D→sidecar→最后Meta；只停止自己启动的实验。某个owner退出后其他rank可能仍持有IPC引用，不能单独重启该rank复用旧域。先结束本机所有CAM owner、确认旧进程退出，再完整重建；必要时两机整个实验一起重启。不要清空共享Pool或删除别人的缓存，probe只删除自己的UUID key。


## 10 第七步设计实际 serving 工作负载

不预先承诺三类流量一定同时阻塞同一条链路。单个 batch 中 Attention 通信、MoE dispatch、专家计算和 combine 有依赖。关闭 TBO 的初始版本尤其可能只与 KV 搬运重叠，而不与同一引擎的另一个算子重叠。

构造一个持续 decode 背景和间歇恢复 burst。背景请求输出 512 至 1024 tokens，保持 D 每 worker 活跃批次；burst 使用相同长前缀的 agent 恢复请求，每次8或16个请求。扫描输入4K、8K、16K，缓存冷读、热读两种条件，以及总并发8、16、32、64。为每次请求记录 rid、输入输出 token 数、cached_tokens、TTFT、逐 token 时间和端到端时间，固定 seed 和 arrival schedule。

完整 BF16 KV 的理论逻辑量为每 token 96KiB，4K前缀384MiB，8K前缀768MiB，16K前缀1.5GiB。每个 Attention TP4 shard 约为四分之一。实际传输还受缓存命中、页对齐、staging 和请求落在哪个 DP 组影响，以实际注册和搬运字节为准。

可先用随机请求验证冷缓存吞吐，随后用保存的相同前缀列表产生热缓存恢复。以下命令是冷缓存示例；它不能替代 Pool 命中实验。

```bash
python -m sglang.bench_serving --backend sglang  --base-url "http://${A_IP}:8000" --model "$MODEL_PATH" --tokenizer "$MODEL_PATH"  --dataset-name random --random-input-len 8192 --random-output-len 512  --random-range-ratio 1 --num-prompts 256 --max-concurrency 32 --seed 17
```

固定请求数不足以保证负载一致：若希望测相同压力，使用相同 arrival trace；若希望测最大吞吐，再单独扫并发。不要把两者混成一个结论。首轮不启 EPLB，固定专家映射，避免迁移专家权重或负载调整引入第四类流量；之后可独立增加 EPLB 场景。

可复制的综合负载由 serving_workload.py 提供。它在固定到达时间提交短输入长输出背景请求，在第10和20秒各提交8个共享8K历史前缀的恢复请求，记录每个 rid 的 TTFT、平均 TPOT、SSE 间隔、客户端排队、缓存信息和失败。每个模式使用同样参数，run-id 和输出文件名只作区分。

```bash
python experiments/qwen3_cam/serving_workload.py  --model-path "$MODEL_PATH" --base-url "http://${A_IP}:8000"  --duration 30 --background-rps 0.5 --burst-every 10 --burst-size 8  --prefix-tokens 8192 --background-output-tokens 512  --restore-output-tokens 128 --max-concurrency 64  --run-id cam-r1 --output experiments/qwen3_cam/results/serving-cam-r1.jsonl
```

先用小负载确认无失败，再提高 background-rps 和 burst-size。若 client_queue_ms 显著增加，实际服务到达已被客户端并发门限改变，应提高门限或将该轮记为客户端限流实验。SSE 若一次返回多个 token，脚本记录的是按新增 token 数摊分的间隔，不是假装测到了逐 token device TPOT。8K前缀加输出应在16384上下文内；16K实验需增大启动 context-length，或把前缀缩到15360并核对加上输出后的总长度。共享前缀命中仍可能仅在本地，必须另外证明远端 Pool 读取。

## 11 第八步定位共享资源和时间重叠

| 组合 | 可能共享的资源 | 必须补充的证据 |
| --- | --- | --- |
| D 的多路径加 MoE | A4到A5 或 B0到B1 等直连边及 D 卡 HBM DMA | 原生 MoE 实际也走该边 并且执行区间重叠 |
| MoE 加跨机 KV | 同一 NIC 端口或共享跨机交换机上联 | 指向同一物理瓶颈的路由映射及端口计数器 |
| 多路径加 KV | 中继或目标卡 HBM DMA 以及实际存在的同机搬运边 | KV 真实路径 不能虚构它经中继转发 |
| 三者一起 | 上述多段资源及 D 卡 HBM 端口 | 各流量区间和资源使用的联合证据 |

例如 B0到B1 的 direct leg 可同时被 Attention 多路径和 EP 内通信使用；绕行 B0到B4到B1 访问 Pool 卡 HBM。如果 KV 从 B4 自己的 NIC 直接出机，它不经过 B4到B1。这种情况下共同竞争的是 B4 的 HBM或DMA，而不是那条 HCCS边。三类流量可以共同影响一个资源集合，并不要求全部走同一条物理链路。

在每个 serving 实例分别使用原生 /start_profile 和 /stop_profile。PD 的 P 与 D 采集需分别请求实例，不依赖路由器自动广播。API 中 GPU 表示通用加速设备，NPU 后端会使用对应 profiler。

```bash
curl -X POST "http://${A_IP}:30000/start_profile"  -H 'Content-Type: application/json'  -d '{"output_dir":"/ssd_2/sglang/experiments/qwen3_cam/profiles/p","start_step":3,"num_steps":30,"activities":["CPU","GPU"]}'
curl -X POST "http://${A_IP}:30001/start_profile"  -H 'Content-Type: application/json'  -d '{"output_dir":"/ssd_2/sglang/experiments/qwen3_cam/profiles/d","start_step":3,"num_steps":30,"activities":["CPU","GPU"]}'
```

发送 workload 后检查所有 worker 的实际 trace。关注 CAM_DETOUR、NATIVE_MOE、HCCL task 和 KV transfer。CPU record_function 范围只是关联入口；真正的重叠从 device task 的开始结束时间计算。不同机器时钟要同步，跨机器不要直接拼接未校准的 timestamp。若普通 profiler 看不到 RDMA 或 SDMA 任务，使用目标 CANN 所支持的系统 interconnect 采集或 MemFabric completed transfer 指标补充，不能把缺失事件当成没有流量。

链路或端口计数器须从目标硬件工具读取，先用工具帮助确认当前驱动支持哪些字段。记录忙闲时间、收发字节、拥塞或重传计数、队列积压、带宽利用率，以及每卡 HBM 使用量和可用的 HBM 带宽指标。不要把 HBM 容量占用直接当成带宽利用率。只有时间重叠加共用资源，还只是可能冲突；需要与 standalone 或串行对照比较才判断影响。

## 12 第九步受控流量回放

实际模型中如果未观察到 MoE 和多路径同时执行，可用 replay_conflict.py 单独验证硬件资源竞争的可能性。先停止全体模型和 Pool，脚本占用整套16卡。它不需要权重。

回放中的 multipath 使用真实 CAM adapter；MoE 是两次 HCCL EP8 All-to-All 的通信代理，KV 是 Pool 到 D 的 HCCL P2P payload 代理。它没有专家计算、token 不均衡路由、真实缓存对象管理或 MemFabric 协议，所以结果只能解释链路与 HBM 等资源竞争，不能作为真实 MoE 或 KV Pool 的服务性能。

```bash
# A 机
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0  --master_addr="$A_IP" --master_port=29620  experiments/qwen3_cam/replay_conflict.py --mode cam-detour  --output experiments/qwen3_cam/results/replay-cam-a --kv-target a
# B 机 同时启动
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1  --master_addr="$A_IP" --master_port=29620  experiments/qwen3_cam/replay_conflict.py --mode cam-detour  --output experiments/qwen3_cam/results/replay-cam-b --kv-target a
```

脚本自动跑单类、三组双类及三类并发，共七种组合，固定相同 payload。任务使用独立线程和 NPU stream；每段记录 NPU Event 时间。还需 profiler 确认 stream 中真正的 device task 重叠，不以线程存在代替并发证据。把 warmup 增至20、iterations增至100，每轮结束后重复至少5轮，并将单类与并发运行顺序在不同轮次交替，减少热状态和顺序偏差。

扫描 tp-mib 为1、8、32，moe-mib为1、8、32、128，kv-mib为16、64、256。kv-target a 测跨机 Pool 到 A 的 D，kv-target b 测同机 Pool 到 B 的 D。再把 mode 改为 hccl-a2a，保持所有 payload 与线程配置相同。比较同一种实现自身 standalone 与并发的比值，再比较两个实现的退化幅度。

复制两机的 rank JSONL 到不同结果目录，避免同名覆盖，随后汇总：

```bash
python experiments/qwen3_cam/analyze_replay.py  experiments/qwen3_cam/results/replay-cam-a  experiments/qwen3_cam/results/replay-cam-b  --output experiments/qwen3_cam/results/replay-summary.json
```

报告按 rank、路径目标、payload 和模式分别给出 p50 p95 p99 及 slowdown。不平均掉最慢 rank。每个测量文件必须有同条件 standalone 基线，否则脚本拒绝计算。跨轮置信区间和分位数误差需从原始 samples 再算；20个样本的 p99 只适合冒烟，不足以支持尾延迟结论。

## 13 第十步判断冲突和影响程度

| 指标 | 比较方式 | 含义 |
| --- | --- | --- |
| 通信 p95 | 并发 p95 除以同实现 standalone p95 | 特定通信流的退化倍率 |
| TTFT 和 TPOT p95 | 同输入与同到达率下实验组对原生及串行基线 | 业务关键路径影响 |
| 吞吐 | 同并发或相同 SLO 下输出 tokens每秒 | 可用服务能力变化 |
| KV 完成时延 | restore 或 transfer 完成时间减开始时间 | 恢复 deadline 被影响程度 |
| SLO miss rate | 预先固定 deadline 后逾期请求占比 | 调度层的业务代价 |
| Link busy queue HBM 带宽 | 同时间窗口的计数器或系统 trace | 区分链路堵塞与内存竞争 |

先定实验阈值再看结果。例如 p95退化低于5%且重复轮次差异落入测量噪声，记为未发现显著影响；5%到10%可记轻度；10%到30%可记明显；超过30%或明显提高 SLO miss rate可记严重。这些是本实验的报告分档，不是行业标准。5轮 paired 比较后，95%置信区间仍含1时不做确定性退化结论。

若同资源并发有显著退化，改为串行或错峰后退化恢复，并且链路 queue 或 busy 同时回落，可支持该资源拥塞的因果解释。若链路指标不变而 HBM 带宽或计算竞争变强，归类为内存或计算干扰，不写成物理链路拥塞。若没有设备区间重叠，则是正常依赖串行，不能作为统一控制器必须介入的证据。

三类共同影响不是几个倍率简单相乘。需观察并发是否拉长 MoE dispatch 或 combine、TP 通信是否阻塞后续 Attention、KV 是否延误可运行请求，再关联 TPOT 或 TTFT。对 KV 限速或恢复错峰仅作为额外干预实验，先记录完成时间和业务代价，避免只改善 MoE 而把 KV deadline 全部拖过期。

最终每条结论至少附拓扑和实际路径、device timeline、独立与并发数值、重复轮次和实现版本。允许三种结果：有链路冲突、有 HBM或DMA竞争但无同链路冲突、在当前负载下未发现冲突。开发环境无卡，当前文档不填写任何假想的实测百分比。

## 14 更新分支与 GitHub 推送

用户仓库是 https://github.com/COLDTEA-L/qwen3，代码为 SGLang Fork，不需要再建立独立模型源码仓。README 和配套工具随实验分支提交；权重/私有配置不推送。

**无卡开发服务器 8.146.225.22** 上执行；这里仍是 /home/liuyuanwen，不是 A2 的 /ssd_2：

```bash
cd /home/liuyuanwen/sglang
git status --short --branch
git switch feat/qwen3-cam-detour-conflict
git remote get-url publish
# 核对是 https://github.com/COLDTEA-L/qwen3.git，再 push
env -u GIT_ASKPASS -u SSH_ASKPASS GIT_TERMINAL_PROMPT=1 git -c core.askPass= -c credential.helper= push -u publish feat/qwen3-cam-detour-conflict
```

该命令绕过先前失效的 VS Code askpass。若要求认证，用户名 COLDTEA-L，密码位置输入具有本仓库写权限的 PAT，不是账号密码；PAT 不写入命令、URL 或 README，也不发给协作者。已有可用 SSH/凭据管理器也可按既有流程认证。本次只更新本地/开发服务器分支，不代用户发布 GitHub。

UMDK 的构建分支是另一个仓库，尚未发布时按第 4.1 节用 bundle 即可。以后若发布到真实存在的 UMDK Fork，先核对 remote 和历史，不使用虚构的 URL：

```bash
# 无卡开发服务器；UMDK_FORK_URL 由用户提供
cd /home/liuyuanwen/umdk
git status --short --branch
git switch feat/qwen3-cam-detour-build
export UMDK_FORK_URL=你的真实UMDKFork地址
# 仅当 publish 不存在时添加，已有则先核对 git remote get-url publish
git remote add publish "$UMDK_FORK_URL"
git push -u publish feat/qwen3-cam-detour-build
```

## 15 失败定位和回滚

CAM 初始化一直等待时，检查 A 上 P4与D4是否全启动，B 上 D4与Pool4是否全启动，端口10067和comm_id7是否一致，以及物理rank有没有重复或隐藏。PID或IPC错误首先检查容器可见设备、共享IPC和peer访问权限，不改成跳过检查。

只有原生模式成功、cam-detour失败时，保留算子输入shape、dtype、rank映射和device日志，重新跑单机冒烟。关闭模式只停止替换，不释放其他rank已经引用的旧窗口；修改模式和重启部分worker时按完整域重建。

只有启动日志没有CAM_CALL时，检查融合和层边界是否绕过接入点。若HCCL baseline也失败，问题尚未证明来自CAM。若Pool热请求只有本地cache命中，先确认MemCache容量和placement，不把cache命中率简单等同Pool hit rate。

回滚为 SGLANG_CAM_DETOUR_MODE=off，并按先停止模型后停止sidecar的顺序结束本次实验。恢复原生服务前确认没有旧CAM进程。不要删除或清空共享缓存池；pool_roundtrip只使用带UUID的探测key，清理也仅限这个key。

## 16 来源和验证记录

架构与命令来自已检查的 SGLang 源码和随仓文档，而非通用 CUDA 教程。主要依据如下。

- https://github.com/sgl-project/sglang/blob/f620d73/python/sglang/srt/models/qwen3_moe.py
- https://github.com/sgl-project/sglang/blob/f620d73/python/sglang/srt/distributed/parallel_state.py
- https://github.com/sgl-project/sglang/blob/f620d73/docs/docs/advanced_features/pd_disaggregation.mdx
- https://github.com/sgl-project/sglang/blob/f620d73/python/sglang/srt/mem_cache/storage/npu_memcache/README.md
- https://github.com/sgl-project/sglang/blob/f620d73/docs/docs/hardware-platforms/ascend-npus/optimization/profiling.mdx
- https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
- [Docker host network](https://docs.docker.com/engine/network/drivers/host/) 和 [容器运行参数](https://docs.docker.com/reference/cli/docker/container/run/)
- [PyTorch torchrun 环境变量与启动说明](https://docs.pytorch.org/docs/stable/elastic/run.html) 和 [Docker commit 限制](https://docs.docker.com/reference/cli/docker/container/commit/)
- [Docker 容器清单](https://docs.docker.com/reference/cli/docker/container/ls/)、[指定字段检查](https://docs.docker.com/reference/cli/docker/container/inspect/) 和 [running 容器内诊断](https://docs.docker.com/reference/cli/docker/container/exec/)
- [Ascend PyTorch 版本与安装说明](https://github.com/Ascend/pytorch)
- [ModelScope 官方下载说明](https://github.com/modelscope/ms-cookbook/blob/main/content/chapters/chapter-02.md)
- [A2 npu-smi 拓扑查询](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A2/A2npu/npusmi_0072.html) 和 [HCCN 网络健康查询](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A2/A2HCCN/hccntoola2_021.html)
- 本地UMDK bf74c6db的All-to-All Detour kernel、CamComm和socket exchange实现
- [MemCache安装/API版本说明](https://raw.githubusercontent.com/Ascend/memcache/master/doc/install_whl.md)：1.0.x与1.1.x配置API不同
- [MemCache 1.1.0包元数据](https://pypi.org/pypi/memcache-hybrid/1.1.0/json) 和 [MemFabric 1.1.0包元数据](https://pypi.org/pypi/memfabric-hybrid/1.1.0/json)：Python3.11/aarch64 wheel、依赖与SHA256
- 当前 python/pyproject_npu.toml：实验源码要求Transformers5.19.0，不能直接沿用基础镜像5.12.1

开发端已执行9项CPU单元测试（含私有配置生成与防覆盖）、修改Python文件语法检查、shell及README命令语法检查和git diff检查。原有容器采集脚本已用mock Docker检查只读采集行为。本次新增Dockerfile、安装和启动步骤未在目标A2执行：镜像构建、CANN/旧kernel与当前源码的native兼容性、CAM设备正确性、EP8、PD、零容量客户端和Pool命中均需逐节验收。已核对候选MemCache/MemFabric 1.1.0的wheel元数据、导出API和SHA256，但不能据此宣称CANN9/A2实测通过；任何性能结论都等待目标机器实测。
