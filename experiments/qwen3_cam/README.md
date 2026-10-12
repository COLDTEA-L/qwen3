# Qwen3-30B-A3B：A2 两机部署与流量冲突验证

更新：2026-10-12。A=尾号228，B=尾号238；两机各8张910B4、32GB HBM，机内FullMesh/HCCS，个人工作目录均为 /ssd_2。

执行顺序：拉代码 → 私有配置 → 构建/复制镜像 → 每机创建一个实验容器 → 下载权重 → 原生PD → CAM正确性 → KV Pool → 三模式对照与冲突测量。每个命令块逐行执行；不用输入续行反斜杠。

当前交付为源码、配置和工具。目标A2上的构建、通信正确性、模型质量和性能尚未实测；必须通过各步验收再往后走。

## 1 实验布局与版本

| 主机 | 物理卡 | 角色 | 并行设置 |
| --- | --- | --- | --- |
| A | 0–3 | P4 | Attention TP4、EP4 |
| A | 4–7 | D8前半 | D全局启动TP8、Attention DP2×TP4、EP8 |
| B | 0–3 | D8后半 | 与A4–7组成同一D组；专家内部TP1 |
| B | 4–7 | KV Pool4 / CAM中继 | 不加载模型，每卡初始8GB HBM Pool |

P与D的MoE域独立。CAM按本机物理卡号组成8-rank内存资源域：A由P4+D4提供窗口，B由D4+Pool4提供窗口。某次Attention通信仍只有4个参与rank，其余卡提供中继。**容器必须看到全部8卡，不隐藏、不重编号**；模型用base-gpu-id选择运行卡。

- SGLang：基于 f620d73，分支 feat/qwen3-cam-detour-conflict，仓库 https://github.com/COLDTEA-L/qwen3。
- UMDK：基于 all_gather_4_scheduler / bf74c6db，配套分支 feat/qwen3-cam-detour-build，预期HEAD为9c0bbdadfcfe882426cabdce5cdf2d5d2fce92ef。
- 模型：原版 Qwen/Qwen3-30B-A3B BF16；48层、128专家、top-k=8、4 KV heads、head_dim=128。权重仅在A2下载。

两机代码、镜像、模型校验值必须一致，不临时切最新main。

## 2 已确认的软件栈

| 内容 | 已确认值 / 本次处理 |
| --- | --- |
| 宿主机 | aarch64、EulerOS2.0 SP10、driver25.5.2 |
| 主机控制网 | 两机均enp67s0f5；不是NPU RoCE接口/IP |
| B的基础镜像ImageID | sha256:39c97f336132e02b9e7066428ddde439e763623e4024c1f405049f79b5be4ad6 |
| 镜像内环境 | Ubuntu22.04.5、Python3.11.15、CANN9.0.0 |
| 原镜像包 | Torch2.10.0+cpu、torch_npu2.10.0、sgl_kernel_npu2026.6.1、DeepEP1.0.0+48942e37.cann.9.0.0.b250 |
| 原镜像通信/模型包 | MemFabric1.0.8、Transformers5.12.1、SGLang0.5.17.dev386+gc5bd3d7dc；缺MemCache |
| 新实验镜像 | 保留上述Torch/kernel/DeepEP；源码依赖Transformers5.19.0；候选MemCache1.1.0+MemFabric1.1.0 |

`torch+cpu`与`torch_npu`是观测到的组合，不另装CUDA Torch。MemCache1.0.x不提供本实验使用的LocalConfig/MetaConfig接口；固定1.1.0 wheel/SHA256不等于已验证CANN9/A2兼容性。因此先在最终新镜像验证原生PD，再加CAM和Pool。两个构建层中，core保留MemFabric1.0.8；pool为最终实验镜像。**主流程只运行pool镜像，每机一个容器**，避免两套服务争用卡。版本依据见第15节。

## 3 对照模式与实验约束

| 模式 | Attention TP通信 | MoE / PD / Pool |
| --- | --- | --- |
| off | 原生集合通信 | 保留原生后端 |
| hccl-a2a | 按4份交换、FP32归约；HCCL All-to-All | 同上 |
| cam-detour | 相同分块/归约；CAM直连+中继交换 | 同上 |

AllReduce的最后AllGather仍是HCCL；ReduceScatter不做这个AllGather。CAM exchange不使用HCCL All-to-All。不能把实验组相对off的全部变慢都叫拥塞，hccl-a2a用于控制额外分块/launch/归约开销。

初始关闭图模式、compile、TBO、单批重叠和EPLB；每个CAM owner预留512MiB窗口。BF16归约顺序变化需容差和模型质量检查。当前未实现统一Fabric控制器、生产图模式或D直接从Pool恢复请求。

## 4 准备代码、镜像、容器与权重

除非写“容器内”，均在宿主机执行。代码/权重/镜像准备不加载模型；从第5节起须有获准使用的整套16卡窗口。不要停止他人任务。

### 4.1 两台宿主机拉代码

先确保开发服务器的本次改动已按第13节push。A、B各执行：

```bash
test -w /ssd_2
git clone --branch feat/qwen3-cam-detour-conflict --single-branch https://github.com/COLDTEA-L/qwen3.git /ssd_2/sglang
git -C /ssd_2/sglang rev-parse HEAD
```

已有目录时不再clone；确认工作区干净、origin为自己的qwen3仓库，再更新：

```bash
cd /ssd_2/sglang
git status --short --branch
git remote -v
git switch feat/qwen3-cam-detour-conflict
git pull --ff-only
```

有改动、remote不符或ff-only失败就停下，不reset。

UMDK是另一仓库。无卡开发服务器执行：

```bash
cd /home/liuyuanwen/umdk
git status --short --branch
git switch feat/qwen3-cam-detour-build
git bundle create /home/liuyuanwen/umdk-qwen3-cam.bundle feat/qwen3-cam-detour-build
```

A、B分别复制并clone（开发服务器账户不是root时改为实际账户）：

```bash
scp root@8.146.225.22:/home/liuyuanwen/umdk-qwen3-cam.bundle /ssd_2/umdk-qwen3-cam.bundle
git clone --branch feat/qwen3-cam-detour-build /ssd_2/umdk-qwen3-cam.bundle /ssd_2/umdk
git -C /ssd_2/umdk rev-parse HEAD
```

已有UMDK目录则先检查分支/改动，再从bundle fetch与ff-only合并，不覆盖用户目录。

### 4.2 生成并复制私有配置

B宿主机执行；交互输入A、B的已确认主机地址及SSH账户yuanwenliu：

```bash
cd /ssd_2/sglang
python3 experiments/qwen3_cam/configure_runtime.py
source experiments/qwen3_cam/runtime-config/deployment.env
ssh "${A2_SSH_USER}@${A_IP}" 'mkdir -p /ssd_2/sglang/experiments/qwen3_cam/runtime-config && chmod 700 /ssd_2/sglang/experiments/qwen3_cam/runtime-config'
scp experiments/qwen3_cam/runtime-config/deployment.env experiments/qwen3_cam/runtime-config/meta.json experiments/qwen3_cam/runtime-config/pool.json experiments/qwen3_cam/runtime-config/client.json "${A2_SSH_USER}@${A_IP}:/ssd_2/sglang/experiments/qwen3_cam/runtime-config/"
```

文件已存在时先核对，不覆盖旧私有配置。生成器拒绝覆盖B已有文件；向A复制前也先确认A这四个文件不存在或确需同步。runtime-config被Git忽略：

- deployment.env：两机地址、接口、模型路径。
- meta.json：B的Pool元数据服务。
- pool.json：B4–7正容量HBM Pool，每卡8GB、DRAM=0。
- client.json：相同域、HBM/DRAM均为0；不能让P/D成为额外存储池。

MemCache配置中的world_size=32是允许接入的最大服务rank数，不是本实验的模型卡数；不要按Pool4改成4。

### 4.3 B构建镜像

不修改旧yuanwen_sglang_a2、不使用opensora:latest。以下标签若已存在，先核对；已是本实验合格镜像可跳过构建，否则用新标签并同步修改后续引用，不覆盖他人标签。

```bash
df -h /ssd_1/docker /ssd_2
docker image inspect sha256:39c97f336132e02b9e7066428ddde439e763623e4024c1f405049f79b5be4ad6 --format '{{.Id}} {{.Architecture}}'
docker tag sha256:39c97f336132e02b9e7066428ddde439e763623e4024c1f405049f79b5be4ad6 qwen3-cam:a2-base-observed
cd /ssd_2/sglang
cp python/pyproject_npu.toml experiments/qwen3_cam/docker/source-pyproject_npu.toml
docker build --progress=plain -f experiments/qwen3_cam/docker/Dockerfile.a2-core -t qwen3-cam:a2-core-v1 experiments/qwen3_cam/docker
docker build --progress=plain -f experiments/qwen3_cam/docker/Dockerfile.a2-pool --build-arg A2_CORE_IMAGE=qwen3-cam:a2-core-v1 -t qwen3-cam:a2-pool-v1 experiments/qwen3_cam/docker
```

core安装编译工具、源码NPU依赖、Router0.3.2；pool固定安装MemCache/MemFabric1.1.0。要求当前Python3.11的Python.h。依赖解算或headers失败，保留错误，不解除Torch/kernel约束或换A3/950 wheel。

B此前Docker盘剩166GB，构建前以实际df为准；不要docker system prune或删除他人镜像。验证最终包版本：

```bash
docker run --rm --network=none --runtime=runc --entrypoint python qwen3-cam:a2-pool-v1 -m pip show torch torch-npu sgl-kernel-npu deep-ep memfabric-hybrid memcache-hybrid transformers | awk '/^(Name|Version):/ {print}'
```

预期见第2节；此条无NPU查询用runc。后面的模型容器使用ascend。

### 4.4 复制镜像，在228和238各创建一个容器

B执行：

```bash
docker save -o /ssd_2/qwen3-cam-a2-v1.tar qwen3-cam:a2-pool-v1
sha256sum /ssd_2/qwen3-cam-a2-v1.tar
source /ssd_2/sglang/experiments/qwen3_cam/runtime-config/deployment.env
scp /ssd_2/qwen3-cam-a2-v1.tar "${A2_SSH_USER}@${A_IP}:/ssd_2/"
```

A核对tar校验值与B相同，再导入：

```bash
sha256sum /ssd_2/qwen3-cam-a2-v1.tar
df -h /ssd_1/docker
docker load -i /ssd_2/qwen3-cam-a2-v1.tar
```

**两机各执行以下命令**；新的容器名为qwen3-cam-a2-v2，不替换旧容器：

```bash
docker info --format '{{json .Runtimes}}'
docker image inspect qwen3-cam:a2-pool-v1 --format '{{.Id}} {{.Architecture}}'
mkdir -p /ssd_2/models/Qwen3-30B-A3B /ssd_2/qwen3-download-venv
bash /ssd_2/sglang/experiments/qwen3_cam/docker/create_container.sh qwen3-cam:a2-pool-v1 qwen3-cam-a2-v2
docker ps --filter name=qwen3-cam-a2-v2 --format '{{.Names}} {{.Image}} {{.Status}}'
```

两机ImageID应一致；Runtimes中必须有ascend，缺失时请管理员处理，脚本不会静默切换runtime。脚本按你提供的OpenSora配置整理为：

| 保留/增加 | 本实验用途 |
| --- | --- |
| --runtime=ascend、--privileged、--network=host | 沿用已用过的A2启动方式；仅用于获准的实验环境 |
| --ipc=host、memlock=-1 | 保留CAM/共享内存与锁页配置 |
| davinci0–7、manager、devmm_svm、hisi_hdc | 八个物理设备及管理设备；不按P/D隐藏卡 |
| ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 | 明确Ascend Runtime挂载范围 |
| /usr/local/dcmi、driver、npu-smi只读 | 使用宿主驱动与管理库；npu-smi优先取/usr/local/bin |
| /var/log/npu→/usr/slog读写 | 沿用NPU日志挂载 |
| /ssd_2/sglang、umdk、模型目录、下载venv读写 | 仅挂载本实验工作目录，保持容器内外路径一致 |
| firmware只读 | 宿主存在该目录时保留 |

已删去OpenSora/ModelZoo目录以及整个/home、/data、/var挂载，也不挂整个/ssd_2。`--privileged`仍有很高宿主权限，收窄挂载不等于严格隔离；不要在容器内操作无关宿主资源。[Docker权限说明](https://docs.docker.com/engine/containers/run/#runtime-privilege-and-linux-capabilities)、[Ascend Runtime设备参数](https://www.hiascend.com/document/detail/en/mindcluster/2610/clustersched/schedulingug/docs/en/scheduling/usage/containerization/02_usage_on_the_docker_client.md)。

缺DCMI/日志/设备等路径时脚本停止，不自动创建系统目录。需要先查看实际docker run时，在创建前加第三个参数`--print-only`；该模式只检查并打印，不启动容器。已创建的旧容器无法通过docker exec改变runtime/mount，使用新名字，不自动删除旧容器。

容器先只运行sleep，不加载模型。每个新终端从宿主机进入：

```bash
docker exec -it qwen3-cam-a2-v2 /bin/bash
source /ssd_2/sglang/experiments/qwen3_cam/env.sh A
```

B的source改为B。env.sh初始化CANN/PYTHONPATH/接口/PD store，清除旧OpenSora的RANK等变量。**每个模型/测试终端都source一次；宿主机export不会自动进入docker exec。**

### 4.5 B下载权重，复制给A

进入B新容器，使用Python3.11独立下载环境；不加载模型：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python -m venv /ssd_2/qwen3-download-venv
/ssd_2/qwen3-download-venv/bin/python -m pip install modelscope
/ssd_2/qwen3-download-venv/bin/python -c 'from modelscope import snapshot_download; print(snapshot_download("Qwen/Qwen3-30B-A3B", local_dir="/ssd_2/models/Qwen3-30B-A3B"))'
mkdir -p /ssd_2/sglang/experiments/qwen3_cam/results
/ssd_2/qwen3-download-venv/bin/python -m pip freeze > /ssd_2/sglang/experiments/qwen3_cam/results/download-versions.txt
/ssd_2/qwen3-download-venv/bin/python -c 'import json; from pathlib import Path; p=Path("/ssd_2/models/Qwen3-30B-A3B"); c=json.loads((p/"config.json").read_text()); expected={"model_type":"qwen3_moe","num_hidden_layers":48,"num_experts":128,"num_experts_per_tok":8,"num_key_value_heads":4,"head_dim":128}; assert all(c.get(k)==v for k,v in expected.items()),c; i=json.loads((p/"model.safetensors.index.json").read_text()); shards=set(i["weight_map"].values()); assert all((p/n).is_file() and (p/n).stat().st_size>0 for n in shards); print("MODEL_FILES_PASS",len(shards))'
```

约60GB BF16权重，不选AWQ/GPTQ/FP8。已有完整权重可跳过下载但保留校验。exit返回B宿主机：

```bash
source /ssd_2/sglang/experiments/qwen3_cam/runtime-config/deployment.env
rsync -a --info=progress2 /ssd_2/models/Qwen3-30B-A3B/ "${A2_SSH_USER}@${A_IP}:/ssd_2/models/Qwen3-30B-A3B/"
```

两机宿主机各生成校验清单，内容应一致：

```bash
cd /ssd_2/models/Qwen3-30B-A3B
find . -type f \( -name '*.safetensors' -o -name '*.json' \) -print0 | sort -z | xargs -0 sha256sum > /ssd_2/qwen3-model-sha256.txt
```

只下载一次快照再复制，避免两机分别下载漂移的main。权重/私有配置/results/profiles不进入Git。

### 4.6 启动前只查本实验所需资源

两机宿主机执行；确认16卡使用窗口、控制网、指定端口：

```bash
npu-smi info
ip -br addr show enp67s0f5
ss -lnt | awk 'NR==1 || $4 ~ /:(5000|6000|7100|8100|8998|10067|29500|29610|29611|29612|29620|30000|30001|8000)$/'
```

A：7100 PD store，8998 P bootstrap，29500 D建组，30000/30001 P/D HTTP，8000 Router；B：5000/6000/8100 Pool meta/config/metrics。10067为同机CAM，296xx用于torchrun测试/sidecar。启动后属于本实验的监听是正常的，不需要全量listen清单。

AICore=0不代表卡已空闲，检查进程/HBM并与任务所有者确认；不杀他人进程。主机控制网成功不等于NPU RoCE通过，网络失败时才按第14节定向查询。

## 5 先跑原生PD（不开HiCache/Pool服务）

以下均在新容器内；A/B各先source对应env.sh：

```bash
python experiments/qwen3_cam/test_cpu.py
python -c 'import sglang; print(sglang.__file__); assert "/ssd_2/sglang/python/" in sglang.__file__'
python experiments/qwen3_cam/preflight.py --config experiments/qwen3_cam/configs/p_a.json --model-path "$MODEL_PATH" --mode off --require-pool --import-model
```

B的preflight用configs/d_b.json。要求CPU测试OK、MODEL_IMPORT_OK、DEVICES_OK 8、POOL_API_OK、PREFLIGHT_PASS。此步不加载权重即可发现当前源码与旧kernel API的导入问题；失败停止，不装其他型号wheel。池API通过仍不等于远端存取通过。

三个**独立前台终端**分别启动；每个终端先docker exec和source：

**A终端1：P**

```bash
export SGLANG_CAM_DETOUR_MODE=off
bash experiments/qwen3_cam/launch.sh p-a --context-length 4096 --max-running-requests 4 --chunked-prefill-size 1024
```

**A终端2：D主节点**

```bash
export SGLANG_CAM_DETOUR_MODE=off
bash experiments/qwen3_cam/launch.sh d-a --context-length 4096 --max-running-requests 4
```

**B终端1：D第二节点**

```bash
export SGLANG_CAM_DETOUR_MODE=off
bash experiments/qwen3_cam/launch.sh d-b --context-length 4096 --max-running-requests 4
```

D两端都启动才完成建组。SGLang自己起模型worker，**不套16进程torchrun**。待权重加载/EP/PD初始化成功，A终端3启动Router：

```bash
python -m sglang_router.launch_router --pd-disaggregation --prefill "http://${A_IP}:30000" --decode "http://${A_IP}:30001" --host 0.0.0.0 --port 8000
```

A终端4发送请求：

```bash
curl --fail-with-body http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d '{"text":"Explain tensor parallelism briefly.","sampling_params":{"max_new_tokens":32,"temperature":0}}'
```

验收：合理文本、P→D迁移成功；日志/trace确认D EP8=A4–7+B0–3、Attention两组同机TP4。OOM先排查其他进程/上下文/临时内存，不能悄悄退化EP1或假传输。最终环境已将MemFabric升级到1.1.0，必须在这个环境通过PD，不沿用旧镜像结论。

结束本步：在自己的终端Ctrl+C，先Router、再P/D，确认子进程释放卡。不要pkill python、stop原SGLang容器或停止别人的任务。

## 6 编译CAM并做单机8卡正确性测试

P/D/Pool全部停下。A/B分别在新容器中source env.sh后：

```bash
command -v msopgen
command -v cmake
command -v g++
bash experiments/qwen3_cam/install_cam.sh build
```

缺工具就停下；msopgen应来自当前CANN。脚本按ascend910b4和Torch CXX ABI构建，只安装本次生成的.run/wheel，发现多份输出会停止。它更新实验生成目录和新容器的CAM vendor，不删除原src/cam/comm_operator/build。

重新source正确的env.sh，再每机执行：

```bash
torchrun --nproc_per_node=8 --master_port=29610 experiments/qwen3_cam/smoke_collectives.py --mode hccl-a2a --iterations 100
torchrun --nproc_per_node=8 --master_port=29611 experiments/qwen3_cam/smoke_collectives.py --mode cam-detour --iterations 100
```

两次都要求8个rank打印SMOKE_PASS。覆盖FP16/BF16/FP32、不整除长度、跨chunk输入；超时、少rank、容差失败即停止，不测模型性能。

```bash
python experiments/qwen3_cam/preflight.py --config experiments/qwen3_cam/configs/p_a.json --model-path "$MODEL_PATH" --mode cam-detour --require-pool --import-model
```

B改configs/d_b.json；要求CAM_IMPORT_OK和PREFLIGHT_PASS。

## 7 验证真实KV Pool，再给P接入HiCache

### 7.1 独立远端put/get

模型全部停下，A卡0供probe独占。每个终端先source env.sh。

**B Meta终端：**

```bash
python -m sglang.srt.mem_cache.storage.npu_memcache.start_meta_service --config_path experiments/qwen3_cam/runtime-config/meta.json
```

**B Pool终端：**

```bash
torchrun --nproc_per_node=4 --master_port=29612 experiments/qwen3_cam/pool_sidecar.py --config experiments/qwen3_cam/configs/d_b.json --pool-config experiments/qwen3_cam/runtime-config/pool.json --mode off
```

要求B4、5、6、7各打印KV_POOL_READY和RELAY_READY。只有这四个服务有正HBM容量。

**A probe终端：**

```bash
python experiments/qwen3_cam/pool_roundtrip.py --client-config experiments/qwen3_cam/runtime-config/client.json --device 0 --mib 16 --iterations 20
curl --fail-with-body "http://${B_IP}:8100/metrics"
```

要求20次checksum PASS，B4–7容量/占用及Pool日志/metrics证明远端读写。仅checksum不能证明placement。若零容量客户端或DEVICE_RDMA初始化失败，停下保存错误，不改客户端为正容量后仍声称Pool4。

### 7.2 P接HiCache，验证远端恢复

保留B Meta和off sidecar。P的启动改为以下命令；D和Router仍用第5节：

```bash
export SGLANG_CAM_DETOUR_MODE=off
CLIENT_JSON=$(tr -d '\n' < experiments/qwen3_cam/runtime-config/client.json)
bash experiments/qwen3_cam/launch.sh p-a --context-length 4096 --max-running-requests 4 --chunked-prefill-size 1024 --enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-storage-backend npu_memcache --hicache-mem-layout page_first_kv_split --hicache-storage-backend-extra-config "$CLIENT_JSON"
```

先短请求，再两次相同约2K历史前缀。第一次查存储写入；第二次查远端Pool read/bytes或hit。仅cached_tokens或本地Radix命中不足以证明Pool恢复。可用不同前缀压力逐出本地缓存；仅在off模式下也可尝试重启P再读，但须证明L3键/域仍有效。

当前路径为Pool→host staging→P的模型KV→D，不是D直接Pool恢复。若只看到本地命中，先解决远端命中证据，不继续把它当KV Pool恢复实验。

## 8 多路径与三模式对照

每轮都在同一镜像、相同权重/并行/HiCache/负载下运行。切模式时先Router→P/D→sidecar停止；Meta可保持。某个CAM owner退出后不能单独重启复用旧IPC引用，先结束完整本机CAM域，再重建。

**B sidecar终端**先启动：

```bash
torchrun --nproc_per_node=4 --master_port=29612 experiments/qwen3_cam/pool_sidecar.py --config experiments/qwen3_cam/configs/d_b.json --pool-config experiments/qwen3_cam/runtime-config/pool.json --mode cam-detour
```

它可能等待B的D owner，不要求先独立完成CAM就绪。然后三个模型终端各source对应A/B，并设置：

```bash
export SGLANG_CAM_DETOUR_MODE=cam-detour
export SGLANG_CAM_TRACE_MOE=1
```

**A的P终端：**

```bash
export SGLANG_CAM_DETOUR_TRACE='experiments/qwen3_cam/results/cam-p-{rank}.jsonl'
CLIENT_JSON=$(tr -d '\n' < experiments/qwen3_cam/runtime-config/client.json)
bash experiments/qwen3_cam/launch.sh p-a --context-length 4096 --max-running-requests 4 --chunked-prefill-size 1024 --enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-storage-backend npu_memcache --hicache-mem-layout page_first_kv_split --hicache-storage-backend-extra-config "$CLIENT_JSON"
```

**A的D终端：**

```bash
export SGLANG_CAM_DETOUR_TRACE='experiments/qwen3_cam/results/cam-d-a-{rank}.jsonl'
bash experiments/qwen3_cam/launch.sh d-a --context-length 4096 --max-running-requests 4
```

**B的D终端：**

```bash
export SGLANG_CAM_DETOUR_TRACE='experiments/qwen3_cam/results/cam-d-b-{rank}.jsonl'
bash experiments/qwen3_cam/launch.sh d-b --context-length 4096 --max-running-requests 4
```

不要复制第5/7节的MODE=off。A的P4+D4、B的D4+Pool4全部owner启动后再启动Router，发送短请求，检查CAM_READY/CAM_CALL。JSONL是host提交日志，不是device执行时间。

off、hccl-a2a、cam-detour分别完整重启：P/D的MODE与sidecar的mode一致，真实Pool均在线。每轮先固定输入、temperature=0、seed检查token/logprob/代表性准确率，再压测；明显输出错误不进入性能结果。

## 9 综合serving负载

先把P/D上下文从4096提高到16384，并发从4逐步提高：可先P=8、D=16，再扫P=16、D=64；其他设置不变。32GB HBM同时要容纳权重、静态KV、临时张量、CAM512MiB，Pool卡另加8GB池。

背景请求持续decode，恢复burst使用共享长历史前缀。固定到达计划，先小负载，再扫输入4K/8K、并发8/16/32/64和缓存冷热。16K前缀加输出会超16384上下文，需扩大上下文或用15360前缀。理论完整BF16 KV每token96KiB，8K约768MiB；实际以传输字节为准。

在A的新容器另开客户端终端、source env.sh A后：

```bash
python experiments/qwen3_cam/serving_workload.py --model-path "$MODEL_PATH" --base-url "http://${A_IP}:8000" --duration 30 --background-rps 0.5 --burst-every 10 --burst-size 8 --prefix-tokens 8192 --background-output-tokens 512 --restore-output-tokens 128 --max-concurrency 64 --run-id cam-r1 --output experiments/qwen3_cam/results/serving-cam-r1.jsonl
```

每种模式同一参数，到达计划不变；run-id/输出文件改为本轮唯一名称，不能混写。记录TTFT、TPOT/SSE间隔、缓存信息、失败和client_queue_ms。客户端限流会改变实际负载；SSE批量返回token时，脚本间隔不是device逐token TPOT。热前缀仍须证明远端Pool读取。

## 10 找共享资源与真实重叠

| 组合 | 可能共享 | 必须证明 |
| --- | --- | --- |
| 多路径+MoE | D卡HCCS边、HBM/DMA | 原生MoE实际使用该资源且device区间重叠 |
| MoE+跨机KV | NIC/跨机交换机资源 | 实际路由、端口计数器与时间重叠 |
| 多路径+Pool读取 | 中继Pool卡HBM/DMA或真实同机搬运边 | Pool真实路径，不虚构KV绕行 |
| 三类共同影响 | 上述资源集合 | 联合时间线、独立/并发对照 |

例如绕行B0→B4→B1访问B4 HBM；若KV从B4自己的NIC出机，它不经过B4→B1，只可能共享B4的HBM/DMA。三类不一定同时堵同一条物理边。关闭TBO时，同一batch的Attention/MoE通常有依赖，不能预设两者并发。

在P/D分别启动profile，不依赖Router广播：

```bash
curl -X POST "http://${A_IP}:30000/start_profile" -H 'Content-Type: application/json' -d '{"output_dir":"/ssd_2/sglang/experiments/qwen3_cam/profiles/p","start_step":3,"num_steps":30,"activities":["CPU","GPU"]}'
curl -X POST "http://${A_IP}:30001/start_profile" -H 'Content-Type: application/json' -d '{"output_dir":"/ssd_2/sglang/experiments/qwen3_cam/profiles/d","start_step":3,"num_steps":30,"activities":["CPU","GPU"]}'
```

这里GPU是通用设备活动名称，NPU后端使用对应profiler。发送workload后检查所有worker的device task开始/结束、CAM_DETOUR、NATIVE_MOE、HCCL与KV事件。CPU范围不能代替device重叠证据；跨机时钟需校准。缺RDMA事件时补MemFabric完成指标或当前CANN支持的系统采集。

记录实际工具支持的收发字节、busy/queue/拥塞或重传、带宽、HBM容量/带宽。HBM容量不等于带宽；共享资源+重叠仍只是可能冲突，必须与独立/串行对照比较。

## 11 可选：16卡受控回放

只有真实serving未观察到需要的重叠时再做。先停全部模型/sidecar，脚本占用16卡，不需要权重。MoE是两次HCCL EP8 All-to-All代理，KV是HCCL P2P代理，不是DeepEP实际token路由或MemFabric Pool，因此不能当成真实服务结果。

A/B各在新容器source正确env.sh，然后同时启动：

```bash
# A
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr="$A_IP" --master_port=29620 experiments/qwen3_cam/replay_conflict.py --mode cam-detour --warmup 20 --iterations 100 --output experiments/qwen3_cam/results/replay-cam-a --kv-target a
# B
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr="$A_IP" --master_port=29620 experiments/qwen3_cam/replay_conflict.py --mode cam-detour --warmup 20 --iterations 100 --output experiments/qwen3_cam/results/replay-cam-b --kv-target a
```

自动测单类、双类、三类七种组合；仍需profile证明device重叠。可扫tp-mib=1/8/32、moe-mib=1/8/32/128、kv-mib=16/64/256，kv-target a/b对应跨机/同机目标；再测hccl-a2a，同参数至少5轮，结果目录每轮独立。

把B结果目录复制到A同名路径（先核对不要覆盖旧结果），A汇总：

```bash
python experiments/qwen3_cam/analyze_replay.py experiments/qwen3_cam/results/replay-cam-a experiments/qwen3_cam/results/replay-cam-b --output experiments/qwen3_cam/results/replay-summary.json
```

脚本按rank/payload/模式算p50/p95/p99和slowdown，不平均掉最慢rank；缺同条件standalone时拒绝计算。置信区间需要从跨轮原始样本另算，不能把少量样本p99当可靠尾延迟。

## 12 判断是否冲突、影响多少

至少汇报通信p95 slowdown、TTFT/TPOT p95、tokens/s、KV完成时间、SLO miss rate和共享资源指标。

通信slowdown=同实现并发p95÷独立p95。先预定报告分档，例如<5%且在噪声内记未见显著退化、5–10%轻度、10–30%明显、>30%严重；这些是实验阈值，不是行业标准。至少5轮paired比较，置信区间含1则不做确定性退化结论。

同资源并发退化，错峰/串行后恢复，并有queue/busy回落，才支持链路冲突解释。只有HBM/DMA指标变化则归类内存/搬运竞争；没有device重叠则是依赖串行。不能把实现额外开销、计算竞争、网络拥塞混为一谈。

每条结论附版本/拓扑/路径、device时间线、独立/并发/干预数值和重复轮次。允许结果为“链路冲突”“HBM/DMA竞争但无同边冲突”“当前负载未发现冲突”，不填写假想百分比。

## 13 发布与更新

无卡开发服务器执行，不是在A2的/ssd_2：

```bash
cd /home/liuyuanwen/sglang
git status --short --branch
git remote get-url publish
env -u GIT_ASKPASS -u SSH_ASKPASS GIT_TERMINAL_PROMPT=1 git -c core.askPass= -c credential.helper= push -u publish feat/qwen3-cam-detour-conflict
```

publish应为https://github.com/COLDTEA-L/qwen3.git。用户名COLDTEA-L，密码位置填PAT，不将token写进URL/文档。推送后两台A2按第4.1节ff-only更新。UMDK尚未发布，使用bundle；不需要先新建另一个GitHub仓库。

## 14 失败定位与安全停止

| 问题 | 先查 |
| --- | --- |
| ascend runtime缺失、挂载路径缺失 | docker info与宿主路径；请管理员处理，不改Docker全局配置、不切旧opensora镜像 |
| 源码导入失败 | 是否用/ssd_2/sglang/python、CANN环境、新源码要求的kernel API |
| EP/PD/Pool网络失败 | 当前后端DEVICE_RDMA、接口/端口与各NPU网络健康 |
| CAM等待 | 全部8 owner、10067、comm_id7、物理rank映射/IPC权限 |
| 有日志无CAM_CALL | MODE/配置及融合是否绕过接入点 |
| Pool只有本地hit | 存储写入、远端read/placement、键与域；不把Radix hit当Pool hit |

网络问题时才在宿主机定向查（工具不在PATH时用driver/tools中的实际路径）：

```bash
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i "$i" -ip -g; done
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i "$i" -net_health -g; done
```

停止仅限自己的实验终端：Router→P/D→sidecar→Meta。切回off前确认旧CAM域退出；不pkill python、不docker prune、不清共享Pool。probe只删除自己的UUID key。未授权时不删除或替换旧容器。

## 15 依据与验证边界

- 你提供的A2/OpenSora docker run：ascend、privileged、设备、DCMI、驱动与日志挂载；不采用OpenSora应用挂载。
- [Ascend Docker Runtime](https://www.hiascend.com/document/detail/en/mindcluster/2610/clustersched/schedulingug/docs/en/scheduling/usage/containerization/02_usage_on_the_docker_client.md)、[Docker运行参数](https://docs.docker.com/reference/cli/docker/container/run/)。
- [SGLang Qwen3](https://github.com/sgl-project/sglang/blob/f620d73/python/sglang/srt/models/qwen3_moe.py)、[PD](https://github.com/sgl-project/sglang/blob/f620d73/docs/docs/advanced_features/pd_disaggregation.mdx)、[NPU MemCache](https://github.com/sgl-project/sglang/blob/f620d73/python/sglang/srt/mem_cache/storage/npu_memcache/README.md)及本仓库NPU pyproject。
- [MemCache API版本说明](https://raw.githubusercontent.com/Ascend/memcache/master/doc/install_whl.md)、[1.1.0包元数据](https://pypi.org/pypi/memcache-hybrid/1.1.0/json)、[MemFabric1.1.0元数据](https://pypi.org/pypi/memfabric-hybrid/1.1.0/json)；wheel和SHA256见docker/requirements-pool.txt。

本次通过11项CPU测试、39段Bash命令及Python/shell语法检查。容器脚本测试覆盖ascend/八卡/IPC参数构造及挂载范围；这些不等于目标实机验收。本次未在228/238实际创建容器，也未构建或运行A2模型。
