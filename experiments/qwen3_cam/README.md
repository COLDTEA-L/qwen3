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

## 4 第一步准备两机软件和权重

在两机记录以下信息，先不要启动实验：

```bash
npu-smi info
python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print(torch.npu.device_count())'
python -m pip show sglang sgl-kernel-npu memfabric-hybrid memcache-hybrid
git -C /home/liuyuanwen/sglang rev-parse HEAD
git -C /home/liuyuanwen/umdk rev-parse HEAD
```

核对实际 A2 型号、每卡 HBM、同机 HCCS 或其他互联、跨机 RoCE 和每卡 NIC 的连通性。8 卡不是天然保证 fullmesh。CAM 的 GetShareMemOffset 按物理 rank 寻址，所需 peer IPC 无法访问时必须停止，不要跳过报错继续测。拓扑、驱动和 RoCE 配置由目标 A2 环境确认。

本次源码是 2026 年 10 月拉取的 main 快照，不是官方教程验证过的 v0.5.16。优先使用与此快照匹配的 Ascend SGLang 软件栈；docker/npu.Dockerfile 是依赖核对依据，不要将其中 A3 或 950 的 kernel wheel 直接用于 A2。若现有 A2 只支持较早 CANN，先完成版本适配再继续，不能只 pip install 最新版本混装。当前主线 Dockerfile 使用的 Torch2.10、TorchNPU2.10、MemFabric1.2.1 等仅供对照，不构成这次 A2 已验证兼容性的承诺。

两台目标服务器均保留本地 BF16 权重目录，例如 /models/Qwen3-30B-A3B。如使用 ModelScope CLI，可在目标环境按其已有安装方式下载：

```bash
modelscope download --model Qwen/Qwen3-30B-A3B \
  --local_dir /models/Qwen3-30B-A3B
```

选择的是原版，不与 Instruct2507、量化版或 Qwen3Next 混用。两机权重、tokenizer 和 config 必须一致。BF16 权重约 60GB，仅为磁盘预留参考；部署另需 KV、临时张量、通信窗口和服务内存。不要在无卡开发机下载。

## 5 第二步安装实验分支和 CAM

在目标两机拉取你的 SGLang Fork，切到实验分支。若已经有完整源码，先检查工作区，不要覆盖其他修改。

```bash
cd /home/liuyuanwen/sglang
git switch feat/qwen3-cam-detour-conflict
export PYTHONPATH=/home/liuyuanwen/sglang/python${PYTHONPATH:+:$PYTHONPATH}
python experiments/qwen3_cam/test_cpu.py
```

可以在已有匹配环境里用 PYTHONPATH 指向源码，避免此时重新解析依赖。如果要 editable install，先核对 python/pyproject_npu.toml 和既有 NPU 包的版本，不用普通 CUDA 依赖替换 NPU 环境。确认 python 导入的是实验目录，不是镜像自带的另一份 SGLang。

UMDK 的构建分支保留原 All-to-All Detour 实现，仅增加独立构建目录和 CXX ABI 匹配的脚本。在专用实验容器中执行，CANN 环境路径以实际安装为准：

```bash
cd /home/liuyuanwen/umdk
git switch feat/qwen3-cam-detour-build
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_PYTHON_EXECUTABLE=$(command -v python)
export SOC_VERSION=ascend910b4
# 上一行只是示例 必须替换为 npu-smi 确认的真实 SoC
bash build/cam/comm_operator/build_qwen3_all2all_detour.sh "$SOC_VERSION"
ls output/cam/qwen3-all2all/wheel
# 用上一行显示的确切 wheel 文件名安装 不使用其他分支留下的 wheel
python -m pip install --no-deps /确切路径/umdk_cam_op_lib版本.whl
```

脚本会在该容器中安装生成的 CAM 自定义算子包，可能更新既有 CAM vendor 包，因此必须使用独立实验环境并保留原镜像或安装包。它不删除原来的 src/cam/comm_operator/build。编译参数与当前 PyTorch 的 CXX ABI 匹配。两机分别构建或使用完全匹配的同一架构 wheel，不能跨 CANN、TorchNPU 或 CPU 架构复用。

## 6 第三步只跑算子正确性

这一步占用整机 8 张卡，不同时运行模型或 Pool。两台机器分别执行。停止仅限自己启动的实验进程，不使用 pkill python 一类广泛命令。

```bash
cd /home/liuyuanwen/sglang
export PYTHONPATH=$PWD/python${PYTHONPATH:+:$PYTHONPATH}
unset CAM_COMM_ID
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0
torchrun --nproc_per_node=8 --master_port=29610 \
  experiments/qwen3_cam/smoke_collectives.py --mode hccl-a2a
torchrun --nproc_per_node=8 --master_port=29611 \
  experiments/qwen3_cam/smoke_collectives.py --mode cam-detour
```

每次要求 8 个 rank 都打印 SMOKE_PASS；测试包含 FP16、BF16、FP32，非 4 整除的 AllReduce 输入以及跨 chunk 边界的大输入。至少额外重复 100 次小输入确认 magic 和共享缓冲复用稳定。无输出、flag 等待超时、少一个 rank PASS 或输出不一致都属于失败，先定位算子，不进行模型压测。

测试容差初始 rtol0.03 atol0.003，仅用于随机小幅值张量的冒烟检查，不是模型质量合格标准。模型阶段还要比较原生模式与新实现的相同输入 token、logprob 和代表性准确率。

## 7 第四步先部署原生 PD

先使用 off 模式验证两机模型、EP8 和真实 P 到 D 的传输。各终端都设置下列变量，替换真实地址。HCCL_SOCKET_IFNAME 和 GLOO_SOCKET_IFNAME 是主机控制网络接口，不代表已经配置好了各 NPU 的 RoCE 数据网络。

```bash
export A_IP=机器A的IP
export B_IP=机器B的IP
export MODEL_PATH=/models/Qwen3-30B-A3B
export HCCL_SOCKET_IFNAME=实际控制网络接口
export GLOO_SOCKET_IFNAME=实际控制网络接口
export ASCEND_MF_STORE_URL="tcp://${A_IP}:7100"
export ASCEND_MF_TRANSFER_PROTOCOL=device_rdma
export SGLANG_CAM_DETOUR_MODE=off
cd /home/liuyuanwen/sglang
```

Ascend PD 使用 MemFabric TransferEngine，在源和目标注册 KV 缓冲，store URL 在所有 worker 中一致。7100 为 PD 传输的独立协调端口；它不是后面的 MemCache 6000。检查端口未被占用并允许互通。已安装 MemFabric 必须支持 DEVICE_RDMA，否则 preflight 应失败，不接受自动回退 SDMA 后继续测跨机 RDMA。

分别在三个终端启动：

```bash
# A 机终端一
bash experiments/qwen3_cam/launch.sh p-a
# A 机终端二
bash experiments/qwen3_cam/launch.sh d-a
# B 机终端一
bash experiments/qwen3_cam/launch.sh d-b
```

随后在能访问 A 的客户端启动已经安装的 SGLang Router：

```bash
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill "http://${A_IP}:30000" --decode "http://${A_IP}:30001" \
  --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/generate -H 'Content-Type: application/json' \
  -d '{"text":"Explain tensor parallelism briefly.","sampling_params":{"max_new_tokens":32,"temperature":0}}'
```

先确认 HTTP200 和合理输出，再从日志及设备 trace 确认 D 的 EP8 横跨 A4至7 和 B0至3、Attention 是两组同机 TP4、实际 KV 传输完成。P 和 D 的 Attention TP 均为 4，避免把启动 TP8 当成 KV heads 必須再分成 8 份。Ascend PD 内存注册和目标映射仍以原生后端实际校验为准。

launch.sh 是原生参数和角色布局的实验启动模板，不保证在任意 CANN 镜像零修改启动。若 DeepEP NPU normal 模式或 PD 不支持该布局，先停止于原生模式的失败点并保存日志；不要退化成 EP1 或 fake 传输后宣称已经验证本场景。

## 8 第五步部署真实 KV Pool

Pool 使用 MemCache，不是只有常驻 CAM 内存的 sidecar。首先在 B 上为本实验创建独立 MemCache 元数据域，避免影响其他用户。通过编辑器写入 runtime-config/meta.json：

```json
{
  "meta_service_url": "tcp://机器B的IP:5000",
  "config_store_url": "tcp://机器B的IP:6000",
  "metrics_url": "http://机器B的IP:8100",
  "log_level": "info"
}
```

创建 runtime-config/pool.json，初始每卡 8GB Pool，DRAM 不作为存储池。将真实地址写入 JSON，而不是保留中文占位符。

```json
{
  "meta_service_url": "tcp://机器B的IP:5000",
  "config_store_url": "tcp://机器B的IP:6000",
  "world_size": 32,
  "protocol": "device_rdma",
  "dram_size": 0,
  "hbm_size": "8GB",
  "max_hbm_size": "8GB",
  "log_level": "info"
}
```

客户端配置 runtime-config/client.json 使用相同地址、world_size、protocol，但 dram_size 和 hbm_size 都为 0。正容量服务只能是 B4至7 的四个 Pool sidecar；否则命中只证明命中了分布式缓存，不能证明数据放在预定四张卡上。

```bash
# B 机另开终端启动元数据服务
python -m sglang.srt.mem_cache.storage.npu_memcache.start_meta_service \
  --config_path experiments/qwen3_cam/runtime-config/meta.json
# B 机 Pool 终端 此时先不建立 CAM 域
torchrun --nproc_per_node=4 --master_port=29612 \
  experiments/qwen3_cam/pool_sidecar.py \
  --config experiments/qwen3_cam/configs/d_b.json \
  --pool-config experiments/qwen3_cam/runtime-config/pool.json --mode off
# A 机执行真实远端 put get 探测
python experiments/qwen3_cam/pool_roundtrip.py \
  --client-config experiments/qwen3_cam/runtime-config/client.json \
  --device 0 --mib 16 --iterations 20
```

要求四张 Pool 卡 HBM 用量增加、探测 checksum 全部 PASS，并从 MemCache metrics 或日志确认远端服务读写。零容量客户端是否能 init 及 NPU buffer 注册受 MemCache 版本约束；不支持时必须按该版本原生客户端配置完成适配，不能偷偷增加 P 或 D 容量后仍声称固定 Pool4。

服务级缓存接入优先只加在 P 上，使用已有 HiCache npu_memcache 后端。启动 P 时增加如下参数，并把 client.json 的完整 JSON 内容作为 extra-config 的字符串：

```bash
bash experiments/qwen3_cam/launch.sh p-a \
  --enable-hierarchical-cache --hicache-io-backend kernel_ascend \
  --hicache-storage-backend npu_memcache \
  --hicache-mem-layout page_first_kv_split \
  --hicache-storage-backend-extra-config '替换为client.json的完整JSON对象'
```

该参数行含显式待替换值，不能原样执行。HiCache 的 L3 读取通常先到 host staging，再进入本地模型 KV，所以 Pool 到模型的数据移动不一定是一条直接 HBM 到 HBM 读取。此阶段 P 写回和恢复前缀，P 再向 D 搬运结果。不要声称 D 已直接从 Pool 恢复请求；D 的直接 offload resume、缓存键一致性和 decode HiCache 需另行验证。

发两遍相同的长前缀确认第一次产生存储写入，第二次看到 pool read hit 或实际读取字节，同时 cached_tokens 或 cached_input_len 增加。仅本地 Radix 命中不足以证明 Pool 读取；可以采用本地缓存容量压力使其逐出，但不要通过停 P 来复用持有 P 窗口的 CAM 域。彻底重启 P 时，后面 CAM 模式必须连同 D 和 sidecar 一起重建。

## 9 第六步打开多路径接入

先停止这次实验启动的所有模型 worker 和 Pool sidecar，记录 PID，不动其他用户进程。不能只重启 P：D 的 CAM 地址仍引用旧 P 进程 IPC 窗口。启动新域时 A 的 P4 和 D4、B 的 D4 和 Pool4 必须全部在线，窗口保持到整个实验结束。

在角色对应的两机终端进行 preflight，然后设定模式。D 的两机必须同一模式，P 也采用一致模式，配置文件仍按角色区分。

```bash
export PYTHONPATH=/home/liuyuanwen/sglang/python${PYTHONPATH:+:$PYTHONPATH}
python experiments/qwen3_cam/preflight.py \
  --config experiments/qwen3_cam/configs/p_a.json \
  --model-path "$MODEL_PATH" --mode cam-detour
export SGLANG_CAM_DETOUR_MODE=cam-detour
export SGLANG_CAM_TRACE_MOE=1
export SGLANG_CAM_DETOUR_TRACE='/home/liuyuanwen/sglang/experiments/qwen3_cam/results/p-a-{rank}.jsonl'
```

其他角色使用 d_a.json 和 d_b.json；trace 文件名改成 d-a 或 d-b，以免 P 和 D 的逻辑 rank 同号覆盖。将正确输出目录设为 SGLANG_TORCH_PROFILER_DIR。trace JSONL 记录的是 host 提交时间，不能据此计算设备实际重叠。

并行启动四组进程。建议先在 B 启 Pool sidecar，它会等 B 的 D rank0；再同时启动 P、D 的两机角色。CAM 首次握手默认会等待其他物理 rank，不要把尚未启动的角色误判成程序死锁，但超过原生超时要检查 rank 数与端口。

```bash
# B 机 Pool 终端 其余环境同前
torchrun --nproc_per_node=4 --master_port=29612 \
  experiments/qwen3_cam/pool_sidecar.py \
  --config experiments/qwen3_cam/configs/d_b.json \
  --pool-config experiments/qwen3_cam/runtime-config/pool.json --mode cam-detour
# A 机 P 终端 可加前面的 HiCache 参数
bash experiments/qwen3_cam/launch.sh p-a
# A 机 D 终端
bash experiments/qwen3_cam/launch.sh d-a
# B 机 D 终端
bash experiments/qwen3_cam/launch.sh d-b
```

验收不能只看服务启动成功。每个模型 worker 应有 CAM_READY；非空实际推理应有 CAM_CALL；profiler 中要看到 all2_all_detour 对应 device kernel。若只有 CAM_READY 而没有 CAM_CALL，说明实际层边界可能用了未覆盖的融合或其他路径，本次接入未生效。不要测一个实际未调用多路径算子的模型。

分别用 off、hccl-a2a、cam-detour 模式跑相同的确定性样例。初始选择 100 个固定 prompt，temperature0，输出32或128 tokens，保存 token IDs 和 logprobs。若结果有差异，检查归约误差、专家路由和并行映射；大面积差异必须停止。再做一组代表性准确率验证，达到事先约定的容差后才跑性能。

## 10 第七步设计实际 serving 工作负载

不预先承诺三类流量一定同时阻塞同一条链路。单个 batch 中 Attention 通信、MoE dispatch、专家计算和 combine 有依赖。关闭 TBO 的初始版本尤其可能只与 KV 搬运重叠，而不与同一引擎的另一个算子重叠。

构造一个持续 decode 背景和间歇恢复 burst。背景请求输出 512 至 1024 tokens，保持 D 每 worker 活跃批次；burst 使用相同长前缀的 agent 恢复请求，每次8或16个请求。扫描输入4K、8K、16K，缓存冷读、热读两种条件，以及总并发8、16、32、64。为每次请求记录 rid、输入输出 token 数、cached_tokens、TTFT、逐 token 时间和端到端时间，固定 seed 和 arrival schedule。

完整 BF16 KV 的理论逻辑量为每 token 96KiB，4K前缀384MiB，8K前缀768MiB，16K前缀1.5GiB。每个 Attention TP4 shard 约为四分之一。实际传输还受缓存命中、页对齐、staging 和请求落在哪个 DP 组影响，以实际注册和搬运字节为准。

可先用随机请求验证冷缓存吞吐，随后用保存的相同前缀列表产生热缓存恢复。以下命令是冷缓存示例；它不能替代 Pool 命中实验。

```bash
python -m sglang.bench_serving --backend sglang \
  --base-url http://路由器IP:8000 --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" \
  --dataset-name random --random-input-len 8192 --random-output-len 512 \
  --random-range-ratio 1 --num-prompts 256 --max-concurrency 32 --seed 17
```

固定请求数不足以保证负载一致：若希望测相同压力，使用相同 arrival trace；若希望测最大吞吐，再单独扫并发。不要把两者混成一个结论。首轮不启 EPLB，固定专家映射，避免迁移专家权重或负载调整引入第四类流量；之后可独立增加 EPLB 场景。

可复制的综合负载由 serving_workload.py 提供。它在固定到达时间提交短输入长输出背景请求，在第10和20秒各提交8个共享8K历史前缀的恢复请求，记录每个 rid 的 TTFT、平均 TPOT、SSE 间隔、客户端排队、缓存信息和失败。每个模式使用同样参数，run-id 和输出文件名只作区分。

```bash
python experiments/qwen3_cam/serving_workload.py \
  --model-path "$MODEL_PATH" --base-url http://路由器IP:8000 \
  --duration 30 --background-rps 0.5 --burst-every 10 --burst-size 8 \
  --prefix-tokens 8192 --background-output-tokens 512 \
  --restore-output-tokens 128 --max-concurrency 64 \
  --run-id cam-r1 --output experiments/qwen3_cam/results/serving-cam-r1.jsonl
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
curl -X POST "http://${A_IP}:30000/start_profile" \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/home/liuyuanwen/sglang/experiments/qwen3_cam/profiles/p","start_step":3,"num_steps":30,"activities":["CPU","GPU"]}'
curl -X POST "http://${A_IP}:30001/start_profile" \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/home/liuyuanwen/sglang/experiments/qwen3_cam/profiles/d","start_step":3,"num_steps":30,"activities":["CPU","GPU"]}'
```

发送 workload 后检查所有 worker 的实际 trace。关注 CAM_DETOUR、NATIVE_MOE、HCCL task 和 KV transfer。CPU record_function 范围只是关联入口；真正的重叠从 device task 的开始结束时间计算。不同机器时钟要同步，跨机器不要直接拼接未校准的 timestamp。若普通 profiler 看不到 RDMA 或 SDMA 任务，使用目标 CANN 所支持的系统 interconnect 采集或 MemFabric completed transfer 指标补充，不能把缺失事件当成没有流量。

链路或端口计数器须从目标硬件工具读取，先用工具帮助确认当前驱动支持哪些字段。记录忙闲时间、收发字节、拥塞或重传计数、队列积压、带宽利用率，以及每卡 HBM 使用量和可用的 HBM 带宽指标。不要把 HBM 容量占用直接当成带宽利用率。只有时间重叠加共用资源，还只是可能冲突；需要与 standalone 或串行对照比较才判断影响。

## 12 第九步受控流量回放

实际模型中如果未观察到 MoE 和多路径同时执行，可用 replay_conflict.py 单独验证硬件资源竞争的可能性。先停止全体模型和 Pool，脚本占用整套16卡。它不需要权重。

回放中的 multipath 使用真实 CAM adapter；MoE 是两次 HCCL EP8 All-to-All 的通信代理，KV 是 Pool 到 D 的 HCCL P2P payload 代理。它没有专家计算、token 不均衡路由、真实缓存对象管理或 MemFabric 协议，所以结果只能解释链路与 HBM 等资源竞争，不能作为真实 MoE 或 KV Pool 的服务性能。

```bash
# A 机
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr="$A_IP" --master_port=29620 \
  experiments/qwen3_cam/replay_conflict.py --mode cam-detour \
  --output experiments/qwen3_cam/results/replay-cam-a --kv-target a
# B 机 同时启动
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
  --master_addr="$A_IP" --master_port=29620 \
  experiments/qwen3_cam/replay_conflict.py --mode cam-detour \
  --output experiments/qwen3_cam/results/replay-cam-b --kv-target a
```

脚本自动跑单类、三组双类及三类并发，共七种组合，固定相同 payload。任务使用独立线程和 NPU stream；每段记录 NPU Event 时间。还需 profiler 确认 stream 中真正的 device task 重叠，不以线程存在代替并发证据。把 warmup 增至20、iterations增至100，每轮结束后重复至少5轮，并将单类与并发运行顺序在不同轮次交替，减少热状态和顺序偏差。

扫描 tp-mib 为1、8、32，moe-mib为1、8、32、128，kv-mib为16、64、256。kv-target a 测跨机 Pool 到 A 的 D，kv-target b 测同机 Pool 到 B 的 D。再把 mode 改为 hccl-a2a，保持所有 payload 与线程配置相同。比较同一种实现自身 standalone 与并发的比值，再比较两个实现的退化幅度。

复制两机的 rank JSONL 到不同结果目录，避免同名覆盖，随后汇总：

```bash
python experiments/qwen3_cam/analyze_replay.py \
  experiments/qwen3_cam/results/replay-cam-a \
  experiments/qwen3_cam/results/replay-cam-b \
  --output experiments/qwen3_cam/results/replay-summary.json
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

## 14 GitHub 建仓和推送

不需要单独建立名为 Qwen3 的模型源码仓库。改动在 SGLang 中，推荐 Fork https://github.com/sgl-project/sglang 到你的账号，Fork 可命名为 qwen3-cam-serving。Fork 自带上游历史，适合当前 shallow clone；不要向一个空仓库直接推浅历史。

在 GitHub 页面点击 Fork 和 Create fork 后，在开发服务器执行。YOUR_GITHUB_USER 需要替换为你的实际用户名。该终端已在 root 下操作不影响归属，GitHub 身份由认证决定，使用 PAT 或 SSH，不用账号密码。

```bash
cd /home/liuyuanwen/sglang
git status --short --branch
git switch feat/qwen3-cam-detour-conflict
git remote add publish https://github.com/YOUR_GITHUB_USER/qwen3-cam-serving.git
git push -u publish feat/qwen3-cam-detour-conflict
```

如 publish 已存在，先 git remote get-url publish 核对，勿再次添加。保留 origin 指向原 SGLang 上游，本任务不替你发布到 GitHub。目标 A2 主机从该 Fork clone 并切分支即可，不上传权重、profile、运行日志或私有地址配置。

UMDK 配套构建改动在独立分支。可将服务器上的 UMDK 仓库推到你已有的 UMDK Fork：

```bash
cd /home/liuyuanwen/umdk
git switch feat/qwen3-cam-detour-build
git remote add publish https://github.com/YOUR_GITHUB_USER/你的UMDK仓库.git
git push -u publish feat/qwen3-cam-detour-build
```

若没有 UMDK Fork 而想新建空仓库，应先从该仓库真实 origin 拉齐历史，或按现有 UMDK 发布流程建立 Fork，再推送。无需为两个构建改动上传模型权重。运行配置 runtime-config 已忽略，给 A2 用户单独提供真实地址即可。

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
- 本地UMDK bf74c6db的All-to-All Detour kernel、CamComm和socket exchange实现

开发端已执行配置、分组、窗口容量、分块对齐、归约代数、分析脚本等CPU单元测试，以及修改Python文件的语法检查、shell语法检查和git diff检查。开发容器也未安装PyTorch，因此没有用CPU PyTorch进程组运行适配器。A2编译、CAM实际IPC、设备collective、EP8、PD和Pool命中以及性能测试均待目标环境执行，不能将CPU测试通过写成模型部署已通过。
