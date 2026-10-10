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

## 4 部署准备：宿主机、代码、Docker 和权重

本节更新于 2026-10-10。**网络接口和 FullMesh 拓扑已确认；当前只补简短硬件/版本信息，代码和权重可以先准备。**两台 **目标 A2 宿主机**的个人目录均为 /ssd_2，容器内也用同一路径。无卡开发服务器 8.146.225.22 的源码仍在 /home/liuyuanwen，不搬家、不下载权重。A_IP、B_IP 是两台 A2 的主机通信 IP，不是开发服务器 IP。

| 内容 | 两台 A2 的统一路径 |
| --- | --- |
| SGLang 实验源码 | /ssd_2/sglang |
| UMDK / CAM 实验源码 | /ssd_2/umdk |
| 原版 BF16 权重 | /ssd_2/models/Qwen3-30B-A3B |
| 隔离的下载工具环境 | /ssd_2/qwen3-download-venv |
| 私有配置、结果、profile | /ssd_2/sglang/experiments/qwen3_cam 下的 runtime-config、results、profiles |

运行数据已通过实验目录 .gitignore 排除，不提交私有地址、权重和采样文件。下文区分宿主机和容器内操作；宿主机中的 export 不会自动进入 docker exec 新终端。

### 4.0 已确认什么；现在还需要什么

用户已确认两机均为 **8 卡 FullMesh / HCCS**，并提供此前跨机 16 卡 OpenSora 的配置。该配置将两机控制网接口均设为 enp67s0f5，A 为主节点；本实验沿用这个控制网首选，不再要求重复提交全量接口、镜像或监听端口清单。历史配置不等于当前 SGLang、DeepEP、MemFabric 软件栈已经验收。

| 项目 | 当前结论 | 本次处理 |
| --- | --- | --- |
| 两机主机地址、个人目录 | 登录地址已确认；两机 /ssd_2 | 私有 hosts.env 保存地址，不写入 Git 跟踪文档 |
| 同机拓扑 | 两机均 8 卡 FullMesh / HCCS，用户确认 | 不再要求重复查 topo；第 6 节仍做 CAM peer IPC/正确性测试 |
| 控制网接口 | 两机 enp67s0f5，此前用于跨机 OpenSora | 沿用 HCCL/Gloo 接口；建连测试时再验证，不重复全量网络盘点 |
| B 的个人 SGLang 容器 | yuanwen_sglang_a2，镜像标注 cann9.0.0-910b-v0.5.16 | 优先查版本；以其原始镜像为候选，新建实验镜像和容器 |
| A 的旧 SGLang 容器 | 标签 v0.5.13.post1，与 B 不同 | 不再要求进入他人的 A 容器；最终两机复制同一合格实验镜像 |
| 芯片与每卡 HBM | 两机均 Ascend 910B4、每卡 32GB，用户确认 | 不再重复询问型号/标称容量；编译按本仓库 ascend910b4 参数准备 |
| CPU 架构、驱动和可用空间 | 尚未补齐 | 只回传 4.0.1 的简短查询 |
| 当前代码与候选栈兼容性 | 仍待确认 | 只回传 B 的 4.0.2；合格后在新容器复核 |
| 16 卡使用窗口 | 未确认 | 用户确认何时能独占两机各 8 卡；新容器隔离软件，不隔离 HCCS/HBM/NPU |
| 跨机 EP8、PD/Pool RDMA | 历史 OpenSora 不构成本版本的功能验收 | 启动前做第 5—8 节测试，失败再定向查询 HCCN/交换机，不盲改网络 |

**现在可以先拉代码和下载权重**，只需有目录权限、足够磁盘和下载网络；不依赖端口是否空闲，也不要求先完成 NPU 通信验证。建实验镜像与启动模型分别有自己的门槛，见 4.0.3。当前快照基于 f620d73，不能仅凭旧容器能跑别的模型就承诺当前分支可直接运行。

#### 4.0.1 只补两台宿主机的简短信息

A、B 宿主机各执行以下命令，输出标明 A/B。无需先 clone、source hosts.env 或执行第 1—3 节；命令缺失/权限不足时直接保留报错，不为了查询安装或修改系统：

```bash
hostname
uname -m
cat /etc/os-release
npu-smi info
cat /usr/local/Ascend/driver/version.info
df -h /ssd_2
df -h /dev/shm
docker info --format 'Architecture={{.Architecture}} DockerRootDir={{.DockerRootDir}}'
```

npu-smi 此时用于检查设备健康、当前 HBM/进程占用，不再要求用户重新确认 910B4/32GB。Docker 镜像通常存放在 DockerRootDir，还需在**构建/导入前**检查其所在磁盘空间，无需发送全部磁盘清单。

不用再发完整 ss 输出；仅在实际启动前检查实验选用的端口，见第 4.2 节。

#### 4.0.2 只补 B 的个人候选容器版本

B 宿主机执行。按用户说明选择其个人容器 yuanwen_sglang_a2；仅查询，**不重命名、停止、升级、安装实验代码或加载模型**。如果它不是 running，只提供 inspect 输出，不为查询启动原模型：

```bash
export CANDIDATE=yuanwen_sglang_a2
docker inspect "$CANDIDATE" \
  --format 'Name={{.Name}} Image={{.Config.Image}} ImageID={{.Image}} State={{.State.Status}} Network={{.HostConfig.NetworkMode}} IPC={{.HostConfig.IpcMode}}'
if [ "$(docker inspect "$CANDIDATE" --format '{{.State.Status}}')" = running ]; then
  docker exec "$CANDIDATE" env PYTHONDONTWRITEBYTECODE=1 python -c \
    'import torch, torch_npu; print("Torch", torch.__version__, "TorchNPU", torch_npu.__version__); print("NPU_AVAILABLE", torch.npu.is_available(), "NPU_COUNT", torch.npu.device_count())'
  docker exec "$CANDIDATE" python -m pip show \
    sglang torch torch-npu sgl-kernel-npu deep-ep deep-ep-npu memfabric-hybrid memcache-hybrid \
    | awk '/^(Name|Version):/ {print}'
  docker exec "$CANDIDATE" bash -c \
    'find /usr/local/Ascend -maxdepth 4 -type f \( -name version.cfg -o -name ascend_toolkit_install.info \) -print -exec cat {} \;'
else
  echo "候选不是running；不启动原业务，保留inspect输出"
fi
```

包缺失或 import 报错请保留；pip show 的 missing 警告也有意义，不先 pip install。若原服务用了 conda/venv，提供真实解释器或激活方法；默认 python 报错不证明容器环境坏了。以上导入可能有少量初始化开销，不分配模型张量。

只查询 B 是因为准备统一采用 B 的候选基础镜像，不是忽略 A 的驱动。**候选容器可写层与原始镜像可能不同**，查完还需在第 4.4 节的干净新容器复核。缺少 DeepEP/MemFabric 或当前源码不兼容时，需要先明确匹配版本/适配，不能直接混装最新版，更不能用一机 v0.5.13、一机 v0.5.16 混跑。

#### 4.0.3 实际部署顺序和验收门槛

| 顺序 | 操作与章节 | 何时可执行 / 通过标准 |
| --- | --- | --- |
| 1 | 两机拉 SGLang；开发端导出 UMDK bundle 后复制（4.3） | 目录可写、网络可达即可；不依赖 NPU 或端口 |
| 2 | 一机下载原版 BF16 权重，复制另一机并校验（4.6） | 空间足够即可；下载工具独立于推理环境 |
| 3 | 回传 4.0.1、4.0.2，并确认 16 卡窗口 | 确定 SoC、驱动与候选栈差异；无需重复拓扑/全量端口查询 |
| 4 | B 上新建工具镜像，复制 A，再各建一个独立容器（4.4—4.5） | 基础栈兼容性明确、两机架构一致；新容器版本/源码路径复核 |
| 5 | 原生依赖预检、CAM 编译/8 卡正确性（5—6） | 新环境 import/API 通过；每机 8 个物理 rank 的 CAM 可达且结果正确 |
| 6 | 原生 off 的 P4 + D8 + Pool4（7—8） | 模型输出、EP8、PD 迁移和 Pool 读写验证；不是只看启动日志 |
| 7 | 切 cam-detour、做 hccl-a2a 对照及冲突回放（9—13） | 真实 CAM_CALL、相同输入/软件/负载；验证时间重叠、资源共享和增量影响 |

先不做第 5—7 步的性能承诺。需要设备独占的测试要等维护窗口；拉代码、下载权重可以提前完成。

**“16 卡可用实验窗口”不是额外硬件参数**，而是确认实验期间两台机器的卡 0—7 没有其他训练/推理任务，并且不会临时加入其他任务。P/D/Pool 共占满 16 卡，CAM 绕行还会访问同机其余卡，因此不能只确认自己的模型卡空闲。

确认方法：与机器负责人/调度系统约定起止时间，明确两机各 8 卡可供本实验使用；开始前在两机运行 npu-smi info，核对没有无关计算进程、利用率回落、HBM 没有旧模型常驻。驱动/系统基础占用不必为零；有未知进程或大额 HBM 占用时先问归属，不杀进程。容器显示 Up 不等于正在占卡，也不等于卡空闲。监控快照只能证明当时情况，后续无其他任务需要预约/管理约定保证。可直接回复：“两机各 8 卡在某时段可独占，没有其他任务”；无需提供审批截图。

**32GB HBM 的部署预算：**Qwen3-30B-A3B BF16 约 60GB 的完整权重须常驻所有专家，不能按每 token 只激活约 3B 参数来估算容量。P4 平均权重约 15GB/卡只是粗估，实际还有复制权重、KV、激活、通信缓冲与默认 512MiB CAM 窗口；D8 的 Attention DP2 也包含复制内容，不能简单把所有内存除以 8。Pool 卡的初始 8GB 缓存另加 CAM/运行时开销。P4+D8+Pool4 暂不因此改布局，但要先小 batch、短上下文验证实际峰值，再逐步增加；launch.sh 的请求上限/上下文上限不是可同时占满的内存承诺，OOM 时保留日志并下调后统一重跑对照。

### 4.1 可选扩展诊断：出现问题再查，不是当前必做清单

当前只需第 4.0.1、4.0.2 和设备使用窗口。后续发现驱动/挂载/网络异常时再按对应问题补查；已知拓扑、接口和镜像清单不重复收集，不发送密码、PAT、SSH 私钥或完整环境变量。

代码已拉取后可用以下脚本做完整诊断；它会输出较多信息，因此现在不要求执行或复制全部输出：

```bash
# 可选，A2 宿主机；填另一台 A2 的主机通信 IP
export PEER_HOST_IP=另一台A2的主机IP
bash /ssd_2/sglang/experiments/qwen3_cam/collect_machine_info.sh
```

脚本只查询，不改网卡、不启停容器，每项查询最多等待 30 秒。每卡 HCCN/IP/link 查询见 4.2；只在跨机通信预检失败或需要确认设备网时使用。用户已确认同机 FullMesh，但具体可用带宽与 peer IPC 权限仍以实际 CAM 测试为准。

部署时在编辑器中创建两机私有 experiments/qwen3_cam/runtime-config/hosts.env（只含可信 export 行），或者复制开发端同名私有文件。**它被 Git 忽略，clone/pull 不会生成它，查询阶段不要 source 不存在的文件。**如下是公开模板，地址/账号必须替换：

```bash
export A_IP=实际A机地址
export B_IP=实际B机地址
export A2_SSH_USER=实际A2登录账号
export HOST_IF_A=enp67s0f5
export HOST_IF_B=enp67s0f5
export A2_CANDIDATE_IMAGE_B=quay.io/ascend/sglang:cann9.0.0-910b-v0.5.16
```

候选镜像变量不等于已合格的 A2_BASE_IMAGE。真实内网地址仅放私有文件，不 git add -f；IP 若变化，要手动更新两机私有配置。

#### 4.1.1 OpenSora 旧配置如何用于本实验

沿用的是控制网接口与两机地址，不照搬旧模型的进程布局。你提供的 B 示例中，第一条 node_rank=0 与后面的 node_rank=1 互相矛盾：固定两机作业应 A 为 0、B 为 1。

torchrun 会为各 worker 设置 RANK、LOCAL_RANK、WORLD_SIZE；**不要用 RANK=1 表示整个 B 节点，也不要手工把 LOCAL_RANK 都设为 0**。两机各 8 个 worker 的 16 rank 回放由 torchrun 自动分配这些变量。命令中显式使用 --master_addr/--master_port，不靠只 export MASTER_PORT 来推测最终 rendezvous 端口。[PyTorch 启动说明](https://docs.pytorch.org/docs/stable/elastic/run.html)

```bash
# 新实验终端；清理旧 OpenSora 手工 export 的进程变量
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export GLOO_SOCKET_IFNAME=enp67s0f5
export HCCL_SOCKET_IFNAME=enp67s0f5
```

TP_SOCKET_IFNAME 不是本实验已核实的 SGLang 接口选择项，不以它代替 HCCL/Gloo 设置。历史端口 12355 不强制沿用，本实验各服务有各自端口。

| 本实验作业 | A | B | 启动方式 |
| --- | --- | --- | --- |
| P4 | 物理卡 0—3 | 无 | launch.sh p-a，TP4/EP4 |
| D8 | 物理卡 4—7，node-rank=0 | 物理卡 0—3，node-rank=1 | launch.sh d-a / d-b，TP8/EP8、Attention DP2 |
| Pool4 | 无 | 物理卡 4—7 | Pool 服务和 sidecar，详见第 8 节 |
| 控制回放 | 物理卡 0—7 | 物理卡 0—7 | 第 12 节 torchrun，每机 8 worker；不能与上面的服务同时抢卡 |

P/D 服务用 SGLang 自己的 --nnodes/--node-rank 启动和派生 worker，**不在外面再套两机 16 进程 torchrun**；否则会重复启动模型。D8 只占每机 4 张卡，不能直接照搬 OpenSora 每机 8 卡配置。

### 4.2 启动前的端口检查；网络异常时的定向查询

控制网首选已确定为两机 enp67s0f5，当前不必重复执行下面的网卡/逐卡查询。主机控制网用于 HCCL/Gloo 建连、torchrun、HTTP 和元数据；NPU 的 RoCE 数据网用于跨机设备搬运，历史控制网设置不代替 MemFabric DEVICE_RDMA 的预检。以下网络查询仅在建连异常或需要确认设备网时执行。

```bash
# A2 宿主机
export PEER_HOST_IP=另一台A2的主机IP
ip -br -4 addr
ip route get "$PEER_HOST_IP"
# 输出中 dev 为本机接口名，src 为本机到达对端时使用的 IP
ip -details link show dev 实际接口名
ethtool 实际接口名
ethtool -i 实际接口名
```

两机网卡名可以不同。后续 HCCL_SOCKET_IFNAME、GLOO_SOCKET_IFNAME 使用各自主机的真实控制网接口，A_IP/B_IP 使用互通的 src 地址。多网口时还需验证各后端建连；一次路由查询不能证明全部通信选对网卡。host network 不会替你配置 NPU 的 IP。

每张卡的设备数据网只用 -g 查询，不使用 -s 修改；以下自动尝试常见工具位置：

```bash
# A2 宿主机
HCCN_QUERY=$(command -v hccn_tool)
if [ -z "$HCCN_QUERY" ]; then
  HCCN_QUERY=/usr/local/Ascend/driver/tools/hccn_tool
fi
if [ ! -x "$HCCN_QUERY" ]; then
  HCCN_QUERY=/usr/local/Ascend/driver/tools/hccn_tool/hccn_tool
fi
if [ -x "$HCCN_QUERY" ]; then
  for i in {0..7}; do
    echo "NPU $i"
    "$HCCN_QUERY" -i "$i" -ip -g
    "$HCCN_QUERY" -i "$i" -link -g
    "$HCCN_QUERY" -i "$i" -net_health -g
  done
else
  echo "未找到 hccn_tool，请提供实际安装路径"
fi
```

部分环境工具在 tools/hccn_tool/hccn_tool，完整采集脚本会自动检查。net_health 的 Init 或接收超时可能是未配置对应探测对象，不能单凭它判定网络故障；主机 ping 通不证明全部 NPU RoCE 路径可用。MTU、PFC/ECN、NPU IP 和交换机配置由管理员确认，不盲目改 IP、关防火墙。

端口检查：A 上 7100 为 PD store、8998 为 P bootstrap、29500 为 D rendezvous、30000/30001 为模型 HTTP；B 上 5000/6000/8100 为 Pool metadata/config/metrics；8000 为 Router。10067 为同机 CAM 握手，不要求对外开放。冒烟使用 29610/29611/29612，跨机回放使用 29620。仅向实验私网开放必要访问；后端可能另有动态连接，这不是完整防火墙规则。

**只在启动前**在两机宿主机检查所选端口，不需要全量 listen 清单：

```bash
ss -H -ltn | awk '$4 ~ /:(5000|6000|7100|8000|8100|8998|10067|29500|29610|29611|29612|29620|30000|30001)$/ {print $4}'
```

无输出表示这些 TCP 端口当时没有监听，不证明防火墙/动态端口可达。若有输出，仅回传匹配端口；核对归属后改实验配置，不杀其他服务。下载代码/权重不用等待端口检查。

### 4.3 拉代码：SGLang 与 UMDK 是两个仓库

**两台 A2 宿主机**执行。先确认 /ssd_2 可写、空间足够；BF16 权重约 60GB，另留镜像、下载缓存、构建和 profile 空间。Docker 存储通常不在 /ssd_2，要检查 DockerRootDir 所在磁盘，不擅自迁移 Docker 数据。

```bash
mkdir -p /ssd_2/models
cd /ssd_2
git clone --branch feat/qwen3-cam-detour-conflict --single-branch \
  https://github.com/COLDTEA-L/qwen3.git sglang
git -C /ssd_2/sglang status --short --branch
git -C /ssd_2/sglang rev-parse HEAD
```

用户需要先把本次 README 更新从开发服务器 push，第 14 节给命令，否则 GitHub 可能仍是旧文档。两机使用同一 commit。仓库名是 qwen3，但落地目录显式命名 sglang，不需要额外拉 Transformers。

若目录已有源码，不重复 clone、不覆盖脏工作区：

```bash
cd /ssd_2/sglang
git status --short --branch
git remote get-url origin
# 仅在工作区干净且 origin 是自己的 qwen3 Fork 时继续
git fetch origin feat/qwen3-cam-detour-conflict
git switch feat/qwen3-cam-detour-conflict
git merge --ff-only FETCH_HEAD
```

UMDK 构建分支不在 qwen3 仓库里，也不能假设已在 GitHub 发布。尚未发布时，用 Git bundle 搬运准确分支，不拿任意 main 代替。

**无卡开发服务器 8.146.225.22 执行**：

```bash
git -C /home/liuyuanwen/umdk status --short --branch
git -C /home/liuyuanwen/umdk rev-parse feat/qwen3-cam-detour-build
# 当前分支应为 9c0bbdadfcfe882426cabdce5cdf2d5d2fce92ef
git -C /home/liuyuanwen/umdk bundle create \
  /home/liuyuanwen/umdk-qwen3-cam.bundle feat/qwen3-cam-detour-build
```

**两台 A2 宿主机各执行**；root 是现有开发服务器账号，不表示 A2 也必须用 root，SSH 需按实际权限连接：

```bash
scp root@8.146.225.22:/home/liuyuanwen/umdk-qwen3-cam.bundle /ssd_2/
git clone --branch feat/qwen3-cam-detour-build \
  /ssd_2/umdk-qwen3-cam.bundle /ssd_2/umdk
git -C /ssd_2/umdk rev-parse HEAD
git -C /ssd_2/umdk status --short --branch
```

已有 UMDK 目录时先核对用户修改和历史，再 fetch bundle/fast-forward，不覆盖或删除。bundle 的 origin 是本地文件，不是长期 GitHub remote；以后发布 Fork 再明确替换。bundle 只携带 Git 源码历史，不携带权重或原来未跟踪的 build 文件。

### 4.4 如何建 Docker 和启动容器

先确认 Docker 已安装且当前用户有管理员授予的访问权限。没有 Docker、权限不足或基础镜像未知时，先反馈第 4.1 节信息，再按管理员认可方案安装；本指南不盲目装 Docker、换驱动或授予 docker 组权限。

**兼容性门槛：**当前源码的 docker/npu.Dockerfile 在 MemFabric 安装分支只接受 a3 和 950，直接传 DEVICE_TYPE=910b 会失败。Torch2.10、TorchNPU2.10、MemFabric1.2.1 等仅供核对，不是已验证的 A2 组合；不能将 A3/950 的 kernel wheel 直接用于 A2。

先选择已确认适配本机驱动、A2、所需 SGLang/DeepEP/MemFabric 功能的镜像；完整 tag、digest 和 CANN 版本等查询后固定。已有较早 A2 软件栈可能需要源码/依赖适配，不能混装最新版。新增 docker/Dockerfile.a2-tools 仅为**已合格的 Debian/Ubuntu A2 SGLang 镜像**加构建和查询工具，不安装/升级 CANN、TorchNPU、DeepEP、MemFabric，不拉权重。仅有 CANN 或 Triton 的镜像不等于完整 SGLang 环境。

**不污染原容器的方案：**保留 B 的 yuanwen_sglang_a2 不动；用其原始 ImageID 创建新的 qwen3-cam:a2-exp-v1，再创建 qwen3-cam-a2 容器。给原镜像加标签/构建子镜像不会把原容器改成实验环境。原容器可写层中额外安装的包不会自动出现在新镜像里，须在干净实验容器复核；需要补包时写进实验镜像的可复现构建步骤，不在原容器里安装。

本次不执行 docker commit：它默认会暂停原容器，而且不包含挂载卷，可能收进旧服务配置/凭据；不是避免污染所必需的步骤。[Docker 说明](https://docs.docker.com/reference/cli/docker/container/commit/)

**以下是基础栈确认后的步骤，不是已经验证通过的承诺。**若第 4.0.2 查到的环境与当前快照不兼容，先确定适配，不靠工具层 Dockerfile 自动解决。

```bash
# B 宿主机：固定个人容器实际使用的原始镜像，避免标签漂移
export A2_SOURCE_IMAGE_ID=$(docker inspect yuanwen_sglang_a2 --format '{{.Image}}')
test -n "$A2_SOURCE_IMAGE_ID" || { echo "未找到原始镜像"; exit 1; }
docker image inspect "$A2_SOURCE_IMAGE_ID" \
  --format 'ID={{.Id}} Architecture={{.Architecture}} Entrypoint={{json .Config.Entrypoint}} Cmd={{json .Config.Cmd}}' \
  || { echo "原始镜像不可用，停止构建"; exit 1; }
# 仅当这个新标签尚未被其他镜像使用时添加；已有时先核对其ID，不覆盖
if docker image inspect qwen3-cam:a2-base-v1 >/dev/null 2>&1; then
  test "$(docker image inspect qwen3-cam:a2-base-v1 --format '{{.Id}}')" = "$A2_SOURCE_IMAGE_ID" \
    || { echo "基础标签已指向另一镜像，请选新标签"; exit 1; }
else
  docker tag "$A2_SOURCE_IMAGE_ID" qwen3-cam:a2-base-v1
fi
export A2_BASE_IMAGE=qwen3-cam:a2-base-v1
cd /ssd_2/sglang
# 实验标签已存在时先核对，不覆盖别人的镜像；重复构建另取exp-v2等新标签
if docker image inspect qwen3-cam:a2-exp-v1 >/dev/null 2>&1; then
  echo "实验标签已存在：先核对来源；本步骤停止，避免覆盖"
  exit 1
fi
docker build --build-arg A2_BASE_IMAGE="$A2_BASE_IMAGE" \
  -t qwen3-cam:a2-exp-v1 \
  -f experiments/qwen3_cam/docker/Dockerfile.a2-tools \
  experiments/qwen3_cam/docker
docker image inspect qwen3-cam:a2-exp-v1 --format 'EXPERIMENT_IMAGE_ID={{.Id}}'
```

如果原始镜像已被清理导致 image inspect 失败，先反馈；不从原容器临时 commit，不盲拉 latest。若最终要用另一个合格镜像，替换 A2_BASE_IMAGE 后再构建。基础镜像及工具层软件来源都需要可访问的网络；构建失败时保留日志，不修改业务容器。

两机 CPU 架构一致且驱动兼容后，**由 B 构建并复制到 A**，避免 A/B 独立构建造成依赖漂移。私有 hosts.env 已创建时才 source，A2_SSH_USER 为 A 机实际登录账号：

```bash
# B 宿主机；此前已经建立两机私有hosts.env
source /ssd_2/sglang/experiments/qwen3_cam/runtime-config/hosts.env
: "${A_IP:?先填写A的实际地址}"
: "${A2_SSH_USER:?先填写A机登录账号}"
docker save qwen3-cam:a2-exp-v1 -o /ssd_2/qwen3-cam-a2-exp-v1.tar
sha256sum /ssd_2/qwen3-cam-a2-exp-v1.tar
scp /ssd_2/qwen3-cam-a2-exp-v1.tar "${A2_SSH_USER}@${A_IP}:/ssd_2/"
```

```bash
# A 宿主机；先确认没有其他镜像占用同名实验标签，再导入
sha256sum /ssd_2/qwen3-cam-a2-exp-v1.tar
# 先单独执行上面校验；与B输出相同后，再执行以下导入块
if docker image inspect qwen3-cam:a2-exp-v1 >/dev/null 2>&1; then
  docker image inspect qwen3-cam:a2-exp-v1 --format '{{.Id}}'
  echo "实验标签已存在：先与B核对ID；不重复导入或覆盖"
else
docker load -i /ssd_2/qwen3-cam-a2-exp-v1.tar
fi
docker image inspect qwen3-cam:a2-exp-v1 --format 'EXPERIMENT_IMAGE_ID={{.Id}}'
```

两机输出的实验 ImageID 应相同。镜像 tar 需要额外磁盘空间；不自动删除，确认两机导入成功后由用户决定是否保留用于复现。

每机**一个新实验容器承载全部本机角色**；两机可以同名 qwen3-cam-a2，B 的原容器仍叫 yuanwen_sglang_a2，互不修改。必须映射全部 8 张卡，不能把 P/D/Pool 容器可见设备切开或重编号。模板不默认 --privileged，设备/peer IPC 权限需要按实际驱动验证；失败先定位，再按获准方案调整，不跳过检查。

```bash
# A、B 宿主机各执行；设备/驱动使用下面这些路径时才继续
for node in /dev/davinci{0..7} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  test -c "$node" || { echo "缺少设备节点: $node"; exit 1; }
done
for directory in /ssd_2 /usr/local/Ascend/driver /usr/local/Ascend/firmware; do
  test -d "$directory" || { echo "缺少目录: $directory"; exit 1; }
done
NPU_SMI_HOST_PATH=$(command -v npu-smi)
test -n "$NPU_SMI_HOST_PATH" || { echo "缺少 npu-smi"; exit 1; }
docker run -d --init --name qwen3-cam-a2 \
  --network=host --ipc=host \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  --mount type=bind,src=/ssd_2,dst=/ssd_2 \
  --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
  --mount type=bind,src=/usr/local/Ascend/firmware,dst=/usr/local/Ascend/firmware,readonly \
  --mount "type=bind,src=${NPU_SMI_HOST_PATH},dst=/usr/local/bin/npu-smi,readonly" \
  --workdir /ssd_2/sglang \
  qwen3-cam:a2-exp-v1 sleep infinity
docker exec -it qwen3-cam-a2 bash
```

同名实验容器已存在就查询状态，不自动删除/覆盖。新容器的软件可写层独立，但 /ssd_2 是共享挂载，因此实验不得修改原业务的目录/权重；NPU、HBM 和 HCCS 也不被容器隔离。不要在原业务仍占卡时启动本实验，也不要为腾卡停止其他人的容器。若基础镜像有自定义 ENTRYPOINT，先确认它接受 sleep infinity，勿盲目绕过初始化。额外 dcmi、ascend_install.info、hccn.conf 的只读挂载和权限以实际驱动/镜像要求为准，查询后补齐。

不把宿主机 CANN toolkit 挂盖容器 toolkit，以免混用库。--mount 在源不存在时报错；--network=host 直接使用主机端口、不需要 -p；--ipc=host 使用主机 /dev/shm，不能靠 --shm-size 扩大它。请使用专用维护窗口，不随意改变他人的共享内存配置。

### 4.5 容器内确认版本与源码路径

**容器内**查询匹配 CANN 的 set_env.sh，实际不同就替换下面路径。driver 版本不等于 toolkit 版本，需提供实际 toolkit 版本文件内容。

```bash
find /usr/local/Ascend -maxdepth 4 -name set_env.sh -print
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export MODEL_PATH=/ssd_2/models/Qwen3-30B-A3B
export PYTHONPATH=/ssd_2/sglang/python${PYTHONPATH:+:$PYTHONPATH}
cd /ssd_2/sglang
npu-smi info
python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print("NPU_COUNT", torch.npu.device_count())'
python -m pip show sglang sglang-router sgl-kernel-npu torch torch-npu \
  deep-ep deep-ep-npu memfabric-hybrid memcache-hybrid
python -c 'import importlib.util; print(importlib.util.find_spec("sglang").origin)'
python experiments/qwen3_cam/test_cpu.py
find /usr/local/Ascend -maxdepth 4 -type f \
  \( -name version.cfg -o -name version.info -o -name ascend_toolkit_install.info \) -print
```

以上用于已经按第 4.4 节准备的新实验容器。未知是否可用的已有容器按第 4.5.1 节查询，不预先切换源码或安装实验。某些包名因安装方式不同而变化，pip show 缺失不是安装最新版的理由。新实验容器要确认 8 卡可见、sglang 来自 /ssd_2/sglang/python、CAM 所需 toolkit 编译器/Python headers 可用；runtime-only 镜像不够。

每次 docker exec 新开终端都重新 source 配套 CANN、export PYTHONPATH/MODEL_PATH 和所需 IP；不会继承另一个交互终端临时 export 的变量。保留已匹配的镜像依赖，不在本步骤 pip install -U sglang/torch/torch_npu 或换 CUDA 包。若原生服务/preflight 不通过，不继续 CAM 性能测量。

#### 4.5.1 不知道哪个已有容器可用时，如何筛选

镜像是创建容器的模板，容器可写层可能额外装过包；“容器能用”不等于“原镜像能复现”。名字含 ascend/sglang、状态 Up、npu-smi 正常，都只是候选线索，不是本实验兼容性证据。

**第一步：两台 A2 宿主机各列清单。** 不需要启动任何容器或先拉实验代码：

```bash
hostname
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
docker image ls --digests
```

优先选自己或获准查询的、已运行的 Ascend/PyTorch/SGLang 容器。不批量进入全部容器，不确定归属先把清单发回来，由用户/管理员确认候选。镜像名中的 910b/A2 可作为线索，A3/950 或 CUDA-only 镜像不能据此当作 A2 环境。

**第二步：查选定候选的配置。** 只输出指定字段，不打印完整 inspect 或 Env，以免泄露凭据：

```bash
export CANDIDATE=你的已获准查询的容器名
docker container inspect "$CANDIDATE" \
  --format 'Name={{.Name}} ImageTag={{.Config.Image}} ImageID={{.Image}} State={{.State.Status}} Network={{.HostConfig.NetworkMode}} IPC={{.HostConfig.IpcMode}} Runtime={{.HostConfig.Runtime}} Privileged={{.HostConfig.Privileged}}'
docker container inspect "$CANDIDATE" --format '{{json .HostConfig.Devices}}'
docker container inspect "$CANDIDATE" --format '{{json .HostConfig.DeviceRequests}}'
docker container inspect "$CANDIDATE" \
  --format '{{range .Mounts}}{{println .Source "->" .Destination "RW=" .RW}}{{end}}'
```

记录映射的 NPU、driver/firmware 来源、/ssd_2 挂载和网络/IPC 模式。使用 Ascend runtime/OCI hook 时 Devices 可能为空，要结合容器内可见性，不能只凭字段判定没卡。缺 /ssd_2 或 host network/shared IPC，可能只是部署配置需改，不等于镜像软件不合格；在独立实验容器里改，不动现有业务容器。

**停止、暂停或重启中的容器到此为止。** 不直接 docker start：原启动命令可能恢复模型服务、抢卡或占端口。只记录 ImageID；若要验证该镜像，先确认维护窗口，再按第 4.4 节建立独立容器。

**第三步：只对已获准的 running 候选查询内部环境。** 不安装包、不加载模型、不创建通信域或测试张量。工具缺失则保留报错：

```bash
# A2 宿主机，仅对已确认 running 的候选执行
docker exec "$CANDIDATE" bash -c \
  'uname -m; cat /etc/os-release; command -v python; command -v python3; find /usr/local/Ascend -maxdepth 4 -name set_env.sh -print'
docker exec "$CANDIDATE" bash -c \
  'npu-smi info; ls -l /dev/davinci{0..7} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc'
```

拉取实验代码后，推荐用新增的脚本，输出 CANN/包版本、模块导入结果和可见卡数。通过宿主机标准输入传入探测内容，候选容器**无需挂载 /ssd_2 或安装脚本**：

```bash
# 默认只列清单，不进入任何容器
bash /ssd_2/sglang/experiments/qwen3_cam/collect_container_info.sh
# 选定一个获准候选后
bash /ssd_2/sglang/experiments/qwen3_cam/collect_container_info.sh "$CANDIDATE"
# 若需要加载 CANN，先从上一步查询中选择匹配镜像的可信路径
bash /ssd_2/sglang/experiments/qwen3_cam/collect_container_info.sh \
  "$CANDIDATE" /usr/local/Ascend/ascend-toolkit/set_env.sh
```

第二个参数是容器内真实路径，默认路径并非保证存在；先核对它与镜像库匹配，不盲目 source。脚本不批量探测，不启动/停止/重启容器，不升级包，不打印完整环境变量，不提交镜像。每项 30 秒超时；超时表示未确认，不是兼容性否决。导入库可能有少量初始化开销，忙碌业务容器先经所有者同意。

若使用专用 conda/venv，默认 docker exec 的 PATH 可能不是原服务使用的 Python。提供激活方法或实际解释器路径后重新查询；不要把“默认 python 没 Torch”等同“容器没 Torch”。脚本只初筛，不自动宣告哪个容器最佳。

| 结果 | 如何理解/下一步 |
| --- | --- |
| 架构和驱动/CANN/TorchNPU 组合明确；TorchNPU 可导入，NPU_AVAILABLE=True，NPU_COUNT=8 | 基础候选，仍需核对目标 A2 和版本兼容性 |
| npu-smi 正常但 TorchNPU 缺失/导入失败 | 可能只有驱动工具，或 Python 环境没激活；保留错误，不盲装最新版 |
| 只见 4 卡或设备访问失败 | 核对设备映射、可见设备设置和权限；其镜像仍可能可复用到新 8 卡容器 |
| sgl_kernel_npu/deep_ep 缺失或不兼容 | 当前 MoE 依赖未满足或后端需要适配；包名存在也不证明支持 A2 EP8 |
| DEVICE_RDMA_API=False 或 MemCache API 缺失 | 当前接口不匹配/未确认，不能直接进入跨机 KV/Pool；保留实际版本 |
| toolkit 编译器/Python headers 不可用 | 可能能运行但不能编译 CAM；工具层不会替你安装 CANN |
| 所有基础查询通过 | 仍只是候选，需第 5—8 节的 CAM 正确性、原生 EP8/PD 和 Pool 读写验证 |

还没安装 umdk_cam_op_lib 很正常，由第 5 节构建，不因此排除候选。已有容器的 SGLang 路径也不必是 /ssd_2，仅新实验容器要求指向实验分支。选定后在干净新容器复核，避免依赖旧容器可写层改动；不擅自 docker commit 现有容器。

本次已有两机清单，不再重复提交；只按第 4.0.1、4.0.2 补信息。只有原候选不可用时，才重新筛选一个获准候选并提供其配置/激活方式。不发密码/Token。根据这些信息再固定基础镜像、依赖、挂载和网卡，不直接对未知容器运行模型。

### 4.6 下载原版 BF16 权重并校验

仅在有卡的 A2 下载；两机均保留本地完整目录。选 Qwen/Qwen3-30B-A3B，不与 Instruct-2507、量化版或 Qwen3-Next 混用。Pool 卡不加载模型，不代表 D 所在机器可以缺权重。

推荐宿主机独立下载工具环境，避免 ModelScope 改动推理依赖；需要已有 Python3/venv，缺失则先报告，不自行换系统 Python：

```bash
# A2 宿主机；两机各下载，或 A 下载后同步 B
python3 -m venv /ssd_2/qwen3-download-venv
/ssd_2/qwen3-download-venv/bin/python -m pip install modelscope
/ssd_2/qwen3-download-venv/bin/modelscope download \
  --model Qwen/Qwen3-30B-A3B \
  --local_dir /ssd_2/models/Qwen3-30B-A3B
```

已有可用 CLI 也可直接执行同一 download 命令，不必重装。上方默认原版模型主分支；记录实际 revision，此后不在对照之间更新。最好 A 下载后同步 B，防止两次下载期间主分支变化：

```bash
# A 宿主机；变量填实际 B 机账号/IP，目标父目录先按第 4.3 节创建
rsync -a --info=progress2 /ssd_2/models/Qwen3-30B-A3B/ \
  "${A2_SSH_USER}@${B_IP}:/ssd_2/models/Qwen3-30B-A3B/"
```

不使用 --delete、不混入已有的另一模型目录；目标已存在时先核对模型身份和文件列表。两机**容器内**只读验证配置和分片存在性，不加载 30B 权重：

```bash
export MODEL_PATH=/ssd_2/models/Qwen3-30B-A3B
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["MODEL_PATH"])
config = json.loads((root / "config.json").read_text())
print({k: config.get(k) for k in (
    "model_type", "num_hidden_layers", "num_experts", "num_experts_per_tok",
    "num_key_value_heads", "head_dim", "torch_dtype", "dtype")})
assert config["model_type"] == "qwen3_moe"
assert config["num_hidden_layers"] == 48
assert config["num_experts"] == 128 and config["num_experts_per_tok"] == 8
index = json.loads((root / "model.safetensors.index.json").read_text())
files = sorted(set(index["weight_map"].values()))
assert all((root / name).is_file() for name in files), "缺少权重分片"
assert (root / "tokenizer.json").is_file()
print("MODEL_FILES_PASS", len(files))
PY
```

再做两机内容校验；大文件校验占用磁盘带宽，在采样之前完成：

```bash
# A 宿主机，清单放在模型目录外
cd /ssd_2/models/Qwen3-30B-A3B
find . -type f \( -name '*.json' -o -name '*.safetensors' -o -name '*.jinja' \) \
  -print0 | sort -z | xargs -0 sha256sum > /ssd_2/Qwen3-30B-A3B.sha256
scp /ssd_2/Qwen3-30B-A3B.sha256 "${A2_SSH_USER}@${B_IP}:/ssd_2/"
# B 宿主机
cd /ssd_2/models/Qwen3-30B-A3B
sha256sum -c /ssd_2/Qwen3-30B-A3B.sha256
```

全部 OK 后再部署。约 60GB 只是权重磁盘量，不是单卡 HBM 需求；模型分片、KV、临时张量、CAM 窗口、Pool 均需独立 HBM 预算。

## 5 第二步安装实验分支和 CAM

完成第 4 节后在两机实验容器内构建。本节及后续 torchrun/服务命令均在容器内执行（宿主机查询和客户端压测另有说明）。不要误装到宿主机或覆盖用户工作区。

```bash
cd /ssd_2/sglang
git switch feat/qwen3-cam-detour-conflict
export PYTHONPATH=/ssd_2/sglang/python${PYTHONPATH:+:$PYTHONPATH}
python experiments/qwen3_cam/test_cpu.py
```

可以在已有匹配环境里用 PYTHONPATH 指向源码，避免此时重新解析依赖。如果要 editable install，先核对 python/pyproject_npu.toml 和既有 NPU 包的版本，不用普通 CUDA 依赖替换 NPU 环境。确认 python 导入的是实验目录，不是镜像自带的另一份 SGLang。

UMDK 的构建分支保留原 All-to-All Detour 实现，仅增加独立构建目录和 CXX ABI 匹配的脚本。在专用实验容器中执行，CANN 环境路径以实际安装为准：

```bash
cd /ssd_2/umdk
git switch feat/qwen3-cam-detour-build
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_PYTHON_EXECUTABLE=$(command -v python)
export SOC_VERSION=ascend910b4
# 用户已确认910B4；这是当前UMDK脚本的参数形式，仍须匹配实际CANN编译工具
bash build/cam/comm_operator/build_qwen3_all2all_detour.sh "$SOC_VERSION"
ls output/cam/qwen3-all2all/wheel
# 用上一行显示的确切 wheel 文件名安装 不使用其他分支留下的 wheel
python -m pip install --no-deps /ssd_2/umdk/output/cam/qwen3-all2all/wheel/确切wheel文件名.whl
```

脚本会在该容器中安装生成的 CAM 自定义算子包，可能更新既有 CAM vendor 包，因此必须使用独立实验环境并保留原镜像或安装包。它不删除原来的 src/cam/comm_operator/build。编译参数与当前 PyTorch 的 CXX ABI 匹配。两机分别构建或使用完全匹配的同一架构 wheel，不能跨 CANN、TorchNPU 或 CPU 架构复用。

## 6 第三步只跑算子正确性

这一步占用整机 8 张卡，不同时运行模型或 Pool。两台机器分别执行。停止仅限自己启动的实验进程，不使用 pkill python 一类广泛命令。

```bash
cd /ssd_2/sglang
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

先使用 off 模式验证两机模型、EP8 和真实 P 到 D 的传输。各容器终端先加载用户已核对的私有 hosts.env；本次 A 的控制口候选为 enp67s0f5，B 的接口仍未提供，不用 A 的接口名猜 B。HCCL_SOCKET_IFNAME 和 GLOO_SOCKET_IFNAME 是主机控制网络接口，不代表已经配置好了各 NPU 的 RoCE 数据网络。必须先完成第 4.0 节查询和兼容性门槛。

```bash
source /ssd_2/sglang/experiments/qwen3_cam/runtime-config/hosts.env
export HOST_ROLE=A
# 上一行在 A 的终端用 A；在 B 的终端必须改为 B
: "${A_IP:?请先填A的地址}"
: "${B_IP:?请先填并确认B的地址}"
case "$HOST_ROLE" in
  A) : "${HOST_IF_A:?请先确认A接口}"; export HCCL_SOCKET_IFNAME="$HOST_IF_A" ;;
  B) : "${HOST_IF_B:?B接口尚未确认，补查询后再启动}"; export HCCL_SOCKET_IFNAME="$HOST_IF_B" ;;
  *) echo "HOST_ROLE只能为A或B"; exit 1 ;;
esac
export GLOO_SOCKET_IFNAME="$HCCL_SOCKET_IFNAME"
export MODEL_PATH=/ssd_2/models/Qwen3-30B-A3B
export ASCEND_MF_STORE_URL="tcp://${A_IP}:7100"
export ASCEND_MF_TRANSFER_PROTOCOL=device_rdma
export SGLANG_CAM_DETOUR_MODE=off
cd /ssd_2/sglang
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
export PYTHONPATH=/ssd_2/sglang/python${PYTHONPATH:+:$PYTHONPATH}
python experiments/qwen3_cam/preflight.py \
  --config experiments/qwen3_cam/configs/p_a.json \
  --model-path "$MODEL_PATH" --mode cam-detour
export SGLANG_CAM_DETOUR_MODE=cam-detour
export SGLANG_CAM_TRACE_MOE=1
export SGLANG_CAM_DETOUR_TRACE='/ssd_2/sglang/experiments/qwen3_cam/results/p-a-{rank}.jsonl'
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
  -d '{"output_dir":"/ssd_2/sglang/experiments/qwen3_cam/profiles/p","start_step":3,"num_steps":30,"activities":["CPU","GPU"]}'
curl -X POST "http://${A_IP}:30001/start_profile" \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/ssd_2/sglang/experiments/qwen3_cam/profiles/d","start_step":3,"num_steps":30,"activities":["CPU","GPU"]}'
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

## 14 更新分支与 GitHub 推送

用户仓库是 https://github.com/COLDTEA-L/qwen3，代码为 SGLang Fork，不需要再建立独立模型源码仓。README 和配套工具随实验分支提交；权重/私有配置不推送。

**无卡开发服务器 8.146.225.22** 上执行；这里仍是 /home/liuyuanwen，不是 A2 的 /ssd_2：

```bash
cd /home/liuyuanwen/sglang
git status --short --branch
git switch feat/qwen3-cam-detour-conflict
git remote get-url publish
# 核对是 https://github.com/COLDTEA-L/qwen3.git，再 push
env -u GIT_ASKPASS -u SSH_ASKPASS GIT_TERMINAL_PROMPT=1 \
  git -c core.askPass= -c credential.helper= \
  push -u publish feat/qwen3-cam-detour-conflict
```

该命令绕过先前失效的 VS Code askpass。若要求认证，用户名 COLDTEA-L，密码位置输入具有本仓库写权限的 PAT，不是账号密码；PAT 不写入命令、URL 或 README，也不发给协作者。已有可用 SSH/凭据管理器也可按既有流程认证。本次只更新本地/开发服务器分支，不代用户发布 GitHub。

UMDK 的构建分支是另一个仓库，尚未发布时按第 4.3 节用 bundle 即可。以后若发布到真实存在的 UMDK Fork，先核对 remote 和历史，不使用虚构的 URL：

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

开发端已执行配置、分组、窗口容量、分块对齐、归约代数、分析脚本等CPU单元测试，以及修改Python文件的语法检查、shell语法检查和git diff检查。容器采集脚本额外用 mock Docker 验证默认只列清单、停止/暂停状态不进入、显式指定 running 候选才诊断；已检查当前 README 的 42 段 Bash 命令语法，以及脚本内层 Bash 与 Python 的语法。本次用户回传只用于记录已观察到的配置，不是实机兼容性验收。这些检查不代表某个实际 A2 容器已合格。开发容器也未安装PyTorch，因此没有用CPU PyTorch进程组运行适配器。A2编译、CAM实际IPC、设备collective、EP8、PD和Pool命中以及性能测试均待目标环境执行，不能将CPU测试通过写成模型部署已通过。
