# Qwen3 A2 实验指导：vLLM-Ascend + Mooncake 基线与 MoE All-to-All 接入

更新：2026-10-14。机器均以 **228 / 238** 指代，个人目录均为 `/ssd_2/yuanwenliu`。本文替代旧的默认执行步骤；[SGLang 历史指导](README.sglang-legacy.md)保留作回退与取证，不与新环境混用。

## 0 本次迁移的范围和两个重要限制

本轮提供：固定镜像/版本检查、新容器创建、离线模型启动、无权重的 NPU 跨机读测试、P4+D4 基线、P4+D8 基线、OpenAI 负载与指标采集、小容量共享池接口验证，以及**显式启用的 MoE All-to-All worker适配包**。没有改写权重、没有把CAM偷换成HCCL、没有声称原三类流量实验已全部迁移。新增All-to-All仍须按8.1节在A2实测。

直接核对固定源码后发现：

1. **A2 的 EP4/EP8 默认 MoE 通信是 AllGather，不是 Dispatch/Combine All-to-All。** `vllm_ascend/ascend_forward_context.py` 的 A2 分支仅在 EP>=16、每卡专家数<=24、token 数满足容量限制时选择 MC2；否则为 AllGather。因此 D8 的 EP8 可以表示专家分片，但不能据此宣称获得原计划的 All-to-All 流量。
2. **Mooncake v0.3.9 的 A2 普通共享池是 Host DRAM，不是四卡 HBM Pool。** `mooncake-store/src/utils.cpp` 的普通 Ascend 分配分支调用 `aclrtMallocHost`；Fabric Memory 的另一路径在官方指导中针对 A3，不能给 A2 开个环境变量就等同实现。本文的小池只验证接口/复用流程，明确标记 DRAM qualification。

这两点修正此前“迁移可能更简单”的评估：原生服务/PD可以作为候选基线，但严格的 MoE All-to-All + CAM + 独立 HBM Pool 不会因换引擎而自动实现。本次先补MoE All-to-All接入和验收步骤；CAM与独立HBM Pool仍未迁移，不能用此基线替代最终实验。

| 阶段 | 本轮提供 | 验收边界 |
| --- | --- | --- |
| 环境/模型/跨机读 | 检查与启动脚本 | 仍由228/238实测；预检查不是运行成功 |
| P4+D4、P4+D8 | 原生PD启动与官方proxy接入 | D8是Attention TP4、DP2、EP8；实际通信后端须看profile |
| MoE All-to-All | 独立worker适配包、EP4/8无权重路由测试、输出对照 | 默认native不变；alltoall显式启用；尚无A2实测结果 |
| 共享池 | 原生MultiConnector小容量DRAM验证 | 不是独立Pool4、不是远端HBM恢复已完成 |
| CAM多路径 | 保留UMDK及旧接入代码，整理移植验收要求 | 新vLLM入口拒绝CAM模式，尚未接入 |
| 三流冲突/控制器 | 给出测量方法 | 未证明存在冲突，未实现统一控制器 |

实验代码仍通过你已有的 `COLDTEA-L/qwen3` 仓库分发。虽然工作目录名是 `sglang`，新脚本**不会把其 `python/sglang` 加入PYTHONPATH，不会运行SGLang引擎**；暂不改目录名，避免移动已有数据和挂载。

## 1 固定环境，不重新混装依赖

已知硬件：两机各8张910B4、每卡32GB HBM、HCCS FullMesh；aarch64；driver25.5.2；主机控制接口`enp67s0f5`。

| 项目 | 固定候选版本/来源 |
| --- | --- |
| vLLM | 0.22.1；源码`0decac0d96c42b49572498019f0a0e3600f50398` |
| vLLM-Ascend | 0.22.1rc1；源码`5f6faa0cb8830f667266f3b8121cd1383606f2a1` |
| CANN / NNAL | 9.0.0 / 9.0.0，来自新镜像 |
| torch / torch_npu | 2.10.0 / 2.10.0 |
| Triton Ascend | 官方配套表3.2.1；不继承旧镜像中的dev包 |
| Mooncake | v0.3.9；源码`a00f75742469d6cc408bc5393b73c501e92dc74a`；USE_ASCEND_DIRECT构建 |
| A2官方镜像tag | `quay.io/ascend/vllm-ascend:v0.22.1rc1` |
| 固定arm64 manifest | `sha256:638fc04eaa3654fcf14688096ed4e9d88ea0d905fa8685eed4b36d5fffe8fd8d` |

镜像tag与arm64子manifest于2026-10-14通过Quay接口核对。Manifest digest不是Docker ImageID。正式镜像的默认Python为3.12；这是**新独立容器**，不升级旧SGLang容器的Python，不重装其MemFabric/DeepEP/kernel。源码包版本、镜像来源和实际NPU运行仍要逐层验收。

已有权重继续用`/ssd_2/yuanwenliu/models/Qwen3-30B-A3B`，无需重新下载。官方Qwen3模型指导的A2示例是64GB，本环境是32GB，所以先2048上下文、4并发、eager模式；不照搬长上下文/高并发配置，也不宣称显存一定足够。

有卡服务器不连接无卡开发服务器。代码经托管站分发，运行时只在228/238之间通信。实验机只拉取代码，不承担源码发布。

## 2 两机准备代码和配置（宿主机）

先确认本次修订已在代码托管站发布。已有实验仓库分别在228、238执行；若有本地改动或origin不是你的qwen3仓库，停下检查，不reset、不覆盖：

```bash
cd /ssd_2/yuanwenliu/sglang
git status --short --branch
git remote get-url origin
git pull --ff-only origin feat/qwen3-cam-detour-conflict
git rev-parse HEAD
```

第一次没有实验仓库时才执行：

```bash
cd /ssd_2/yuanwenliu
git clone --branch feat/qwen3-cam-detour-conflict https://github.com/COLDTEA-L/qwen3.git sglang
```

两机另取固定版本的vLLM-Ascend源码，供官方proxy和源码核对使用；不对宿主机安装Python依赖。目录已存在时先检查，不覆盖：

```bash
cd /ssd_2/yuanwenliu
git clone --depth 1 --branch v0.22.1rc1 https://github.com/vllm-project/vllm-ascend.git vllm-ascend-qwen3
git -C /ssd_2/yuanwenliu/vllm-ascend-qwen3 rev-parse HEAD
```

应为上表的`5f6faa0...`。无需重新拉取UMDK，新基线暂不编译/安装CAM；旧仓库不动。

每台机执行一次配置生成：

```bash
cd /ssd_2/yuanwenliu/sglang
python3 experiments/qwen3_vllm/runtime.py configure
```

作用：记录本地模型/源码路径，生成后续小容量DRAM共享池的配置。文件在`experiments/qwen3_vllm/runtime-config/`，被Git忽略，权限0600，不包含代理密码。两机按相同已知地址分别生成，**不再复制旧SGLang的deployment.env**。重复执行会拒绝覆盖，需先检查已有文件。

## 3 238准备新镜像，再复制给228（宿主机）

先查看磁盘，不能仅凭之前166GB剩余空间就认为足够：

```bash
df -h /ssd_1/docker /ssd_2/yuanwenliu
docker system df
```

不要自动清理共享镜像/容器。联网仅用于代码和镜像准备，不用于推理下载权重。Docker拉镜像走daemon的代理配置，**shell中export代理不一定影响docker pull**；如果出现Proxy Authentication Required，需修正获准的daemon/镜像获取方式，不重启共享Docker服务、不把密码写进Git。

在238拉固定A2 arm64镜像并打本地tag：

```bash
docker pull quay.io/ascend/vllm-ascend@sha256:638fc04eaa3654fcf14688096ed4e9d88ea0d905fa8685eed4b36d5fffe8fd8d
docker tag quay.io/ascend/vllm-ascend@sha256:638fc04eaa3654fcf14688096ed4e9d88ea0d905fa8685eed4b36d5fffe8fd8d qwen3-vllm:a2-base-pinned
docker image inspect qwen3-vllm:a2-base-pinned --format '{{.Id}} {{.Architecture}} {{json .RepoDigests}}'
```

Architecture必须为arm64。新Dockerfile增加版本检查工具和本仓库的纯Python MoE适配包，以`--no-deps --no-build-isolation --no-index`离线安装；不运行apt、不升级Torch/Transformers等依赖。沿用旧Docker支持的builder，不使用`--progress`。本次用v2新tag，不覆盖此前v1镜像：

```bash
cd /ssd_2/yuanwenliu/sglang
DOCKER_BUILDKIT=0 docker build --network=none -f experiments/qwen3_vllm/Dockerfile.a2-baseline --build-arg A2_BASE_IMAGE=qwen3-vllm:a2-base-pinned -t qwen3-vllm:a2-native-v2 experiments/qwen3_vllm
docker image inspect qwen3-vllm:a2-native-v2 --format '{{.Id}} {{.Architecture}}'
```

先得到Successfully built/tagged，才保存；失败时不要继续运行不存在的镜像。构建只查元数据/来源，不加载模型或NPU。出现QUALIFICATION_FAILED就保存检查结果，不自动把不匹配包升级成最新版。

238保存镜像，使用scp传给228，不需要rsync：

```bash
docker save -o /ssd_2/yuanwenliu/qwen3-vllm-a2-native-v2.tar qwen3-vllm:a2-native-v2
sha256sum /ssd_2/yuanwenliu/qwen3-vllm-a2-native-v2.tar
scp /ssd_2/yuanwenliu/qwen3-vllm-a2-native-v2.tar yuanwenliu@10.155.205.228:/ssd_2/yuanwenliu/
```

228核对同一tar校验和并导入：

```bash
sha256sum /ssd_2/yuanwenliu/qwen3-vllm-a2-native-v2.tar
docker load -i /ssd_2/yuanwenliu/qwen3-vllm-a2-native-v2.tar
docker image inspect qwen3-vllm:a2-native-v2 --format '{{.Id}} {{.Architecture}}'
```

两机最终ImageID应一致。镜像tar暂留作为证据，不自动删除。

若已建好此前的v1镜像/容器且原生预检查已通过，可以按8.1.1节在**新vLLM实验容器**中离线安装适配包，跳过重复拉镜像。不要把包安装到旧SGLang容器。首次部署按本节v2流程。

## 4 新建容器与进入方式（两机宿主机）

不替换`qwen3-cam-a2-v3`或`yuanwen_sglang_a2`。新建容器本身不启动模型，但后续测试前必须正常结束自己的旧模型worker，避免同一张卡被两套服务占用；不能停他人的任务。

确认挂载文件存在，尤其HCCN配置：

```bash
ls -l /etc/hccn.conf /etc/ascend_install.info
```

两机各执行：

```bash
cd /ssd_2/yuanwenliu/sglang
bash experiments/qwen3_vllm/create_container.sh qwen3-vllm:a2-native-v2 qwen3-vllm-a2-v2 --print-only
bash experiments/qwen3_vllm/create_container.sh qwen3-vllm:a2-native-v2 qwen3-vllm-a2-v2
docker ps --filter name=qwen3-vllm-a2-v2
```

脚本沿用已知可用的ascend runtime、host网络、host IPC、八卡设备与驱动/DCMI挂载，新增`/etc/hccn.conf`和固定proxy源码；权重只读。不挂整棵`/var`、`/home`。privileged是实验配置，不是生产隔离建议。容器名已存在时拒绝覆盖。

**每个新终端先在宿主机进入容器：**

```bash
docker exec -it qwen3-vllm-a2-v2 bash
```

然后228容器内：

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm/env.sh 228
```

238容器内改成：

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm/env.sh 238
```

不要在容器内再次执行docker exec，不要source旧qwen3_cam/env.sh。新env配置CANN/NNAL、指定主机IP/网卡、A2 RoCE、离线模式，清理下载代理和旧SGLang路径/PD配置。**模型不联网；PD控制/数据通信仍必须联机。**

官方A2 Direct传输要求的`HCCL_INTRA_ROCE_ENABLE=1`允许机内RoCE路径；不能因为硬件有HCCS FullMesh，就假定这轮所有机内通信仍走HCCS。最终冲突实验需要验证每类流量的实际路径，并统一各对照组的该配置，避免把选路变化当成拥塞或优化收益。

## 5 预检查与NPU跨机搬运关口

每机新容器内执行：

```bash
python experiments/qwen3_vllm/verify.py --device --model --source --moe-adapter
```

需QUALIFICATION_PASS。它只证明包、API、来源、设备可见、文件存在，不代表推理或搬运正确。若Mooncake源码来源无法核对、版本不符、NNAL缺失或权重缺失，停止。已有权重不重下；仅在确实缺失时参考历史指导4.5的独立下载环境，不在推理镜像里升级Transformers。

两机宿主机查询NPU IP/链路；无需全量listen列表，也不修改网络：

```bash
for i in 0 1 2 3; do /usr/local/Ascend/driver/tools/hccn_tool -i "$i" -ip -g; done
for i in 0 1 2 3; do /usr/local/Ascend/driver/tools/hccn_tool -i "$i" -link -g; done
for i in 0 1 2 3; do /usr/local/Ascend/driver/tools/hccn_tool -i "$i" -net_health -g; done
```

权限不足时使用获准的sudo。net_health取决于已配置的检测对象，不能仅凭Receive timeout判断线路坏。`10.155.205.x`是Host控制网，NPU的RoCE IP可以不同；不要据此重配IP。后续D8需补查228卡4–7。

**只在获准且空闲的两机物理卡0上运行1MiB跨机读测试，先不要加载模型。**

228容器窗口1，source新env后：

```bash
python -u experiments/qwen3_vllm/te_probe.py serve
```

看到TE_SOURCE_READY并保持前台。该进程最多保留15分钟，只有1MiB测试HBM；不启动MemFabric store，也不用7100。描述文件是当前进程的临时指针/动态RPC端口，不能复用已退出进程的文件。

在238宿主机另一个窗口复制**本轮**描述文件，目标文件不存在时执行：

```bash
scp yuanwenliu@10.155.205.228:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm/runtime-config/te-source-228.json /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm/runtime-config/
```

238新容器source后：

```bash
python -u experiments/qwen3_vllm/te_probe.py read
```

必须TE_NPU_READ_PASS并通过内容校验；初始化通过不算搬运通过。完成后在228源窗口Ctrl+C，释放本次测试内存。文件保留取证；重跑时用`--descriptor`指定新的文件名并复制同名文件，不覆盖、不复用旧指针。失败时先取最早的Mooncake/ADXL原生日志，不反复加载61GB权重。此测试通过仍不证明所有卡、完整PD建组或Pool正确。

## 6 单机四卡模型基线（228容器）

只用228卡0–3，确认归属与空闲。先看生成命令，不启动NPU：

```bash
python experiments/qwen3_vllm/runtime.py serve --role solo-228 --dry-run
```

实际启动，保留新日志：

```bash
set -o pipefail
python -u experiments/qwen3_vllm/runtime.py serve --role solo-228 2>&1 | tee "experiments/qwen3_vllm/results/solo-228-$(date +%Y%m%d-%H%M%S).log"
```

默认BF16、TP4/EP4、2048上下文、4并发、0.75显存比例、eager；关闭FlashComm/Matmul-AllReduce融合，未开CAM、Pool、量化、EPLB、推测解码。注意MoE实际为原生AllGather，不是原SGLang DeepEP后端。

228另一个已source的新容器窗口，检查服务并发短请求：

```bash
curl --noproxy '*' -fsS http://10.155.205.228:30009/health
curl --noproxy '*' -fsS http://10.155.205.228:30009/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen3","prompt":"Explain what a KV cache is in one sentence.","max_tokens":32,"temperature":0}'
```

应HTTP200且有正常文本。OOM要区分权重加载、启动profile临时张量、KV预分配；降低上下文/并发不能解决所有权重峰值问题。保存完整最早traceback和npu-smi，不直接换量化权重。完成后模型窗口Ctrl+C，确认自己的worker退出，才进下一步。

## 7 P4+D4跨机PD（先不启用Pool/CAM）

| 主机 | 卡 | 角色 | HTTP |
| --- | --- | --- | --- |
| 228 | 0–3 | P：TP4、DP1、EP4 | 30010 |
| 238 | 0–3 | D：TP4、DP1、EP4 | 30011 |

这一步是迁移关口，不是最终P4+D8+Pool4布局。不用torchrun，不启动SGLang PD store，不启动Mooncake master（P2P搬运使用P2PHANDSHAKE）。

228容器窗口1：

```bash
set -o pipefail
python -u experiments/qwen3_vllm/runtime.py serve --role p-228 --layout pd4 2>&1 | tee "experiments/qwen3_vllm/results/p-pd4-$(date +%Y%m%d-%H%M%S).log"
```

238容器窗口1：

```bash
set -o pipefail
python -u experiments/qwen3_vllm/runtime.py serve --role d-238 --layout pd4 2>&1 | tee "experiments/qwen3_vllm/results/d-pd4-$(date +%Y%m%d-%H%M%S).log"
```

两端都启动成功后，228容器窗口2检查并启动固定源码里的官方proxy：

```bash
curl --noproxy '*' -fsS http://10.155.205.228:30010/health
curl --noproxy '*' -fsS http://10.155.205.238:30011/health
python -u experiments/qwen3_vllm/runtime.py proxy --layout pd4
```

228容器窗口3发到proxy，而不是直接把D当普通混合推理服务：

```bash
curl --noproxy '*' -fsS http://10.155.205.228:30080/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen3","prompt":"Explain prefill and decode separation.","max_tokens":32,"temperature":0}'
python experiments/qwen3_vllm/workload.py --requests 20 --rps 0.2 --concurrency 2 --output "experiments/qwen3_vllm/results/pd4-$(date +%Y%m%d-%H%M%S).jsonl"
```

验收：两端完整启动、请求成功率、文本、P实际提供KV、D实际拉取KV及转移完成。health/HTTP200本身不足以证明没有recompute/回退，需结合connector日志。初次1MiB测试时延不能当作模型PD性能。

## 8 回到P4+D8资源布局（仍无CAM/独立HBM Pool）

先正常结束自己的proxy、P/D，确认旧worker退出。不要在原pd4服务上叠加新布局。

| 主机 | 卡 | 角色 | 并行/接口 |
| --- | --- | --- | --- |
| 228 | 0–3 | P4 | TP4、DP1、EP4；30010 |
| 228 | 4–7 | D的DP rank0 | 本地TP4，D整体DP2/EP8；30011 |
| 238 | 0–3 | D的DP rank1 | 本地TP4，D整体DP2/EP8；30011 |
| 238 | 4–7 | 为目标Pool4/CAM中继保留 | 本轮不分配HBM Pool、不创建CAM窗口 |

228 P窗口：

```bash
python -u experiments/qwen3_vllm/runtime.py serve --role p-228 --layout pd8
```

228 D窗口：

```bash
python -u experiments/qwen3_vllm/runtime.py serve --role d8-228 --layout pd8
```

238 D窗口：

```bash
python -u experiments/qwen3_vllm/runtime.py serve --role d8-238 --layout pd8
```

两端D同时启动才形成全局组。这里不是两个独立EP4 D服务：DP rank0/1共同参与D的专家并行组，Attention各在本机TP4内；proxy向两个HTTP端点做外部负载均衡。实际EP8和跨机MoE流量必须由运行配置/profile验证；以上不加开关的命令仍为native/AllGather对照。新增All-to-All按下一节启用。

三端就绪后，228另开容器窗口：

```bash
python -u experiments/qwen3_vllm/runtime.py proxy --layout pd8
```

重复第7节短请求和低负载测试。三处启动均可用tee另存日志；结果目录和profile须按轮次区分，不覆盖。

## 8.1 新增：MoE Dispatch/Combine All-to-All 接入与验收

### 8.1.1 接入方式和环境检查

入口是`runtime.py serve --moe-comm alltoall`，它设置显式worker类`qwen_moe_adapter.worker.QwenAllToAllWorker`。worker在加载模型前装配固定版本的选择策略，实际使用上游`AlltoAllCommImpl → TokenDispatcherWithAll2AllV → HCCL all_to_all_single`。**不是CAM，不改KV后端，不改框架源码文件，也不注册默认自动加载的插件。** 不加开关或`--moe-comm native`均保留原生选择策略。

当前支持范围：原始128专家/top8的Qwen3-30B-A3B，BF16无量化、Attention TP4、DP1/DP2（EP4/EP8）、专家内部TP1、eager/V1 runner；不支持EPLB、图捕获、FlashComm、DBO、投机解码和多流重叠。适配补齐token数至TP4的整数倍，combine后按原始token数恢复，覆盖小batch、非整除batch与空DP输入。空输入的实机支持仍以路由测试为准。

v2镜像已离线装好适配包。已有v1新vLLM容器则先按第2节在两机宿主机更新代码，再在**各自v1实验容器内**source新env并执行：

```bash
cd /ssd_2/yuanwenliu/sglang
python -m pip install --no-deps --no-build-isolation --no-index -e experiments/qwen3_vllm/moe_adapter
python experiments/qwen3_vllm/verify.py --image-only --moe-adapter
```

此项不下载、不更新依赖；editable安装让该容器用挂载仓库里的适配包，需记录代码HEAD。不得套用到旧SGLang容器。v2首次部署也运行上面的verify，但不必重复安装。适配包校验vLLM/Ascend/Torch版本及五个关键源码文件SHA256；不匹配就拒绝启用，回传差异，不删除保护逻辑。镜像构建检查不初始化NPU。

两机容器可执行CPU回归及命令预览，不占用卡：

```bash
python experiments/qwen3_vllm/test_moe_adapter.py
python experiments/qwen3_vllm/runtime.py serve --role d8-228 --layout pd8 --moe-comm alltoall --dry-run
```

无需再建一个vllm仓库：固定镜像已有框架，额外宿主机vLLM-Ascend仓库供proxy使用。本实验仓库分发适配代码；UMDK/CAM保持不动。

### 8.1.2 先做无权重EP4/EP8路由测试

确认实验卡可用并正常结束自己的模型服务，不与第6–8节服务同时运行。下面的`torchrun`**仅用于小型路由probe**，正式模型serve依然不用torchrun。probe验证真实HCCL Dispatch/Combine、TP切分/拼回、均匀/不均匀token、热点专家与空DP输入；专家计算用已知缩放替代，不加载权重、不连接PD store、不测试真实FFN或KV。

228容器source新env后先做EP4（物理0–3）：

```bash
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=10.155.205.228 --master_port=29609 experiments/qwen3_vllm/moe_probe.py 2>&1 | tee "experiments/qwen3_vllm/results/moe-ep4-$(date +%Y%m%d-%H%M%S).log"
```

需最终`MOE_A2A_ROUTING_PASS ep=4`，不是只看某个worker初始化日志。结束全部probe进程后，做跨机EP8：228物理4–7、238物理0–3，各本地TP4，保持与D8一致。两机下面两个窗口要同时启动。

228容器：

```bash
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=10.155.205.228 --master_port=29610 experiments/qwen3_vllm/moe_probe.py 2>&1 | tee "experiments/qwen3_vllm/results/moe-ep8-228-$(date +%Y%m%d-%H%M%S).log"
```

238容器（注意node_rank=1）：

```bash
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=10.155.205.228 --master_port=29610 experiments/qwen3_vllm/moe_probe.py 2>&1 | tee "experiments/qwen3_vllm/results/moe-ep8-238-$(date +%Y%m%d-%H%M%S).log"
```

最终rank0需输出`MOE_A2A_ROUTING_PASS ep=8`，所有进程正常退出。若热点/空负载、NPU permute或all_to_all_single失败，保留两端最早错误，停止升到模型测试；不静默改成AllGather。rank失败时其他rank可能等到180秒超时。实际NPU路径仍由HCCL/环境决定，不能从“All-to-All”名称推定HCCS链路。

### 8.1.3 单机模型输出对照，然后PD4/PD8

先按第6节不加开关启动native单机模型，正常生成后，在228另开新容器窗口保存固定短prompt的确定性输出：

```bash
python experiments/qwen3_vllm/compare_outputs.py collect --output experiments/qwen3_vllm/results/solo-native-output.json
```

停止自己的native服务并确认worker退出，再启动All-to-All版本：

```bash
set -o pipefail
python -u experiments/qwen3_vllm/runtime.py serve --role solo-228 --moe-comm alltoall 2>&1 | tee "experiments/qwen3_vllm/results/solo-a2a-$(date +%Y%m%d-%H%M%S).log"
```

启动成功后在另一个已source的客户端容器窗口保存并比较（结果文件已存在时换新轮次名字，工具不会覆盖）：

```bash
python experiments/qwen3_vllm/compare_outputs.py collect --output experiments/qwen3_vllm/results/solo-alltoall-output.json
python experiments/qwen3_vllm/compare_outputs.py compare --native experiments/qwen3_vllm/results/solo-native-output.json --alltoall experiments/qwen3_vllm/results/solo-alltoall-output.json
```

`OUTPUT_SMOKE_MATCH`只是三个短prompt的回归，不是完整模型质量验收。AllGather和All-to-All的浮点归约顺序不同，greedy输出不一定逐字相同；出现差异必须检查logits/生成质量并扩充样本，不能忽略或当作拥塞收益。EP8还需在native/All-to-All的PD8 proxy上分别采集同样输出（collect加`--url http://10.155.205.228:30080`），覆盖DP2路径。

单机通过后正常退出，复用第7节PD4的P/D命令，两端均加`--moe-comm alltoall`；proxy、KV配置不变。PD4通过后按第8节升至PD8，三处模型命令明确如下（各在对应已source的容器窗口）：

```bash
# 228，P4窗口
python -u experiments/qwen3_vllm/runtime.py serve --role p-228 --layout pd8 --moe-comm alltoall
# 228，D的DP rank0窗口
python -u experiments/qwen3_vllm/runtime.py serve --role d8-228 --layout pd8 --moe-comm alltoall
# 238，D的DP rank1窗口
python -u experiments/qwen3_vllm/runtime.py serve --role d8-238 --layout pd8 --moe-comm alltoall
```

启动第8节同样的proxy，重复短请求、输出对照与workload。P的EP4和D的EP8是两个独立通信域，MoE不会横跨P与D组；正常KV流量仍跨这些角色。首轮不同时加DRAM Pool/CAM。

验收日志依次看：`QWEN_MOE_A2A_INSTALLED`、每rank的`QWEN_MOE_POLICY_SELECTED=ALLTOALL`及EP/TP ranks、`QWEN_MOE_A2A_DISPATCH/COMBINE ep=4/8`。日志只打印前两次（可能发生在warmup），还需第11节device profile确认真实请求窗口的payload All-to-All，不只看Host范围`QWEN_MOE_A2A::dispatch/combine`。A2A内部仍有专家计数元数据AllGather、TP输出恢复AllGather；看到它们不等于回退成AllGather MoE。

回退：停止自己的A2A服务，去掉`--moe-comm alltoall`或改为`--moe-comm native`后重新启动。适配包保留在容器中也不会默认生效，无需卸载，不修改权重。任何失败不自动回退，以免污染实验结论。

## 9 可选：小容量Host DRAM共享池接口验证

只有第7/8节原生PD通过才做。此项**不是原四卡HBM Pool**；不会占238卡4–7做独立池。每个推理worker贡献64MB Host DRAM、小型本地buffer16MB，只验证KV put/get与跨请求复用。禁止用它得出原HBM冲突结论。

先正常结束自己的proxy/P/D。238新容器窗口source后，启动master并保持前台：

```bash
python -u experiments/qwen3_vllm/runtime.py master
```

master用50088，管理元数据，不承载四张NPU的HBM。按第7节pd4重新启动P和D，但分别在命令末尾加`--pool-dram-qualification`：

```bash
python -u experiments/qwen3_vllm/runtime.py serve --role p-228 --layout pd4 --pool-dram-qualification
```

238：

```bash
python -u experiments/qwen3_vllm/runtime.py serve --role d-238 --layout pd4 --pool-dram-qualification
```

再启动`proxy --layout pd4`，重复第7节workload。若通过，可在第8节三处serve命令均加同一开关测试pd8；不能只给某一端加。

验收看真实远端put/get字节、读取成功、重用前缀、master对象/segment信息。`cached_tokens`或TTFT变小不足以证明命中远端池；可能是本地缓存/其他变化。默认失败策略fail，不把重算回退当作搬运成功。初期不启用SSD offload，不升级Mooncake到最新版。

## 10 CAM与独立HBM Pool的移植关口

新入口仅允许`QWEN_CAM_MODE=off`，CAM模式会明确拒绝启动。旧`SGLANG_CAM_DETOUR_MODE`不生效。保留UMDK的`feat/qwen3-cam-detour-build`和原All-to-All detour实现，不需要改其通信域来跑上述基线。

进入最终冲突实验前还需完成：

- 将Attention TP的AllReduce/ReduceScatter接入到固定vLLM通信接口；仅修改目标TP组，不拦截MoE/其他组；保持off / hccl-a2a / cam-detour三种对照。
- vLLM使用ASCEND_RT_VISIBLE_DEVICES选择卡，逻辑local rank会重编号。必须显式维护逻辑rank→物理卡→CAM rank映射，不能把D在228的逻辑0当物理0；它对应物理4。
- 原八卡CAM资源域需要P/D/Pool或中继owner全部提供窗口；在三角色启动和退出时保证域生命周期，不能只拉一个CAM worker。
- 按8.1节完成A2 EP4/EP8的All-to-All路由、真实FFN/输出、PD及profile验收；本次已有接入代码，但尚未实机验证。不为满足原生MC2的EP16条件强行改变P4+D8+Pool4布局。
- 若保留独立四卡HBM Pool，需验证可在A2注册/挂载NPU内存的存储后端及provider，再做远端恢复；标准DRAM Store示例不满足这个关口。
- 每个引擎接入点、SDK版本、图模式/融合路径变化均重新检查；MoE适配是显式worker内的固定源码运行时补丁，不是对所有服务生效的默认全局补丁。

这些关口未通过前，只能说“原生基线或DRAM缓存接口验证”，不能说“三流冲突实验已跑通”。

## 11 如何判断冲突与影响程度

### 11.1 当前能测什么

原生PD可比较不同负载下TTFT/TPOT及额外DRAM共享池带来的变化。8.1节验收通过后，可以再比较MoE AllGather与All-to-All，并研究All-to-All与正常P→D KV的干扰。但CAM、独立HBM Pool未接入，**不能单独证明CAM、MoE All-to-All与HBM KV三类流量争用同一条物理链路**。MoE后端性能差异本身也不等于冲突。

客户端指标采集已改为OpenAI `/v1/completions`，不再用SGLang `/generate`。SSE按完整行解析，EOF缺少完成标记计为失败。TPOT使用服务上报的completion_tokens估算，不把每个SSE包当一个token；reported_usage缺失则不计算TPOT。profile运行与纯性能运行分开，避免profile开销污染结果。

```bash
python experiments/qwen3_vllm/workload.py --requests 200 --rps 0.5 --concurrency 4 --prefix-tokens 1024 --output "experiments/qwen3_vllm/results/pd8-load-$(date +%Y%m%d-%H%M%S).jsonl"
python experiments/qwen3_vllm/workload.py --summarize experiments/qwen3_vllm/results/你的结果文件.jsonl
```

先低负载，再在显存和SLO允许时增加；需要输入+输出不超过max-model-len。每轮固定请求序列、模型、并行度、上下文、kernel和镜像，至少3轮。必须报告失败率、客户端排队时间、输出长度及吞吐，不只报告成功请求P99；200次仅为起步，尾延迟需更多样本。

### 11.2 最终实验的对照矩阵

完成第10节后，才组合三种CAM/TP算子模式（off / hccl-a2a / cam-detour）与额外Pool搬运（无 / 有）。这里的TP算子`hccl-a2a`是未来CAM对照，**不同于本次`--moe-comm alltoall`的MoE后端开关**。所有组固定同一种已验收MoE后端、正常P→D KV和请求负载；“Pool搬运无”不是“所有KV流量为零”。hccl-a2a控制CAM分块/归约/launch的额外开销。另做单独通信回放分离各类通信的时延，但不能把回放当成真实serving结果。

判断物理冲突必须同时具备：

1. 资源证据：明确每类流量的源/目标、真实物理链路/端口，排除不存在的绕路。
2. 时间证据：device执行时间发生重叠；Host提交时间不等于device执行时间，跨机时钟需校准。
3. 干扰证据：同一配置的并发组相对串行/错峰组有可重复的通信耗时/排队/链路利用率变化，排除计算、HBM容量、冷连接与额外launch的影响。

可在三处serve命令追加独立目录的`--profile-dir /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm/profiles/本轮角色`。它生成固定vLLM的torch profiler配置，由Ascend worker使用NPU profiler；启动后向**各引擎HTTP端点**（不是proxy）发送`POST /start_profile`和`POST /stop_profile`，只采集短窗口。D8两端都需采集，输出目录必须不同。SDK的外部RDMA完成信息不一定全出现在torch trace，还需其日志/打点和当前环境支持的链路计数。

影响量化：`通信slowdown = 并发时延 / 对照时延`；TTFT/TPOT分别报告P50/P99相对变化；吞吐变化保持相同到达率/请求长度，并记录SLO违反率。仅看到CPU时间重叠或网络吞吐升高，不足以断言拥塞，更不能给出预设影响百分比。

## 12 故障回传、退出与回退

| 失败阶段 | 回传重点 |
| --- | --- |
| 镜像检查 | 版本、源码SHA、ImageID/Architecture；不临时升级pip |
| TE初始化/读失败 | 首条Mooncake/ADXL错误、NPU IP/链路、probe源是否存活；不重下权重 |
| 权重加载/启动OOM | 最早traceback、该轮npu-smi、角色/卡号/预算；不是所有OOM都能靠缩上下文解决 |
| EP8建组失败 | 同轮两端D日志、DP rank/IP/接口；不只启动一端D |
| MoE A2A失败 | 适配包/source指纹、每rank策略日志、EP4/8 probe两端最早错误；不自动回退 |
| HTTPConnectionPool | 确认是228/238内部PD接口，还是误访问公网；公网权重查询与PD通信不是一回事 |
| Store put/get失败 | master/worker同轮日志、真实对象和segment信息；DRAM检查不算HBM验证 |

退出顺序：自己的proxy→自己的P/D→自己的probe→自己的master。各前台窗口Ctrl+C并确认自己的子进程退出；不用pkill python，不reset NPU，不清共享/dev/shm，不删除他人容器。新容器可保留，不占模型卡；重新进入仍用docker exec并source新env。

回退时先停新服务，确认卡释放，再进入保留的旧SGLang容器和历史指导。旧MemFabric join问题仍未解决，回退不等于修复。UMDK、模型权重、原实验代码均保留。

## 13 验证状态与依据

本轮助手只在无NPU环境核对固定源码、镜像manifest、命令构造并运行CPU/模拟回归；**没有在228/238构建或启动新镜像，没有验证A2 MoE All-to-All kernel、TE跨机读、模型质量、PD8、Store命中或三流性能**。只有你实机输出相应成功标记和证据，才进入下一关。

CPU检查入口：`python experiments/qwen3_vllm/test_cpu.py`与`python experiments/qwen3_vllm/test_moe_adapter.py`。它们不导入Torch或初始化NPU；A2无需为了这些测试安装另一套依赖。适配包测试包含源码指纹、模式/并行保护、空/非整除token补齐、原始长度恢复以及输出对照检查；CPU通过不替代NPU路由/模型验收。

固定源码依据：

- [vLLM-Ascend配套表](https://github.com/vllm-project/vllm-ascend/blob/v0.22.1rc1/docs/source/community/versioning_policy.md)、[A2 Dockerfile](https://github.com/vllm-project/vllm-ascend/blob/v0.22.1rc1/Dockerfile)。
- [Qwen3模型指导](https://github.com/vllm-project/vllm-ascend/blob/v0.22.1rc1/docs/source/tutorials/models/Qwen3-30B-A3B.md)、[A2 MoE选择逻辑](https://github.com/vllm-project/vllm-ascend/blob/v0.22.1rc1/vllm_ascend/ascend_forward_context.py)。
- [KV Pool与MultiConnector](https://github.com/vllm-project/vllm-ascend/blob/v0.22.1rc1/docs/source/user_guide/feature_guide/kv_pool.md)、[原生TE调用](https://github.com/vllm-project/vllm-ascend/blob/v0.22.1rc1/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py)。
- [Mooncake A2普通分配路径](https://github.com/kvcache-ai/Mooncake/blob/v0.3.9/mooncake-store/src/utils.cpp)、[TE Python读接口测试](https://github.com/kvcache-ai/Mooncake/blob/v0.3.9/mooncake-wheel/tests/transfer_engine_initiator_test.py)。
