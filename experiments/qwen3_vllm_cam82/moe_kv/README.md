# 原生 MoE 通信重放 + KV 搬运竞争实验

更新：2026-10-22。目标是测量固定 MoE 通信工作量下，加入 KV 搬运后的完成耗时变化。复用已跑通的 CANN8.2、vLLM0.10.2、vLLM-Ascend0.10.2rc1、Mooncake v0.3.9 镜像。本实验不需要重建 Docker、拉取权重、编译 CAM 或接入多路径算子。TP 保持原生，原先 `hccl-a2a` / `cam-detour` 分组不用于本实验。

实现已做 CPU 单元测试；尚需在 228/238 验证 NPU、双机通信和 profiler。不能把本文中的示例百分比、CPU 测试夹具或程序存在本身当成实验结果。

## 1 实验边界与分组

先在完整原生 D8 模型中采集实际通信描述，再停模型，使用独立程序重放。只保存 shape、dtype、各 rank 的 split 长度、顺序和源码指纹，不保存 prompt、模型权重、激活数据或用户 KV。

固定版 `TokenDispatcherWithAll2AllV` 的 BF16 路径为：专家计数 AllGather → Dispatch All-to-All-V → Combine All-to-All-V。采集和重放包含这三项；通过原生 `comm_utils` 调用 HCCL。数据填入可验证常数，通信形状和路由长度来自真实 capture。**不重放专家 GEMM、token permutation、Attention、TP 操作及它们之间的计算间隔**；这是通信干扰微基准，不能直接把其劣化率当成完整 MoE 层或 TPOT 劣化率。

| 组 | MoE | KV | 对照目的 |
|---|---|---|---|
| M | 固定 capture，重复相同次数 | 无传输 | MoE 独立基准 |
| K | 不执行通信 | 固定批次数、大小 | KV 独立基准 |
| MK | 与 M 完全相同 | 后台线程同时提交 | 并发干扰 |
| SERIAL | 与 M 完全相同 | 全部 MoE 完成后再传 KV | 错开重叠对照 |

每轮四组采用相同随机种子的随机顺序。所有组都预先创建 MoE 组、KV 引擎和相同缓冲区，预热建链，避免首次初始化只落入某一组。每次重放 collective 的结束事件会等待本操作所在 stream 完成；没有在每个操作后做全设备同步。这个测量方法会增加 CPU 提交间隔，各组一致，但不等价于原生推理的异步吞吐。

| 服务器 | 物理卡 | 独立实验中的作用 |
|---|---|---|
| 228 | 0～3 | KV 源，模拟 P 的 KV 存储，无模型计算 |
| 228 | 4～7 | MoE EP rank0～3 |
| 238 | 0～3 | MoE EP rank4～7，同时接收 KV |
| 238 | 4～7 | 不使用 |

共 12 个进程，228 启动 8 个、238 启动 4 个。MoE HCCL 组只有 8 个 D rank；12 rank Gloo 组仅用于元数据交换和阶段边界。KV 路径固定为 `228卡i → 238卡i`，i=0..3，读取方向为 D 拉 P 数据。控制网卡 `enp67s0f5` 不代表 NPU payload 也经过该主机网卡。

KV 调用与原生 connector 相同的 `TransferEngine.initialize(ip:0:npu_N, P2PHANDSHAKE, ascend, "")` 和 `batch_transfer_sync_read`，不用 AscendDirect。每个 rank 的每批传输有 48 层×K/V=96 个连续片段。每个片段使用独立的 2MiB 对齐注册区域，区域之间步长 2MiB，专用缓存约192MiB/rank（加对齐预留），不会写入在线模型的 KV。这里模拟连续 block 布局，尚未覆盖碎片化 block、服务端调度和 ACK 生命周期。

## 2 更新代码和适配包

**228、238都执行（相同命令，宿主机）：**

```bash
cd /ssd_2/yuanwenliu/sglang
git -c safe.directory=/ssd_2/yuanwenliu/sglang status --short --branch
git -c safe.directory=/ssd_2/yuanwenliu/sglang pull --ff-only origin feat/qwen3-cam-detour-conflict
docker exec -it qwen3-vllm-cam82 bash
```

已经在容器内时跳过 `docker exec`。如果仍有旧 P、D、proxy，请在各自前台窗口 Ctrl+C 停止自己的服务，再安装和重新拉起采集服务。不要停止他人的容器或进程。

**228、238都执行（相同命令，容器内）：**

```bash
cd /ssd_2/yuanwenliu/sglang
python -m pip install --no-deps --no-build-isolation --no-index --force-reinstall experiments/qwen3_vllm_cam82/adapter
python -c 'import importlib.metadata as m; from qwen_cam82 import moe_trace; assert m.version("qwen-vllm-cam82")=="0.1.4"; print("MOE_CAPTURE_ADAPTER_OK", moe_trace.__file__)'
python experiments/qwen3_vllm_cam82/moe_kv/test_replay.py
```

0.1.4 保留 0.1.3 的 KV 对齐修复，仅在显式设置 `QWEN_MOE_CAPTURE_DIR` 时启用采集；不设置时原生基线行为不变。仅拉代码或 source 环境不能更新已安装适配包。

## 3 从原生模型采集 EP8 通信

以下第一次使用统一目录 `capture-native-01`。重采时两机都改用一个新的相同名称，例如 `capture-native-02`；程序拒绝覆盖旧 trace。不要分别用两机当前时间生成不同目录名。采集只在 D 生效，排除 prefill 和 profile/dummy run，跳过前192个 decode MoE对，保存随后48对。这里“对”指一次 Dispatch/Combine，不是请求数。

**228 窗口1：P，宿主机进入容器后执行：**

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
unset QWEN_MOE_CAPTURE_DIR
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role p-228 --tp-mode off 2>&1 | tee experiments/qwen3_vllm_cam82/results/p-capture-native-01.log
```

**228 窗口2：D：**

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
export QWEN_MOE_CAPTURE_DIR=/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-native-01
export QWEN_MOE_CAPTURE_SKIP_PAIRS=192 QWEN_MOE_CAPTURE_PAIRS=48
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-228 --tp-mode off 2>&1 | tee experiments/qwen3_vllm_cam82/results/d228-capture-native-01.log
```

**238 窗口1：D：**

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
export QWEN_MOE_CAPTURE_DIR=/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-native-01
export QWEN_MOE_CAPTURE_SKIP_PAIRS=192 QWEN_MOE_CAPTURE_PAIRS=48
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-238 --tp-mode off 2>&1 | tee experiments/qwen3_vllm_cam82/results/d238-capture-native-01.log
```

保留三个服务窗口。**228 新窗口（容器内）**检查两侧 D，再向两个 API 同时发送一次长输出。这里直连 D，让两个 DP 实例进入 decode，不依赖 proxy 路由；采集阶段无需启动 proxy：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
curl --noproxy '*' --connect-timeout 3 --max-time 10 -fsS http://10.155.205.228:30011/health -w '\nHTTP=%{http_code}\n'
curl --noproxy '*' --connect-timeout 3 --max-time 10 -fsS http://10.155.205.238:30011/health -w '\nHTTP=%{http_code}\n'
python experiments/qwen3_vllm_cam82/moe_kv/capture_load.py
```

需要两个 HTTP200 和两个请求正常完成。D 日志出现 `MOE_CAPTURE_READY`：228 EP0..3，238 EP4..7。采集会增加观察开销，采集期间的延迟不作为性能结果。

## 4 汇总 trace，检查通信顺序

**228、238都先执行（相同命令，宿主机）：** 建立当前用户可写的交换目录。容器生成的结果目录可能属于root，因此先传到这个目录，再在容器内复制，不改变整个仓库的属主。

```bash
mkdir -p /ssd_2/yuanwenliu/moe-kv-exchange/capture-native-01
```

**再在 228 的宿主机窗口执行：** 从238取四份元数据；SSH提示时使用实验机账号密码。无需 rsync，不连接开发机。

```bash
scp yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-native-01/ep4.json /ssd_2/yuanwenliu/moe-kv-exchange/capture-native-01/
scp yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-native-01/ep5.json /ssd_2/yuanwenliu/moe-kv-exchange/capture-native-01/
scp yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-native-01/ep6.json /ssd_2/yuanwenliu/moe-kv-exchange/capture-native-01/
scp yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-native-01/ep7.json /ssd_2/yuanwenliu/moe-kv-exchange/capture-native-01/
scp /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-native-01/ep0.json /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-native-01/ep1.json /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-native-01/ep2.json /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-native-01/ep3.json yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/moe-kv-exchange/capture-native-01/
```

**228、238都执行（相同命令，宿主机）：** 把交换目录中的四个远端rank文件放入容器。`docker cp`不要求交换目录本身已挂载；目录名必须与本轮一致，勿混入旧trace。

```bash
docker cp /ssd_2/yuanwenliu/moe-kv-exchange/capture-native-01/. qwen3-vllm-cam82:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-native-01/
docker exec -it qwen3-vllm-cam82 bash
```

**228、238都执行（相同命令，容器内）：**

```bash
cd /ssd_2/yuanwenliu/sglang
python experiments/qwen3_vllm_cam82/moe_kv/spec.py --trace-dir experiments/qwen3_vllm_cam82/results/capture-native-01
```

应有相同 `trace_sha256`、`pairs=48`，且 `cross_host_payload_bytes_per_window>0`。校验每一项通信各源/目的 split 转置相等、Dispatch/Combine互逆、源码指纹及物理卡映射一致。出现 mismatch 时停止：不能手改 splits 使检查通过。请在两机使用相同采集参数、同时重启 D、无其他请求条件下，换新目录重采；只缺文件时先确认请求正常完成、日志出现 READY。

校验通过后，回到 P、两个 D、可能仍在运行的 proxy 前台窗口，分别 Ctrl+C 停止。用 `npu-smi info` 确认自己的模型进程已退出。独立重放与原模型不能同时占用相同卡。238剩余4卡不用于本实验；其他任务若共享跨机网络，也应记录，不能把其影响归因于 KV。

## 5 两机独立实验：先 smoke

进入已有容器，环境分别设置。主程序自己启动本机子进程，不要外包一层 torchrun。

**228 实验窗口：**

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
unset QWEN_MOE_CAPTURE_DIR
ss -lnt 'sport = :29480'
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 228 --trace-dir experiments/qwen3_vllm_cam82/results/capture-native-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id smoke-01 --iterations 1 --kv-transfers 8 --repeats 1 --prefix-tokens 256
```

**238 实验窗口：** 在228开始等待后及时执行，不必等228结束：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
unset QWEN_MOE_CAPTURE_DIR
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 238 --trace-dir experiments/qwen3_vllm_cam82/results/capture-native-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id smoke-01 --iterations 1 --kv-transfers 8 --repeats 1 --prefix-tokens 256
```

29480须空闲；若占用，选另一个空闲端口并在**两机命令都加相同** `--master-port`，不要杀不明进程。首次连接还会使用 Mooncake/HCCL 自动建立的通信端口，需要两机原生通信网络互通。

期望：228输出四个 `MOE_KV_PHASE_DONE`，最终228有8个、238有4个 `MOE_KV_COMPLETE`。程序先检查 MoE 的每个接收来源的数据，再检查 KV 全部传输字节。任一失败整个run无效，不允许只删除失败rank继续分析。一个host失败后，应在另一host自己的实验窗口 Ctrl+C，停止本轮；下一次使用新run-id，避免覆盖失败证据。

`--dry-run` 可先检查trace和数据量，不初始化NPU。smoke只验证通信、正确性、日志及分析链路，不用于作性能结论。

## 6 正式运行

沿用第5节已进入并source环境的两个实验窗口，先执行228，再及时执行238。参数除了 `--host` 必须一致。**228：**

```bash
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 228 --trace-dir experiments/qwen3_vllm_cam82/results/capture-native-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id native-kv1024-gap0-01 --iterations 20 --kv-transfers 200 --repeats 5 --prefix-tokens 1024 --kv-gap-ms 0
```

**238：**

```bash
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 238 --trace-dir experiments/qwen3_vllm_cam82/results/capture-native-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id native-kv1024-gap0-01 --iterations 20 --kv-transfers 200 --repeats 5 --prefix-tokens 1024 --kv-gap-ms 0
```

每组20×48=960个 MoE对，每对3项通信，重复5轮。每个接收rank每组200批KV，4rank总数据量为 `200×96MiB`。统计的是200个四rank大小等价的KV载荷，各rank独立提交，不宣称是200个真实服务请求。

`--kv-gap-ms` 是每批完成后的休眠间隔，**不是保证的带宽或25%/50%负载比例**。先看 K 组实测吞吐再调整。需要扫描负载时，两机使用相同的新run-id，分别设置256/1024/1536前缀或0/2/10ms间隔；不同时改变MoE trace和KV参数。所有测试均使用同一traceSHA，不通过人为增加MoE token数宣称当前原生负载发生冲突。

默认参数不保证整个MoE窗口都有KV覆盖。如果分析显示 `host_submission_overlap_fraction` 很低，检查时间线、增加两机相同的 `--kv-transfers`，或减少相同的 `--iterations`，再用新run-id运行。等待/错峰组会更慢或更快均应保留，不能只选符合预期的轮次。

## 7 汇总结果、分析

**228宿主机窗口：** 以正式run为例，拉取238的4份结果；只复制这轮的rank文件。smoke分析时把下面同名run目录统一换成 `smoke-01`。

```bash
mkdir -p /ssd_2/yuanwenliu/moe-kv-exchange/native-kv1024-gap0-01
scp yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/native-kv1024-gap0-01/rank08.jsonl /ssd_2/yuanwenliu/moe-kv-exchange/native-kv1024-gap0-01/
scp yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/native-kv1024-gap0-01/rank09.jsonl /ssd_2/yuanwenliu/moe-kv-exchange/native-kv1024-gap0-01/
scp yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/native-kv1024-gap0-01/rank10.jsonl /ssd_2/yuanwenliu/moe-kv-exchange/native-kv1024-gap0-01/
scp yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/native-kv1024-gap0-01/rank11.jsonl /ssd_2/yuanwenliu/moe-kv-exchange/native-kv1024-gap0-01/
docker cp /ssd_2/yuanwenliu/moe-kv-exchange/native-kv1024-gap0-01/. qwen3-vllm-cam82:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/native-kv1024-gap0-01/
docker exec -it qwen3-vllm-cam82 bash
```

**228容器内：**

```bash
cd /ssd_2/yuanwenliu/sglang
python experiments/qwen3_vllm_cam82/moe_kv/analyze.py --run-dir experiments/qwen3_vllm_cam82/results/moe-kv/native-kv1024-gap0-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv-analysis/native-kv1024-gap0-01
```

CPU分析不需要模型或NPU。输出目录必须是新的；分析已存在结果时另选目录，不覆盖旧报告。

- `report.md`：每轮配对后的劣化率中位数及最小～最大、注意事项。
- `paired_rounds.csv`：每轮Dispatch/Combine、KV变化率、错峰恢复、主机提交重叠率。
- `analysis.json`：完整配置、traceSHA、分组P50/P95/P99、KV有效吞吐、所有配对结果。

定义：`wall_ms` 为提交至本操作stream完成的本机单调时钟耗时，含CPU/排队等待；`stream_ms` 是同一stream的设备事件间隔，仍可能含依赖等待，不是纯链路耗时。`critical` 是同一次调用8个rank耗时最大值；不是相加，也不是跨主机时钟做差。`phase_barrier_upper_ms` 含阶段末同步开销，只是整轮完成耗时的观测上界。

主指标是每轮 MK/M 的中位数比值减一；另输出 MK发生主机提交重叠的样本与M相同iteration/pair/op的配对结果，避免取不同MoE形状比较。KV比较 MK/K，所有KV接收都在238，因此聚合吞吐用238各进程的共同单调时钟跨度计算，包含设置的gap。P99的样本不是独立请求，不能把多个rank或相邻层当作大量独立统计样本；优先看多轮变化是否稳定。

## 8 短 profiling 与结论边界

性能运行结束后，另用短profile运行确认设备通信是否重叠。输出与性能run分开，仍需两机共同启动。

**228：**

```bash
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 228 --trace-dir experiments/qwen3_vllm_cam82/results/capture-native-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id profile-01 --iterations 1 --kv-transfers 8 --repeats 1 --prefix-tokens 1024 --profile
```

**238：**

```bash
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 238 --trace-dir experiments/qwen3_vllm_cam82/results/capture-native-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id profile-01 --iterations 1 --kv-transfers 8 --repeats 1 --prefix-tokens 1024 --profile
```

每个D rank输出 `profile-rankXX`，含 `moe_replay/iteration/pair/op` 主机标记及可用的NPU活动。查看238接收卡上MoE与Mooncake/HCCL搬运的时间区间；某些SDK下profiler不暴露完整传输细节，缺少事件时只能报告“缺设备侧证据”，不能用两条host线程代替链路重叠。必要时结合环境可用的设备网络计数和物理拓扑；本程序不会改网络配置或伪造端口利用率。

结论逐级表述：

1. MK稳定变慢、SERIAL单操作接近M/K：支持“并发执行导致通信性能干扰”。
2. 设备trace也证实相同时段两类传输：支持“设备侧并发时发生干扰”。
3. 进一步确认共享路径、端口负载/拥塞信号：才定位到“该物理链路竞争”。否则还可能是HBM、DMA或运行时资源竞争。
4. 未观察到重叠或稳定劣化也是有效结果，不能强行证明拥塞。微基准结果不能直接推出完整服务TPOT变化，下一阶段才做完整P/D实验。

## 9 常见失败与恢复

- `FileExistsError`：保留原文件，两个host统一换新capture目录或run-id。
- trace不完整/分片不匹配：不要启动HCCL重放；检查两侧采集日志和相同窗口。脚本会在初始化前拒绝不兼容trace。
- 主机参数不一致：12个rank交换配置指纹后报错；检查run-id、iterations、KV参数、profile及traceSHA。
- KV初始化、注册或传输ret非0：该轮作废。保存对应rank日志；不要关闭对齐检查、改变协议或在失败数据上报性能。
- HCCL长时间等待：确认两机都启动、模型已退出、端口空闲及原来的设备通信网络可用。默认超时600秒；一端失败后停止另一端自己的该轮任务。
- `MOE_KV_COMPLETE`齐全但无明显变慢：先看重叠覆盖率、设备trace、当前小batch MoE载荷；不能仅因无劣化就断言代码错误。
- 恢复原生服务：确认重放进程退出，两个host `unset QWEN_MOE_CAPTURE_DIR`，按主README第7节拉起P/D/proxy。无需回退0.1.4。
