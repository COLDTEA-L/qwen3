# 128请求 MoE trace + 2048-token KV 搬运实验

128请求结果已由用户提供。全部已报数据见 [RESULTS.md](RESULTS.md)，后续真实服务计划见 [SERVING_PLAN.md](SERVING_PLAN.md)。本文件保留128请求复现实验的完整命令，重复运行须换新run-id；后续优先做真实服务重叠验证，不自动继续加到256请求。

更新：2026-10-22。**已完成64请求实验者：两机按第2节更新代码，再从第4节重新采集128请求trace，完成第5节校验后进入第6节。不能复用64请求trace来宣称128请求实验。** 本次每个D API发送128个请求（共256个），KV保持2048 tokens；不需重建Docker、重编算子或重装已安装的0.1.4适配包。第3节是历史对照，已完成者跳过。

## 1 要回答的问题

已有 `native-kv1024-gap0-01`：每对 Dispatch+Combine 全 EP8 双向跨机载荷均值 256.5KiB，主机提交重叠率约49%，重叠样本 Dispatch/Combine 变化多数约2%～4%。它支持“小载荷下干扰较小”，不支持“大载荷一定不受影响”，也不能证明物理链路拥塞。

用户随后提供 `native-kv1024-gap0-02` 报告：Dispatch +1.49%、Combine +1.47%、KV +0.42%；错峰 Dispatch/Combine 的轮次变化范围分别为 -0.82%～+2.53%、-1.41%～+2.56%。不再出现低于50%主机提交重叠率的警告，但具体覆盖率尚须从analysis.json读取，不能当成100%。目前未见明显放大，且与错峰波动量级接近；不能据此认定显著竞争。该轮若已完成，直接从第4节开展新载荷采集，不必重跑第3节。

更可能出现较大影响的条件是：MoE 消息足够大、KV 持续覆盖 MoE 窗口，且两者确实共享繁忙的设备端口/链路或 DMA/HBM 资源。增加请求并发可增加真实 MoE 载荷；增加 KV 次数只延长持续时间，不提高单批大小或并发深度。若瓶颈是固定提交开销、硬件有余量、路由不同或运行时串行，影响仍可能很小。

本轮不改网络配置，不增添 TP 分解，不为了得到正数而筛选结果。

8并发实测：每rank Dispatch输入16行，跨机载荷均值510.83KiB/对，主机提交重叠率100%；Dispatch +0.90%、Combine +0.73%、KV +0.85%，未观察到明显劣化。保留这轮作为对照。随后32请求的实测载荷达到2146.83KiB/对，主机提交重叠率100%，Dispatch +1.63%、Combine +1.65%、KV +1.85%，影响仍较小。随后 `dp32-kv2048-n500-01` 实测Dispatch +3.73%、Combine +4.85%、KV +1.97%，错峰Dispatch +0.25%、Combine -0.27%。保留原始结果。64请求实测跨机载荷4211.50KiB/对，Dispatch +4.43%、Combine +6.88%、KV +1.99%；错峰KV批次相对K为+0.17%（-0.42%～+0.42%，不含等待MoE）。本轮只把目标请求并发从64提高到128，重新采集真实通信；KV仍2048 tokens、500批、gap=0、20个重放窗口、5轮，不开启profiling。目录中的`dp32`/`dp64`/`dp128`仅表示每个DP实例的客户端并发数，**不表示DP并行度32、64或128：配置仍是DP2、TP4、EP8，D总共8张卡**。

| 测试 | MoE trace | KV 前缀 | 每接收 rank 的 KV 次数 | 目的 |
|---|---|---:|---:|---|
| 已有结果 | capture-native-01 | 1024 | 200 | 保留原始低负载结果 |
| 覆盖对照 | capture-native-01 | 1024 | 500 | 只延长 KV 窗口 |
| 已完成8并发对照 | capture-dp8-01 | 1024 | 500 | 保留既有结果，不覆盖 |
| 已完成32请求对照 | capture-dp32-01 | 1024 | 500 | 保留 `dp32-kv1024-n500-01` |
| 已完成32请求大KV对照 | capture-dp32-01 | 2048 | 500 | 保留 `dp32-kv2048-n500-01` |
| 已完成64请求对照 | capture-dp64-01 | 2048 | 500 | 保留 `dp64-kv2048-n500-01` |
| 本轮增加MoE批量 | capture-dp128-01（重新采集） | 2048 | 500 | run-id为 `dp128-kv2048-n500-01` |

每种测试内部仍有 M/K/MK/SERIAL 四组，iterations=20、repeats=5。**比较各自 MK/M，不把新 trace 的绝对耗时除以旧 trace 的 M。** 如果两个 trace 的重叠率差很多，先延长不足者的 KV 覆盖，再比较干扰程度。90%可作为覆盖目标，但仍须检查设备 trace，不能把 host 覆盖当作设备并发。

1024-token KV 每批24MiB/rank，4接收rank合计96MiB；500批合计46.875GiB（每个 K/MK/SERIAL 阶段、每轮）。本轮2048-token KV每批48MiB/rank、四rank合计192MiB，500批合计93.75GiB（每个K/MK/SERIAL阶段、每轮）。每rank仍是一个同步搬运调用完成后再提交下一批，不是500并发请求，不会分配500份缓存。与1024-token对照相比仍复用相同注册区域，每片传输长度由256KiB增至512KiB；32、64与128请求的2048-token实验均保持512KiB/片，不扩大192MiB/rank的有效缓冲区（另有对齐预留）。总搬运量翻倍不保证瞬时带宽翻倍，应结合K组实测吞吐判断。

这里的2048仅是独立重放中模拟的KV前缀长度，**不修改采集时每请求生成128 token；本轮每D API采集并发为128**。

## 2 更新代码

228、238都执行（相同命令，宿主机）。先检查本地修改；有冲突不要覆盖。

```bash
cd /ssd_2/yuanwenliu/sglang
git -c safe.directory=/ssd_2/yuanwenliu/sglang status --short --branch
git -c safe.directory=/ssd_2/yuanwenliu/sglang pull --ff-only origin feat/qwen3-cam-detour-conflict
```

已经装了采集适配包0.1.4时，本次只更新外部脚本和文档，**不需要重建镜像、重编算子或重装适配包**。若还没有0.1.4，先完成主实验README第2节。所有下面的目录/run-id均为新名称；若已存在，保留旧结果，统一选另一名称。

## 3 先测旧 trace 的500次覆盖对照

确认自己的模型、proxy和上一轮重放都已退出，再运行。已在容器时跳过docker exec。228启动后立即执行238，不等待228完成。

228实验窗口：

```bash
docker exec -it qwen3-vllm-cam82 bash
cd /ssd_2/yuanwenliu/sglang
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
unset QWEN_MOE_CAPTURE_DIR
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 228 --trace-dir experiments/qwen3_vllm_cam82/results/capture-native-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id native-kv1024-gap0-02 --iterations 20 --kv-transfers 500 --repeats 5 --prefix-tokens 1024 --kv-gap-ms 0
```

238实验窗口：

```bash
docker exec -it qwen3-vllm-cam82 bash
cd /ssd_2/yuanwenliu/sglang
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
unset QWEN_MOE_CAPTURE_DIR
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 238 --trace-dir experiments/qwen3_vllm_cam82/results/capture-native-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id native-kv1024-gap0-02 --iterations 20 --kv-transfers 500 --repeats 5 --prefix-tokens 1024 --kv-gap-ms 0
```

完成后，228宿主机执行：

```bash
mkdir -p /ssd_2/yuanwenliu/moe-kv-exchange/native-kv1024-gap0-02
scp 'yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/native-kv1024-gap0-02/rank*.jsonl' /ssd_2/yuanwenliu/moe-kv-exchange/native-kv1024-gap0-02/
docker cp /ssd_2/yuanwenliu/moe-kv-exchange/native-kv1024-gap0-02/. qwen3-vllm-cam82:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/native-kv1024-gap0-02/
docker exec qwen3-vllm-cam82 python /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/moe_kv/analyze.py --run-dir /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/native-kv1024-gap0-02 --output-dir /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv-analysis/native-kv1024-gap0-02
```

新版报告直接列出每轮覆盖率和重叠样本劣化，无需另写JSON查询。若500仍不足，保留这轮，下一轮两机同改1000、用新run-id；不是无限增加来“制造”劣化。

## 4 采集每个 D API 128个并发请求的实际通信

确认之前的两机重放/profiling已完整退出。如果旧服务仍在运行：在228窗口1停止P、228窗口2停止D228、238窗口1停止D238，在proxy原窗口停止proxy，均在自己服务的前台窗口按Ctrl+C；若已回到shell提示符，不需要重复停止。确认自己的workers退出，不要杀其他人的进程。本阶段直连两个D API，只需拉起D8，**不需要P和proxy**；D本地处理这些请求的prefill，再采集decode。不能把这种采集请求的端到端时间当作P/D性能。

保持TP4、DP2、EP8、原生off。`--capture-load-test`仅允许D原生采集时把max-num-seqs提高到最多128；普通基线默认上限4不变。KV预留仍2048×64token，不扩大缓存。本轮显式设为128，并显式设置 `--max-model-len 1024`。容量检查不保证算子workspace不OOM；若OOM或请求失败，保留失败日志，停止本轮，不混入成功样本。本轮不自动减少并发或扩大显存预算。

128×2048会超过现有131072-token KV容量，因此只将本档采集服务的最大上下文从2048降到1024；128×1024恰好满足容量检查，仍不保证workspace不OOM。采集prompt很短且只生成128 tokens，无需2048上下文。它不改变第6节独立重放的 `--prefix-tokens 2048`，也不改变KV物理block-size=64或num-gpu-blocks-override=2048。该服务上限调整应记录在实验条件中。以下启动命令会重新拉起两个D；旧进程不会因git pull或source自动更新参数。已在容器内则跳过docker exec。

228服务窗口：

```bash
docker exec -it qwen3-vllm-cam82 bash
cd /ssd_2/yuanwenliu/sglang
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
export QWEN_MOE_CAPTURE_DIR=/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp128-01
export QWEN_MOE_CAPTURE_SKIP_PAIRS=768 QWEN_MOE_CAPTURE_PAIRS=48
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-228 --tp-mode off --capture-load-test --max-num-seqs 128 --max-model-len 1024 2>&1 | tee experiments/qwen3_vllm_cam82/results/d228-capture-dp128-01.log
```

238服务窗口（及时启动，不等待228服务ready）：

```bash
docker exec -it qwen3-vllm-cam82 bash
cd /ssd_2/yuanwenliu/sglang
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
export QWEN_MOE_CAPTURE_DIR=/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp128-01
export QWEN_MOE_CAPTURE_SKIP_PAIRS=768 QWEN_MOE_CAPTURE_PAIRS=48
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-238 --tp-mode off --capture-load-test --max-num-seqs 128 --max-model-len 1024 2>&1 | tee experiments/qwen3_vllm_cam82/results/d238-capture-dp128-01.log
```

保持两服务窗口。768对相当于跳过更多decode通信，让并发请求有机会进入稳定批次，不保证采集时batch恰为128。228新窗口：

```bash
docker exec -it qwen3-vllm-cam82 bash
cd /ssd_2/yuanwenliu/sglang
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
curl --noproxy '*' --connect-timeout 3 --max-time 10 -fsS http://10.155.205.228:30011/health -w '\nHTTP=%{http_code}\n'
curl --noproxy '*' --connect-timeout 3 --max-time 10 -fsS http://10.155.205.238:30011/health -w '\nHTTP=%{http_code}\n'
python experiments/qwen3_vllm_cam82/moe_kv/capture_load.py --requests-per-dp 128 --tokens 128
```

两个health均200后才执行负载。期望256个请求完成，最后输出：

```text
CAPTURE_LOAD_COMPLETE {"requests": 256, "requests_per_dp": 128, "output_tokens_requested": 128}
```

每个D日志出现4个`MOE_CAPTURE_READY`，两机合计EP0～7各一份。脚本非流式，请求完成才打印，开始时安静并不等于卡死。别再发其他请求。采集只记录真实路由，不人为放大split；任何请求失败或trace不完整都先排查，不进入性能测试。

## 5 两机合并新 trace 并核实实际载荷

228、238都执行（相同命令，宿主机）：

```bash
mkdir -p /ssd_2/yuanwenliu/moe-kv-exchange/capture-dp128-01
```

然后仅228宿主机执行：

```bash
scp 'yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp128-01/ep[4-7].json' /ssd_2/yuanwenliu/moe-kv-exchange/capture-dp128-01/
scp /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp128-01/ep0.json /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp128-01/ep1.json /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp128-01/ep2.json /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp128-01/ep3.json yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/moe-kv-exchange/capture-dp128-01/
```

228、238都执行（相同命令，宿主机）：

```bash
docker cp /ssd_2/yuanwenliu/moe-kv-exchange/capture-dp128-01/. qwen3-vllm-cam82:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp128-01/
docker exec qwen3-vllm-cam82 python /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/moe_kv/spec.py --trace-dir /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp128-01
```

必须两机相同trace_sha256、split校验通过。看`cross_host_kib_per_dispatch_combine_pair`是否大于64请求对照的4211.50KiB，以及`dispatch_input_rows_per_rank_min/max`。这些行是专家路由后的行数，不直接等于请求batch。根据当前TP4切分和top-8路由，若采到了每个D实例128个token的稳定decode批次，预计每rank有`128 / 4 × 8 = 256`行（64请求稳定批次预期128行）；这是诊断预期，不是保证。跨机载荷取决于实际专家路由，不要求恰好2倍。若行数仍与64请求trace相同，应先检查采集窗口和实际调度，不能仅凭命令里写128就宣称载荷扩大。若载荷没有增加，不能称其为大消息实验；先检查调度、采集窗口，不继续盲目重放。如果分片序列不匹配，换新目录两机同步重启采集，禁止手改trace。

验证后在两个D前台窗口Ctrl+C，确认自己的workers退出，才能下一步。

## 6 使用128请求 trace，保持KV前缀2048

首次128请求实验必须先完成第4～5节；只有两机已有且验证过相同的 `capture-dp128-01` 后才能从本节开始。上一轮重放/profiling已退出，自己的P/D/proxy均已停止。只跑独立重放，不重新启动模型。已在容器内则跳过docker exec；本节给出两机完整命令。保留32及64请求结果。本节固定示例run-id只用于首次运行；复测时两机及下方汇总、分析路径统一改成 `dp128-kv2048-n500-02` 等未用过的新名称。`run.py` 的rank文件使用排他创建、`analyze.py` 的输出目录也必须不存在；FileExistsError是防覆盖保护，不要删除旧结果或混用两次实验文件。

228实验窗口：

```bash
docker exec -it qwen3-vllm-cam82 bash
cd /ssd_2/yuanwenliu/sglang
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
unset QWEN_MOE_CAPTURE_DIR
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 228 --trace-dir experiments/qwen3_vllm_cam82/results/capture-dp128-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id dp128-kv2048-n500-01 --iterations 20 --kv-transfers 500 --repeats 5 --prefix-tokens 2048 --kv-gap-ms 0
```

238实验窗口（紧接228启动）：

```bash
docker exec -it qwen3-vllm-cam82 bash
cd /ssd_2/yuanwenliu/sglang
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
unset QWEN_MOE_CAPTURE_DIR
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 238 --trace-dir experiments/qwen3_vllm_cam82/results/capture-dp128-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id dp128-kv2048-n500-01 --iterations 20 --kv-transfers 500 --repeats 5 --prefix-tokens 2048 --kv-gap-ms 0
```

全部完成后，228宿主机：

```bash
mkdir -p /ssd_2/yuanwenliu/moe-kv-exchange/dp128-kv2048-n500-01
scp 'yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/dp128-kv2048-n500-01/rank*.jsonl' /ssd_2/yuanwenliu/moe-kv-exchange/dp128-kv2048-n500-01/
docker cp /ssd_2/yuanwenliu/moe-kv-exchange/dp128-kv2048-n500-01/. qwen3-vllm-cam82:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/dp128-kv2048-n500-01/
docker exec qwen3-vllm-cam82 python /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/moe_kv/analyze.py --run-dir /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/dp128-kv2048-n500-01 --output-dir /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv-analysis/dp128-kv2048-n500-01
```

## 7 如何判读、是否继续加压

1. 先检查完整性、KV正确性、真实载荷增加和覆盖率。失败run不能用于性能结论。
2. 看每轮MK/M、KV的MK/K、重叠样本配对变化及SERIAL恢复。变慢必须超过同条件重复波动；不设必须达到10%之类的结论门槛。
3. 若更大载荷仍仅变化几个百分点，这是有效结果。确认设备侧重叠和共享资源后，才能讨论硬件余量或竞争不敏感。
4. 本轮改变MoE采集请求批量，保持KV大小2048 tokens、搬运次数、重放窗口数、轮数和通信算法不变。128请求实验先与自身M/K比较，再对照64请求实验的劣化率；不同trace的绝对耗时不能直接相除来计算竞争影响。同一轮两机trace_sha256必须相同，64与128请求的trace无需相同。覆盖率和实际载荷变化须一并报告。
5. 出现稳定干扰后，另做短profiling（主README第8节，使用对应新trace），区分CPU提交等待、设备串行、DMA/HBM竞争与物理链路拥塞。微基准不能直接推出TPOT劣化。

8、32及64请求、1024及2048-token KV已有用户实测；本次128请求的采集和重放仍需228/238验证。CPU测试不代表32GB NPU容量或通信性能已验证。恢复普通P/D时清除采集环境，不加`--capture-load-test`，按主部署文档运行。

## 8 第4、5步与“并发”的含义

本文件的第4步是真实模型采集：只启动D8（DP2、TP4、EP8），不启动P/proxy；向两个D API各发128个并发请求，记录decode阶段实际collective形状、dtype和各目的rank的split。请求并发不保证引擎批次恰好128；采集时延不用于性能结论。第5步仅交换EP0～7文件，校验顺序、split和源码指纹并核实载荷，不启动模型、不执行通信重放。

注意另一份 `moe_kv/README.md` 编号不同：它的第4步是汇总校验trace，第5步是短smoke（检查通信连通、载荷正确性和输出完整性，不用于性能结论）。不要把两份文档的编号混用。

正式重放期间所有模型服务停止，采用12个实验进程：

- 228物理卡0～3：4个KV源，只提供注册缓冲区，不运行P模型。
- 228物理卡4～7 + 238物理卡0～3：8个MoE rank，跨机EP8。
- 238物理卡0～3同时作为4个KV接收rank，分别从228物理卡0～3读取。

| 组别 | MoE通信循环 | KV搬运循环 |
|---|---|---|
| M | 8 rank运行 | 不运行 |
| K | 不运行 | 4个接收rank运行 |
| MK | 8 rank运行；238接收rank的主线程负责MoE | 238各接收rank的后台线程独立搬运，与MoE并行推进 |
| SERIAL | 先运行，与M相同 | 等所有MoE rank完成后再搬运；用于错峰对照 |

每个MoE rank按trace顺序调用“计数AllGather → Dispatch AllToAllV → Combine AllToAllV”，当前48对/窗口、iterations=20，即每个M/MK/SERIAL阶段每rank各960次Dispatch和Combine（另有计数AllGather）。每个操作等待本stream完成才进入下一个，不是同时提交960个算子；这里没有专家GEMM、Attention、TP通信或完整模型计算，也没有真实计算间隔。

MK中每个KV接收rank连续完成500次同步批量读取，gap=0表示批次之间不主动休眠，不表示没有提交开销。每批模拟2048-token全层KV的一个TP分片（48MiB/rank），4个接收rank并行、每rank在途批次数最多1；复用固定缓冲区，不是500个请求或500份缓存。96段/批的内部调度由搬运库决定。

所以可概括为“KV连续搬运，同时D卡循环执行MoE通信”，但两者都是有界循环，不是无限运行或完整在线serving。各自完成后等阶段结束；不能保证整个MoE区间始终有KV覆盖。每轮含四组，组序按seed打乱，repeats=5；覆盖率和设备profiling仍需验证。只增加kv-transfers会延长窗口，不提高每rank的在途并发。此次128请求修改只增大MoE真实消息形状，不增加KV并发深度。
