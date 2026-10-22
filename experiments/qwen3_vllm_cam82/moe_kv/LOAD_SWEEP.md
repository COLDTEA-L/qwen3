# 低负载之后：KV 覆盖与真实 MoE 载荷扫描

## 1 要回答的问题

已有 `native-kv1024-gap0-01`：每对 Dispatch+Combine 全 EP8 双向跨机载荷均值 256.5KiB，主机提交重叠率约49%，重叠样本 Dispatch/Combine 变化多数约2%～4%。它支持“小载荷下干扰较小”，不支持“大载荷一定不受影响”，也不能证明物理链路拥塞。

用户随后提供 `native-kv1024-gap0-02` 报告：Dispatch +1.49%、Combine +1.47%、KV +0.42%；错峰 Dispatch/Combine 的轮次变化范围分别为 -0.82%～+2.53%、-1.41%～+2.56%。不再出现低于50%主机提交重叠率的警告，但具体覆盖率尚须从analysis.json读取，不能当成100%。目前未见明显放大，且与错峰波动量级接近；不能据此认定显著竞争。该轮若已完成，直接从第4节开展新载荷采集，不必重跑第3节。

更可能出现较大影响的条件是：MoE 消息足够大、KV 持续覆盖 MoE 窗口，且两者确实共享繁忙的设备端口/链路或 DMA/HBM 资源。增加请求并发可增加真实 MoE 载荷；增加 KV 次数只延长持续时间，不提高单批大小或并发深度。若瓶颈是固定提交开销、硬件有余量、路由不同或运行时串行，影响仍可能很小。

本轮不改网络配置，不增添 TP 分解，不为了得到正数而筛选结果。

| 测试 | MoE trace | KV 前缀 | 每接收 rank 的 KV 次数 | 目的 |
|---|---|---:|---:|---|
| 已有结果 | capture-native-01 | 1024 | 200 | 保留原始低负载结果 |
| 覆盖对照 | capture-native-01 | 1024 | 500 | 只延长 KV 窗口 |
| 新载荷 | capture-dp8-01 | 1024 | 500 | 每个 D API 并发8请求，重采实际 MoE |

每种测试内部仍有 M/K/MK/SERIAL 四组，iterations=20、repeats=5。**比较各自 MK/M，不把新 trace 的绝对耗时除以旧 trace 的 M。** 如果两个 trace 的重叠率差很多，先延长不足者的 KV 覆盖，再比较干扰程度。90%可作为覆盖目标，但仍须检查设备 trace，不能把 host 覆盖当作设备并发。

1024-token KV 每批24MiB/rank，4接收rank合计96MiB；500批合计46.875GiB（每个 K/MK/SERIAL 阶段、每轮）。每rank仍是一个同步搬运调用完成后再提交下一批，不是500并发请求，不会分配500份缓存。

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
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
unset QWEN_MOE_CAPTURE_DIR
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 228 --trace-dir experiments/qwen3_vllm_cam82/results/capture-native-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id native-kv1024-gap0-02 --iterations 20 --kv-transfers 500 --repeats 5 --prefix-tokens 1024 --kv-gap-ms 0
```

238实验窗口：

```bash
docker exec -it qwen3-vllm-cam82 bash
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

## 4 采集每个 D API 8个并发请求的实际通信

确认第3节两机重放已完整退出。本阶段直连两个D API，只需拉起D8，**不需要P和proxy**；D本地处理这些请求的prefill，再采集decode。不能把这种采集请求的端到端时间当作P/D性能。

保持TP4、DP2、EP8、原生off。`--capture-load-test`仅允许D原生采集时把max-num-seqs提高到最多32；普通基线默认上限4不变。KV预留仍2048×64token，不扩大缓存。容量检查不保证算子workspace不OOM，先8，再视情况16/32。

228服务窗口：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
export QWEN_MOE_CAPTURE_DIR=/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp8-01
export QWEN_MOE_CAPTURE_SKIP_PAIRS=768 QWEN_MOE_CAPTURE_PAIRS=48
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-228 --tp-mode off --capture-load-test --max-num-seqs 8 2>&1 | tee experiments/qwen3_vllm_cam82/results/d228-capture-dp8-01.log
```

238服务窗口（及时启动，不等待228服务ready）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
export QWEN_MOE_CAPTURE_DIR=/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp8-01
export QWEN_MOE_CAPTURE_SKIP_PAIRS=768 QWEN_MOE_CAPTURE_PAIRS=48
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-238 --tp-mode off --capture-load-test --max-num-seqs 8 2>&1 | tee experiments/qwen3_vllm_cam82/results/d238-capture-dp8-01.log
```

保持两服务窗口。768对相当于跳过更多decode通信，让并发请求有机会进入稳定批次，不保证采集时batch恰为8。228新窗口：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
curl --noproxy '*' --connect-timeout 3 --max-time 10 -fsS http://10.155.205.228:30011/health -w '\nHTTP=%{http_code}\n'
curl --noproxy '*' --connect-timeout 3 --max-time 10 -fsS http://10.155.205.238:30011/health -w '\nHTTP=%{http_code}\n'
python experiments/qwen3_vllm_cam82/moe_kv/capture_load.py --requests-per-dp 8 --tokens 128
```

两个health均200后才执行负载。期望16个请求完成、最后`CAPTURE_LOAD_COMPLETE`；每个D日志出现4个`MOE_CAPTURE_READY`。别再发其他请求。采集只记录真实路由，不强行将split乘8。

## 5 两机合并新 trace 并核实实际载荷

228、238都执行（相同命令，宿主机）：

```bash
mkdir -p /ssd_2/yuanwenliu/moe-kv-exchange/capture-dp8-01
```

然后仅228宿主机执行：

```bash
scp 'yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp8-01/ep[4-7].json' /ssd_2/yuanwenliu/moe-kv-exchange/capture-dp8-01/
scp /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp8-01/ep0.json /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp8-01/ep1.json /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp8-01/ep2.json /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp8-01/ep3.json yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/moe-kv-exchange/capture-dp8-01/
```

228、238都执行（相同命令，宿主机）：

```bash
docker cp /ssd_2/yuanwenliu/moe-kv-exchange/capture-dp8-01/. qwen3-vllm-cam82:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp8-01/
docker exec qwen3-vllm-cam82 python /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/moe_kv/spec.py --trace-dir /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/capture-dp8-01
```

必须两机相同trace_sha256、split校验通过。看`cross_host_kib_per_dispatch_combine_pair`是否明显大于旧256.5KiB，以及`dispatch_input_rows_per_rank_min/max`。这些行是专家路由后的行数，不直接等于请求batch。若载荷没有增加，不能称其为大消息实验；先检查调度、采集窗口，不继续盲目重放。如果分片序列不匹配，换新目录两机同步重启采集，禁止手改trace。

验证后在两个D前台窗口Ctrl+C，确认自己的workers退出，才能下一步。

## 6 重放新 trace

228实验窗口：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
unset QWEN_MOE_CAPTURE_DIR
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 228 --trace-dir experiments/qwen3_vllm_cam82/results/capture-dp8-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id dp8-kv1024-n500-01 --iterations 20 --kv-transfers 500 --repeats 5 --prefix-tokens 1024 --kv-gap-ms 0
```

238实验窗口（紧接228启动）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
unset QWEN_MOE_CAPTURE_DIR
python -u experiments/qwen3_vllm_cam82/moe_kv/run.py --host 238 --trace-dir experiments/qwen3_vllm_cam82/results/capture-dp8-01 --output-dir experiments/qwen3_vllm_cam82/results/moe-kv --run-id dp8-kv1024-n500-01 --iterations 20 --kv-transfers 500 --repeats 5 --prefix-tokens 1024 --kv-gap-ms 0
```

全部完成后，228宿主机：

```bash
mkdir -p /ssd_2/yuanwenliu/moe-kv-exchange/dp8-kv1024-n500-01
scp 'yuanwenliu@10.155.205.238:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/dp8-kv1024-n500-01/rank*.jsonl' /ssd_2/yuanwenliu/moe-kv-exchange/dp8-kv1024-n500-01/
docker cp /ssd_2/yuanwenliu/moe-kv-exchange/dp8-kv1024-n500-01/. qwen3-vllm-cam82:/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/dp8-kv1024-n500-01/
docker exec qwen3-vllm-cam82 python /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/moe_kv/analyze.py --run-dir /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv/dp8-kv1024-n500-01 --output-dir /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/results/moe-kv-analysis/dp8-kv1024-n500-01
```

## 7 如何判读、是否继续加压

1. 先检查完整性、KV正确性、真实载荷增加和覆盖率。失败run不能用于性能结论。
2. 看每轮MK/M、KV的MK/K、重叠样本配对变化及SERIAL恢复。变慢必须超过同条件重复波动；不设必须达到10%之类的结论门槛。
3. 若更大载荷仍仅变化几个百分点，这是有效结果。确认设备侧重叠和共享资源后，才能讨论硬件余量或竞争不敏感。
4. 如要继续，下一轮可单独将每DP并发及max-num-seqs升到16（必要时32），使用新capture目录；或保持同一trace单独将KV前缀升到2048。不要同时变更二者。更大的KV消息也不保证更高带宽，需看K组实测吞吐。
5. 出现稳定干扰后，另做短profiling（主README第8节，使用对应新trace），区分CPU提交等待、设备串行、DMA/HBM竞争与物理链路拥塞。微基准不能直接推出TPOT劣化。

代码新增项只做了CPU测试；8并发模型采集、NPU重放和新的性能结果仍需228/238实测。恢复普通P/D时清除采集环境，不加`--capture-load-test`，按主部署文档运行。
