# Qwen3：CANN 8.2 / vLLM 原生 MoE + CAM TP 多路径实验

更新：2026-10-15。本页为新独立环境，替代本实验后续执行入口；不在此前CANN9/Torch2.10容器里降级。代码已接入，**未在910B4实机编译/跑通**；每个关口必须通过才继续。本文不实现统一控制器，不实现独立HBM KV Pool。

## 1 布局、版本与接入边界

| 主机 | 物理卡 | 模型角色 / 并行 | TP CAM子组 |
| --- | --- | --- | --- |
| 228 | 0–3 | P4：Attention TP4、DP1、MoE EP4、Expert TP1 | 0,1,2,3 |
| 228 | 4–7 | D8的DP rank0：Attention TP4 | 4,5,6,7 |
| 238 | 0–3 | D8的DP rank1：Attention TP4 | 0,1,2,3 |
| 238 | 4–7 | CAM中继窗口owner：不加载模型，不存储KV | 不发起TP算子 |

D整体Attention TP4＋DP2、MoE EP8；专家在8卡间分片，每卡16个。P专家每卡32个。P和D的MoE域独立，KV走原生P→D Mooncake。每个host单独一个8卡CAM资源域，含两个TP4子组或TP4＋4中继；CAM不跨机。八卡都是910B4/32GB HBM、机内HCCS FullMesh，主机网卡enp67s0f5，驱动保持已知25.5.2，不修改宿主机驱动或CANN。

| 组件 | 新环境固定目标 |
| --- | --- |
| 官方A2镜像 | `quay.io/ascend/vllm-ascend:v0.10.2rc1`，必须arm64 |
| 固定arm64子manifest | `sha256:610df8ee1ba35261cd324e2ecb161710f34a1f0c32875bc454872a8a64291f12`，于2026-10-14通过Quay核对 |
| CANN | **8.2.RC1**，不是把9.0镜像里的包替换成8.2 |
| vLLM / vLLM-Ascend | 0.10.2 / 0.10.2rc1 |
| Ascend源码 | `048bfd55531909e1827616102dd40b4cf233f250` |
| Python / torch / torch_npu | 3.11 / 2.7.1 / 2.7.1.dev20250724 |
| Mooncake源码 | v0.3.9，`a00f75742469d6cc408bc5393b73c501e92dc74a` |
| Mooncake构建 | USE_ASCEND=ON（旧HCCL Ascend transport），USE_ASCEND_DIRECT=OFF；仅TE，不建Store |
| CAM源码 | GitCode `yuanwenliu/umdk`，分支`feat/qwen3-cam-detour-build`，`9c0bbdadfcfe882426cabdce5cdf2d5d2fce92ef` |

版本来源：[官方8.2 Dockerfile](https://github.com/vllm-project/vllm-ascend/blob/v0.10.2rc1/Dockerfile)、[配套表](https://github.com/vllm-project/vllm-ascend/blob/v0.22.1rc1/docs/source/community/versioning_policy.md)。旧版本维护/特性有差异；源码/版本保护拒绝不匹配，不取消保护去混装。25.5.2驱动与这组旧用户态软件的实际兼容性还需NPU关口确认。

**MoE保持原生，不使用上一轮`qwen_moe_adapter`。** 固定旧版的eager Qwen3使用`CustomSparseMoeBlock → AscendFusedMoE`，EP4/8时原生`FusedMoEState.All2All`、`TokenDispatcherWithAll2AllV`。启动启用`--enable-expert-parallel`，设置`VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP=0`，不是假设存在Qwen配置文件的“alltoall开关”。通用model runner的A2策略字符串可能仍写allgather，验收要看实际Qwen分支/dispatcher与device trace。[原生选择源码](https://github.com/vllm-project/vllm-ascend/blob/v0.10.2rc1/vllm_ascend/ascend_forward_context.py)

**TP payload的三种语义全部通过CAM All-to-All Detour分解：** ReduceScatter＝CAM交换后本地归约；AllGather＝复制本地片段后CAM交换；AllReduce＝上述ReduceScatter＋AllGather。AllReduce不会再把最后一步payload gather留在HCCL。每chunk保留TP组HCCL控制barrier以防窗口复用竞态；这是正确性基线，不是最优性能方案。

这不是直接调用仓库中专用的`reduce_scatter_detour` / `all_gather_detour` kernel：前者的物理域rank_size与子组推导需要进一步校核，后者有共享调度flag协议；本轮用已独立构建的All-to-All Detour统一分解，避免一边宣称接入一边混用HCCL payload。若要求专用kernel本身的性能，需要另做移植/验收。

适配只识别**TP组对象本身**，不按“4个rank”模糊匹配；MoE内部调用处于排除范围，EP、DP及MoE自己的TP恢复通信不替换。推理默认eager且启用**原生prefill SP**，使prefill出现TP ReduceScatter/AllGather，decode仍有TP AllReduce。不开SP时RS不一定出现；可以用`--no-sp`诊断，但所有对照组必须保持同一SP设置。图捕获、量化、EPLB、投机解码均不在支持范围。

## 2 两机宿主机：拉代码与准备目录

所有个人文件放`/ssd_2/yuanwenliu`，不写`/ssd_2/sglang`。有卡机不连接8.146开发机；运行期间只有228/238互通。实验机只拉代码，不发布代码。

已有实验仓库在两机分别执行；若有本地改动，先处理，不reset/覆盖：

```bash
cd /ssd_2/yuanwenliu/sglang
git status --short --branch
git pull --ff-only origin feat/qwen3-cam-detour-conflict
git rev-parse HEAD
```

没有仓库时才clone：

```bash
cd /ssd_2/yuanwenliu
git clone --branch feat/qwen3-cam-detour-conflict https://github.com/COLDTEA-L/qwen3.git sglang
```

两机各准备固定版本proxy源码，不移动旧`vllm-ascend-qwen3`：

```bash
cd /ssd_2/yuanwenliu
git clone --depth 1 --branch v0.10.2rc1 https://github.com/vllm-project/vllm-ascend.git vllm-ascend-cam82
git -C /ssd_2/yuanwenliu/vllm-ascend-cam82 rev-parse HEAD
mkdir -p /ssd_2/yuanwenliu/models
```

238准备Docker构建源码（目录存在就检查，不重复覆盖）：

```bash
cd /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82
mkdir -p vendor
git clone --depth 1 --branch feat/qwen3-cam-detour-build https://gitcode.com/yuanwenliu/umdk.git vendor/umdk
git clone --depth 1 --branch v0.3.9 --recurse-submodules https://github.com/kvcache-ai/Mooncake.git vendor/Mooncake
git clone --depth 1 --branch 0.5.7 https://github.com/alibaba/yalantinglibs.git vendor/yalantinglibs
git -C vendor/umdk rev-parse HEAD
git -C vendor/Mooncake rev-parse HEAD
git -C vendor/Mooncake submodule status
```

UMDK应为上表9c0bbdad。若GitCode分支尚未到该提交，需要先从开发端发布已有CAM分支，再在实验机拉取；不使用228上旧UMDK混编。此轮无须改UMDK源码，也无须下载MoE Dispatch/Combine专用算子。vendor被Git忽略，不带运行私有配置。

## 3 权重准备：已有就复用，缺失再下载

两机已有完整Qwen3-30B-A3B BF16权重即可跳过下载，不能用量化权重替代。每台宿主机查看：

```bash
df -h /ssd_2/yuanwenliu /ssd_1/docker
ls -lh /ssd_2/yuanwenliu/models/Qwen3-30B-A3B/config.json /ssd_2/yuanwenliu/models/Qwen3-30B-A3B/model.safetensors.index.json
```

缺权重时仅在238宿主机建独立下载venv，不改推理镜像依赖。先export你已验证可用的http_proxy/https_proxy，不把带密码URL写进仓库。代理密码含特殊字符时需URL编码。

```bash
python3 -m venv /ssd_2/yuanwenliu/qwen3-cam82-download
/ssd_2/yuanwenliu/qwen3-cam82-download/bin/python -m pip install huggingface_hub==0.34.4 requests==2.32.5
export HF_HUB_DISABLE_XET=1
/ssd_2/yuanwenliu/qwen3-cam82-download/bin/python /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/download_weights.py
```

若仍是此前代理证书问题，获准后只在该下载命令末尾加`--insecure`。它仅关闭该进程的HTTPS证书校验，不改系统/Git/Docker证书策略；存在中间人风险，仅在可信代理环境使用。不要升级推理容器的Transformers解决下载问题。

238→228传权重用scp，不依赖rsync（两端模型服务先停；不要在运行时覆盖权重）：

```bash
scp -rp /ssd_2/yuanwenliu/models/Qwen3-30B-A3B yuanwenliu@10.155.205.228:/ssd_2/yuanwenliu/models/
```

两机分别计算safetensors校验和，比较结果：

```bash
find /ssd_2/yuanwenliu/models/Qwen3-30B-A3B -name '*.safetensors' -type f -exec sha256sum {} \;
```

## 4 238宿主机：构建新Docker，复制给228

**当前进度（2026-10-15）：你已完成4.2–4.4，基础镜像已用crane下载、导入并打上本地标签，arm64及四个包版本已核对。现在从4.5继续；不要重复下载或导入。磁盘空间可按4.1复查，vendor源码尚未准备齐全时先补第2节对应部分。**

### 4.1 检查宿主机磁盘空间

先确认Docker磁盘可容纳新基础镜像、源码编译层和导出tar；之前166GB不是实时保证。不自动删共享容器/镜像、不重启Docker daemon。

```bash
df -h /ssd_1/docker /ssd_2/yuanwenliu
docker system df
```

### 4.2 使用crane下载固定基础镜像（已完成，跳过）

238的Docker daemon不能直接pull，本流程不再依赖docker pull，改用你已验证可用的crane。crane使用当前shell的联网配置；代理凭据不写进仓库。下面保留你已使用的`--insecure`，它允许非安全的TLS/HTTP获取，仅在获准的可信网络使用，不修改Docker daemon或系统证书设置。已有完整tar且已导入时不重新下载、不覆盖：

```bash
/ssd_2/yuanwenliu/tools/crane/crane --insecure pull --platform linux/arm64 --format legacy quay.io/ascend/vllm-ascend@sha256:610df8ee1ba35261cd324e2ecb161710f34a1f0c32875bc454872a8a64291f12 /ssd_2/yuanwenliu/vllm-ascend.tar
```

crane只保存tar，不会自动把镜像加载进Docker；直接docker inspect远端digest出现No such image不代表tar下载失败。

### 4.3 导入Docker并设置本地标签（已完成，跳过）

```bash
docker load -i /ssd_2/yuanwenliu/vllm-ascend.tar
```

本次固定镜像实际输出`Loaded image ID: sha256:3149e4e4e19556fc5cb22c6be9aee3a262f599cdc38cb58f24dfeb47b783e64c`。确认load成功后打标签；若输出不同，先核对，不盲目使用其他镜像：

```bash
docker tag sha256:3149e4e4e19556fc5cb22c6be9aee3a262f599cdc38cb58f24dfeb47b783e64c qwen3-vllm:cam82-base
docker image inspect qwen3-vllm:cam82-base --format '{{.Id}} {{.Architecture}}'
```

已确认ImageID为上面的3149e4e4…，Architecture为arm64。注册表manifest digest和Docker ImageID不是同一概念；tar导入后原来的`仓库@sha256:manifest`引用可能不可用，RepoDigests也可能为空。**后续查询、构建都使用本地标签`qwen3-vllm:cam82-base`，不再用远端digest引用。**

### 4.4 核对基础镜像包版本（已完成，跳过）

下面不联网、不占卡，也不执行Torch import：

```bash
docker run --rm --network=none --runtime=runc --entrypoint python qwen3-vllm:cam82-base -m pip show vllm vllm-ascend torch torch-npu | awk '/^(Name|Version):/ {print}'
```

238已回传并核对：

| 包 | 基础镜像实测版本 |
| --- | --- |
| vllm | 0.10.2+empty |
| vllm_ascend | 0.10.2rc1 |
| torch | 2.7.1+cpu |
| torch_npu | 2.7.1.dev20250724 |

`+empty`与`+cpu`是这套镜像的包版本后缀，不能据此认定只能在CPU运行；NPU由torch_npu与Ascend运行时提供。适配检查会先去掉`+`后缀再比较版本。以上仅确认架构/包元数据，尚未确认CANN元数据、关键源码指纹或NPU执行；Dockerfile会在apt/CAM/Mooncake编译前继续检查基础版本和源码。

### 4.5 检查vendor源码并构建实验镜像（现在从这里继续）

先在238宿主机确认第2节的三个构建源码已经准备好。任何一个文件缺失，先补第2节vendor准备，不要直接构建：

```bash
cd /ssd_2/yuanwenliu/sglang
ls experiments/qwen3_vllm_cam82/vendor/umdk/build/cam/comm_operator/build_qwen3_all2all_detour.sh experiments/qwen3_vllm_cam82/vendor/Mooncake/CMakeLists.txt experiments/qwen3_vllm_cam82/vendor/yalantinglibs/CMakeLists.txt
```

Docker build使用已在238 shell中export的代理变量；下面通过变量传参，不在命令中写明密码。支持原来的legacy builder，不用`--progress`。该Dockerfile只加编译依赖、固定Mooncake、CAM和本仓库适配包，不升级Torch/Transformers。构建CAM指定ascend910b4、按当前Torch ABI编译，不复用CANN9产物。

若当前只有小写`http_proxy`可用，先让HTTPS与大写变量沿用同一代理，不需要重复输入密码：

```bash
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
```

```bash
cd /ssd_2/yuanwenliu/sglang
set -o pipefail
DOCKER_BUILDKIT=0 docker build --network=host --build-arg BASE_IMAGE=qwen3-vllm:cam82-base --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg http_proxy --build-arg https_proxy -f experiments/qwen3_vllm_cam82/Dockerfile -t qwen3-vllm:cam82-v1 experiments/qwen3_vllm_cam82 2>&1 | tee /ssd_2/yuanwenliu/cam82-build.log
```

**必须保留`--build-arg BASE_IMAGE=qwen3-vllm:cam82-base`。** Dockerfile保留固定官方digest作为默认值，但本环境通过参数选择已经导入的本地基础镜像；不加该参数可能再次尝试联网拉取。apt等构建依赖下载仍需要代理，这与不再拉取基础镜像是两件事。

必须本次构建退出成功且看到Successfully built/tagged，才单独执行以下检查、进入4.6；即使旧tag存在，也不能据此认为本次构建成功：

```bash
docker image inspect qwen3-vllm:cam82-v1 --format '{{.Id}} {{.Architecture}}'
```

CANN metadata、源码指纹、库/API或编译不匹配就停止，回传最早错误；不安装最新torch_npu兜底。构建用SDK/devlib编译，不验证设备执行。Mooncake旧HCCL transport与固定CANN的编译/运行是新的关口，不宣称必然通过。

#### 4.5.1 已定位：CAM构建时ErrorManager未定义符号

2026-10-15的238实测：devlib优先时`import torch, torch_npu`报`libruntime_common.so: undefined symbol: _ZN12ErrorManager19ATCReportErrMessage...`；同一基础镜像改为toolkit的`lib64`、`aarch64-linux/lib64`优先后，输出`IMPORT_OK 2.7.1+cpu 2.7.1.dev20250724`。这说明动态库搜索顺序会触发该错误，不是已证明需要换Torch/CANN版本，也不是模型或权重错误；尚未通过loader跟踪定位具体冲突的依赖库文件。

原Dockerfile中devlib优先的写法已修正。新版CAM构建步骤先source CANN/NNAL，然后把toolkit运行库放前、devlib放后；新增`CAM82_BUILD_TORCH_IMPORT_OK`检查，再编译CAM。镜像结束前还检查Torch、torch_npu、CAM扩展及Mooncake原生模块导入，需`CAM82_NATIVE_IMPORT_OK`。运行时`env.sh`同样把toolkit运行库放在前面。

若遇到这次错误：更新本实验仓库后，**直接重跑4.5原来的docker build命令**；不重新下载基础镜像、vendor或权重，不先做4.6。不要加`--no-cache`或清理已有构建缓存，Docker通常能复用前面成功的层，从修改后的CAM层继续。当前只有基础镜像的导入顺序测试通过，新Dockerfile是否完全编译成功仍要看本轮结果。

不要用`TORCH_DEVICE_BACKEND_AUTOLOAD=0`跳过NPU后端加载来当作修复；它不能证明torch_npu或CAM可用。也不要把其他CANN版本的同名`.so`复制到镜像里。如果新版导入检查仍失败，保留第一个错误和对应构建层，停止后续步骤。

### 4.6 238保存并传输实验镜像（构建成功后）

```bash
docker save -o /ssd_2/yuanwenliu/qwen3-vllm-cam82-v1.tar qwen3-vllm:cam82-v1
sha256sum /ssd_2/yuanwenliu/qwen3-vllm-cam82-v1.tar
scp /ssd_2/yuanwenliu/qwen3-vllm-cam82-v1.tar yuanwenliu@10.155.205.228:/ssd_2/yuanwenliu/
```

### 4.7 228核对、导入实验镜像

```bash
sha256sum /ssd_2/yuanwenliu/qwen3-vllm-cam82-v1.tar
docker load -i /ssd_2/yuanwenliu/qwen3-vllm-cam82-v1.tar
docker image inspect qwen3-vllm:cam82-v1 --format '{{.Id}} {{.Architecture}}'
```

两机ImageID必须一致。保留旧容器、旧镜像和模型文件。

## 5 两机建容器、进入和预检查

两机各在宿主机执行；使用新容器名，不替换旧实验：

```bash
cd /ssd_2/yuanwenliu/sglang
bash experiments/qwen3_vllm_cam82/create_container.sh qwen3-vllm:cam82-v1 qwen3-vllm-cam82 --print-only
bash experiments/qwen3_vllm_cam82/create_container.sh qwen3-vllm:cam82-v1 qwen3-vllm-cam82
docker exec -it qwen3-vllm-cam82 bash
```

每个新窗口先docker exec进入容器，再source。228容器：

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
python experiments/qwen3_vllm_cam82/check.py --device --model
python experiments/qwen3_vllm_cam82/test_cpu.py
```

238容器同样做，source参数改238。需`CAM82_CHECK_PASS`和CPU测试成功。包检查不等于CAM/PD运行成功。不能source旧qwen3_cam/env.sh或qwen3_vllm/env.sh；不要设置ASCEND_RT_VISIBLE_DEVICES，物理选择由新worker明确处理。模型离线；228/238的KV与DP通信必须联网。

这轮旧HCCL Ascend transport不沿用此前Direct TE要求的机内RoCE开关；env固定`HCCL_INTRA_ROCE_ENABLE=0`，用于保持机内HCCS候选路径、跨机仍可RoCE。实际链路仍需profile/计数核实，所有对照组保持同一设置，不把选路变化当成拥塞影响。

适配包在镜像里安装，不会仅因宿主机拉新代码而自动更新。后续纯Python修订可在新CAM82容器内离线执行`python -m pip install --no-deps --no-build-isolation --no-index -e experiments/qwen3_vllm_cam82/adapter`，记录HEAD并重跑关口；涉及CAM/CANN/native库版本变化必须重新构建镜像。不把适配包安装到旧CANN9/SGLang容器。

两机宿主机各查8张NPU IP/链路，无须贴所有listen端口；失败时回传相关卡输出：

```bash
for i in 0 1 2 3 4 5 6 7; do /usr/local/Ascend/driver/tools/hccn_tool -i "$i" -ip -g; /usr/local/Ascend/driver/tools/hccn_tool -i "$i" -link -g; done
```

## 6 先验收CAM三种TP语义，再启动模型

确认自己的旧服务/probe已正常退出，并获得两机16卡实验窗口；不要kill他人任务。下面每台新容器各自运行，**只用于probe，不用于模型启动**：

```bash
set -o pipefail
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29631 experiments/qwen3_vllm_cam82/smoke_tp.py --mode hccl-a2a 2>&1 | tee "experiments/qwen3_vllm_cam82/results/tp-hccl-$(date +%Y%m%d-%H%M%S).log"
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29631 experiments/qwen3_vllm_cam82/smoke_tp.py --mode cam-detour 2>&1 | tee "experiments/qwen3_vllm_cam82/results/tp-cam-$(date +%Y%m%d-%H%M%S).log"
```

需两台机分别`CAM_TP_SMOKE_PASS`并全部进程退出。它同时运行两个TP4子组，测试AllReduce/ReduceScatter/AllGather、fp32/bf16、跨chunk、小输入、维度变换和空输入，对照HCCL结果。发现卡号错配/挂起/不一致，停止启动模型；两组同时使用CAM窗口是重要关口，不用CPU代数验证替代。probe在结束时通过进程退出释放自己的窗口，不手动清其他进程HBM。

## 7 原生P4+D8：TP off对照（MoE保持原生All-to-All）

以下三个服务窗口各自source。先不用中继进程，因为off不创建CAM窗口。模型服务**不用torchrun**。

228窗口1（P4）：

```bash
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role p-228 --tp-mode off 2>&1 | tee "experiments/qwen3_vllm_cam82/results/p-off-$(date +%Y%m%d-%H%M%S).log"
```

228窗口2（D8 DP rank0，物理4–7）：

```bash
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-228 --tp-mode off 2>&1 | tee "experiments/qwen3_vllm_cam82/results/d228-off-$(date +%Y%m%d-%H%M%S).log"
```

238窗口1（D8 DP rank1，物理0–3）：

```bash
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-238 --tp-mode off 2>&1 | tee "experiments/qwen3_vllm_cam82/results/d238-off-$(date +%Y%m%d-%H%M%S).log"
```

两端D需同时启动建立EP8；不是两个独立D4。没有SGLang PD store/7100，没有Mooncake master；KV使用P2PHANDSHAKE的旧Ascend transport。日志需`NATIVE_QWEN_MOE_ALL2ALL ep=4/8`及物理/TP组映射。32GB HBM先上下文2048、max-seqs4、memory0.75；OOM区分权重峰值、profile临时张量与KV预算，不保证仅缩上下文能解决。

诊断时可在三处serve命令均追加`--max-model-len 1024 --max-num-seqs 2 --gpu-memory-utilization 0.70`，同时缩短客户端输入。原生scheduler可能调整profile/批处理token预算，需记录实际值；启动权重峰值导致的OOM不能仅靠这些参数保证解决。比较TP模式时三组预算保持一致。

三处health成功后，228窗口3启动proxy：

```bash
curl --noproxy '*' -fsS http://10.155.205.228:30010/health
curl --noproxy '*' -fsS http://10.155.205.228:30011/health
curl --noproxy '*' -fsS http://10.155.205.238:30011/health
python -u experiments/qwen3_vllm_cam82/launch.py proxy
```

228窗口4发送请求、保存输出与负载（复用CPU客户端，不导入CANN9适配）：

```bash
curl --noproxy '*' -fsS http://10.155.205.228:30080/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen3","prompt":"Explain prefill and decode separation.","max_tokens":32,"temperature":0}'
python experiments/qwen3_vllm/compare_outputs.py collect --url http://10.155.205.228:30080 --output experiments/qwen3_vllm_cam82/results/pd8-off-output.json
python experiments/qwen3_vllm/workload.py --requests 20 --rps 0.2 --concurrency 2 --output "experiments/qwen3_vllm_cam82/results/pd8-off-$(date +%Y%m%d-%H%M%S).jsonl"
```

必须确认真实KV转移完成，不仅health/HTTP200；检查P/D connector日志及实际搬运，排除重算回退。若原生Qwen的非整除token/空DP输入失败，停下回传原生错误，本轮不绕过MoE算法。默认原生prefill SP使用原生padding，不能声称覆盖所有模型输入情况。

## 8 换成CAM TP：三处服务＋238中继

先正常结束自己的proxy与三处模型，确认worker退出。两机CAM域的所有owner每轮一起新建，不能沿用上一轮残留窗口。

238窗口2先启动四个中继owner（随后等D窗口加入CAM域，暂未打印READY属正常）：

```bash
set -o pipefail
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29632 experiments/qwen3_vllm_cam82/relay.py 2>&1 | tee "experiments/qwen3_vllm_cam82/results/relay-$(date +%Y%m%d-%H%M%S).log"
```

再在第7节对应三个窗口启动：

```bash
# 228 P窗口
python -u experiments/qwen3_vllm_cam82/launch.py serve --role p-228 --tp-mode cam-detour
# 228 D窗口
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-228 --tp-mode cam-detour
# 238 D窗口
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-238 --tp-mode cam-detour
```

务必启动全部三个模型＋238中继。228的P/D互相提供中继窗口，238由四个owner补齐域。CAM端口为各自host的127.0.0.1:10067，512MiB窗口/卡；不是跨机10067，更不联系8.146。缺owner时bootstrap可等至600秒超时；不能只启动P或只启动一端D并期待CAM就绪。

验收`CAM_RELAY_READY`四次、每模型rank的`CAM_TP_READY`，请求后`CAM_TP_CALL mode=cam-detour op=all_reduce/all_gather/reduce_scatter`以及原生MoE标记。三种调用不要求每个request/每个角色同时出现：prefill SP主要AG/RS、decode主要AR，logits也可能AG。没有调用日志的类型不能当作已测。

health成功后复用第7节proxy与请求。输出另存`pd8-cam-output.json`，比较：

```bash
python experiments/qwen3_vllm/compare_outputs.py collect --url http://10.155.205.228:30080 --output experiments/qwen3_vllm_cam82/results/pd8-cam-output.json
python experiments/qwen3_vllm/compare_outputs.py compare --native experiments/qwen3_vllm_cam82/results/pd8-off-output.json --alltoall experiments/qwen3_vllm_cam82/results/pd8-cam-output.json
```

工具的`--alltoall`只是第二个快照的参数名，此处比较的是TP模式，不是改MoE。短prompt一致仅是smoke；归约顺序变化可能导致BF16/greedy差异，差异需核对logits/质量，不自动忽略。至少覆盖短/长prefill、混合batch、D一侧空负载、多轮连续请求后才测性能。

额外对照`hccl-a2a`：停自己的整轮服务后，三处`--tp-mode hccl-a2a`重新启动；不启动中继。这组保留相同分块、padding、本地归约及控制barrier，仅交换payload走HCCL，分离分解开销与CAM路径变化。off不是“没有TP通信”，而是原生TP。

## 9 冲突与影响评估

本轮能测**CAM TP多路径、原生MoE All-to-All、正常P→D KV**是否共享资源并产生干扰；不包含KV Pool恢复。尤其228的TP子组可借另一角色的卡中继，但不能凭逻辑箭头认定物理链路被KV使用。MoE域与TP组不同，不代表物理链路一定不共享。

按off / hccl-a2a / cam-detour分别运行相同请求序列、SP/并行/模型/预算，固定到达率与输出长度，每组至少3轮。先低负载，再增加并发/请求长度；运行同样的客户端：

```bash
python experiments/qwen3_vllm/workload.py --requests 200 --rps 0.5 --concurrency 4 --prefix-tokens 1024 --output "experiments/qwen3_vllm_cam82/results/load-$(date +%Y%m%d-%H%M%S).jsonl"
python experiments/qwen3_vllm/workload.py --summarize experiments/qwen3_vllm_cam82/results/本轮结果.jsonl
```

记录成功率、TTFT/TPOT P50/P99、吞吐与SLO违反率；200样本仅起步，不能可靠估计极端尾延迟。输入＋输出不得超过2048。观察到后端变快/变慢不等于证明冲突。

device profile：每个模型启动窗口在启动前设置不同的`VLLM_TORCH_PROFILER_DIR`绝对路径，例如228P使用`/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/profiles/run1-p228`，两处D使用各自不同目录。通过每个引擎HTTP端点POST `/start_profile`，短请求窗口后POST `/stop_profile`；proxy不提供这些控制。纯性能轮次不要开profile。

判定冲突需同时满足：①源/目标及真实共享链路/端口证据；②CAM、MoE、KV在device/SDK完成时间上重叠，跨机时钟校准；③并发相对串行/错峰的通信时延/排队可重复恶化，排除HBM容量、计算、冷连接和分解开销。Host的`CAM_TP::...`范围不是device重叠证据，外部KV transport信息未必都出现在Torch trace。

影响计算：通信slowdown＝并发耗时/单独或错峰耗时；TTFT/TPOT相对变化及吞吐变化保持相同负载。需要同时看到链路拥塞/排队与对应时延恶化，不能预设影响百分比。

## 10 退出、故障与验证边界

退出自己的proxy→P和两端D→238中继。所有CAM owner必须等通信停止后退出；不能中途只杀relay或只重启一个模型再复用旧域。使用各前台窗口Ctrl+C、确认自己的子进程退出；不pkill python、不reset NPU、不清共享/dev/shm、不改他人容器。窗口HBM由owner进程分配，PD store并不负责释放它。

| 失败 | 回传重点 |
| --- | --- |
| build | 第一处错误、ImageID/Architecture、当前源码HEAD；不要继续run不存在镜像 |
| package/source check | 实际元数据/源码指纹、CANN版本；不混装9.0wheel |
| CAM构建Torch import出现ErrorManager符号错误 | 先看4.5.1，使用更新后的toolkit优先顺序；不通过关闭NPU自动加载绕过 |
| CAM init/挂起 | 两机整轮P/D/relay日志、物理映射、10067的具体监听；不贴全部端口 |
| TP correctness | 两个TP4 probe日志、dtype/shape/维度、首次错误 |
| MoE native错误 | Qwen类/原生state/EP ranks、空/非整除batch、最早traceback；本轮不改MoE运输 |
| PD transfer | Mooncake旧Ascend transport首条错误、HCCN配置和NPU IP、PHYSICAL_DEVICES；不是MemFabric store |
| OOM | 角色/卡号/完整最早错误、npu-smi、是否已分配512MiB CAM窗口 |

源码适配、CPU/mock检查不等于A2可运行。Docker构建、CAM三语义probe、原生Qwen质量、EP8、跨机PD与三流profile均要你在228/238逐项验收；没有这些证据之前，不能写“冲突实验已跑通”。本轮未直接控制实验机，没有下载或修改其权重，也没有在宿主机安装CANN。
