# Qwen3：CANN 8.2 / vLLM 原生 MoE + CAM TP 多路径实验

更新：2026-10-16。本页为新独立环境，替代本实验后续执行入口；不在此前CANN9/Torch2.10容器里降级。用户已在228完成HCCL与CAM机内probe（退出码均0）；**238修正后probe及完整P/D模型实验尚待验收**，每个关口必须通过才继续。本文不实现统一控制器，不实现独立HBM KV Pool。

执行约定：标注“228、238都执行（相同命令）”的代码块，要在两台机器上各执行一次，不要求同一时刻执行；标注具体主机/窗口的代码块只在指定位置执行。新开SSH窗口默认位于宿主机；出现`docker exec`的代码块，第一行在宿主机执行，进入容器后再执行后续行。已经在该容器内时，不重复执行`docker exec`，也不在容器内运行Docker构建/导入命令。

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
| CAM源码 | GitCode `yuanwenliu/umdk`，分支`feat/qwen3-cam-detour-build`，`9c0bbdadfcfe882426cabdce5cdf2d5d2fce92ef`；镜像内额外应用本实验仓库的修正，见4.5.2–4.5.6 |

版本来源：[官方8.2 Dockerfile](https://github.com/vllm-project/vllm-ascend/blob/v0.10.2rc1/Dockerfile)、[配套表](https://github.com/vllm-project/vllm-ascend/blob/v0.22.1rc1/docs/source/community/versioning_policy.md)。旧版本维护/特性有差异；源码/版本保护拒绝不匹配，不取消保护去混装。25.5.2驱动与这组旧用户态软件的实际兼容性还需NPU关口确认。

**MoE保持原生，不使用上一轮`qwen_moe_adapter`。** 固定旧版的eager Qwen3使用`CustomSparseMoeBlock → AscendFusedMoE`，EP4/8时原生`FusedMoEState.All2All`、`TokenDispatcherWithAll2AllV`。启动启用`--enable-expert-parallel`，设置`VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP=0`，不是假设存在Qwen配置文件的“alltoall开关”。通用model runner的A2策略字符串可能仍写allgather，验收要看实际Qwen分支/dispatcher与device trace。[原生选择源码](https://github.com/vllm-project/vllm-ascend/blob/v0.10.2rc1/vllm_ascend/ascend_forward_context.py)

**TP payload的三种语义全部通过CAM All-to-All Detour分解：** ReduceScatter＝CAM交换后本地归约；AllGather＝复制本地片段后CAM交换；AllReduce＝上述ReduceScatter＋AllGather。AllReduce不会再把最后一步payload gather留在HCCL。每chunk保留TP组HCCL控制barrier以防窗口复用竞态；这是正确性基线，不是最优性能方案。

这不是直接调用仓库中专用的`reduce_scatter_detour` / `all_gather_detour` kernel：前者的物理域rank_size与子组推导需要进一步校核，后者有共享调度flag协议；本轮用已独立构建的All-to-All Detour统一分解，避免一边宣称接入一边混用HCCL payload。若要求专用kernel本身的性能，需要另做移植/验收。

适配只识别**TP组对象本身**，不按“4个rank”模糊匹配；MoE内部调用处于排除范围，EP、DP及MoE自己的TP恢复通信不替换。推理默认eager且启用**原生prefill SP**，使prefill出现TP ReduceScatter/AllGather，decode仍有TP AllReduce。不开SP时RS不一定出现；可以用`--no-sp`诊断，但所有对照组必须保持同一SP设置。图捕获、量化、EPLB、投机解码均不在支持范围。

## 2 两机宿主机：拉代码与准备目录

所有个人文件放`/ssd_2/yuanwenliu`，不写`/ssd_2/sglang`。有卡机不连接8.146开发机；运行期间只有228/238互通。实验机只拉代码，不发布代码。

**228、238都执行（相同命令，宿主机）：** 更新已有实验仓库；若有本地改动，先处理，不reset/覆盖：

```bash
cd /ssd_2/yuanwenliu/sglang
git status --short --branch
git pull --ff-only origin feat/qwen3-cam-detour-conflict
git rev-parse HEAD
```

**228、238中尚无实验仓库的机器才执行（相同命令，宿主机）：**

```bash
cd /ssd_2/yuanwenliu
git clone --branch feat/qwen3-cam-detour-conflict https://github.com/COLDTEA-L/qwen3.git sglang
```

**228、238都需要（相同命令，宿主机）：** 准备固定版本proxy源码；目录已存在时只核对提交，不重复clone，不移动旧`vllm-ascend-qwen3`：

```bash
cd /ssd_2/yuanwenliu
git clone --depth 1 --branch v0.10.2rc1 https://github.com/vllm-project/vllm-ascend.git vllm-ascend-cam82
git -C /ssd_2/yuanwenliu/vllm-ascend-cam82 rev-parse HEAD
mkdir -p /ssd_2/yuanwenliu/models
```

已有容器时，也可以在容器内更新挂载的`/ssd_2/yuanwenliu/sglang`，宿主机与容器操作的是同一份实验代码，二选一执行即可，不并发进行Git更新。需联网代理时在更新前配置；`env.sh`会清除代理，更新完成后才加载运行环境。算子/依赖源码的准备、Docker构建仍在指定宿主机进行。

`vllm-ascend-cam82`是宿主机挂载目录（主要提供P/D请求代理示例），不会随`docker save/load`传输。228若未准备该目录，创建容器会报`Missing mount`。可以在238宿主机执行下列命令复制，之后在228核对上述Ascend源码提交；不要覆盖已有不同版本目录：

```bash
scp -rp /ssd_2/yuanwenliu/vllm-ascend-cam82 yuanwenliu@10.155.205.228:/ssd_2/yuanwenliu/
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

UMDK应为上表9c0bbdad。若GitCode分支尚未到该提交，需要先从开发端发布已有CAM分支，再在实验机拉取；不使用228上旧UMDK混编。不需手工修改或更新已固定版本的vendor/umdk：Dockerfile会只在镜像内对其副本应用本实验仓库的修正（见4.5.2–4.5.6），包括构建辅助脚本及CAM主机侧IPv4校验，不改宿主机vendor或通信算子算法，也无须下载MoE Dispatch/Combine专用算子。vendor被Git忽略，不带运行私有配置。

## 3 权重准备：已有就复用，缺失再下载

两机已有完整Qwen3-30B-A3B BF16权重即可跳过下载，不能用量化权重替代。**228、238都执行（相同命令，宿主机）：**

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

**228、238都执行（相同命令，宿主机）：** 计算safetensors校验和，比较两机结果：

```bash
find /ssd_2/yuanwenliu/models/Qwen3-30B-A3B -name '*.safetensors' -type f -exec sha256sum {} \;
```

## 4 238宿主机：构建新Docker，复制给228

**当前进度（2026-10-16）：用户已将238构建的实验镜像传入228，并在228补齐运行环境后通过HCCL与CAM机内probe。已有该镜像的228/238不因本次文档/环境脚本更新而重做本节构建与传输，直接按第5、6节继续。以下4.5.1–4.5.6保留历史构建故障及修正记录，供首次构建或排障使用；不是要求重新执行所有修复步骤。开发服务器的8.5隔离产物不能直接用于A2的8.2/Torch2.7环境。**

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

#### 4.5.2 已定位：Missing pregen autogen for aclnn_all_to_all_detour

该报错表示镜像仍未构建成功。`Successfully updated parameters`仅表示中间参数生成成功，不能据此进入4.6。

原因是固定UMDK版本中，算子目录叫`all_to_all_detour`，但已有API文件实际叫`aclnn_all2_all_detour.cpp`和`.h`。单算子构建脚本按目录名拼出`aclnn_all_to_all_detour`，因而找不到文件。这是构建脚本命名映射遗漏，不是少下载依赖，也不是模型权重或显存问题。

本实验仓库新增`patches/umdk-all2all-pregen-prefix.patch`。Dockerfile保留UMDK提交9c0bbdad，先核对原脚本SHA256和两个已有API文件，再检查并应用补丁，只将该算子的查找前缀映射为`aclnn_all2_all_detour`；其他算子、API符号名、算子目录和安装包名称不变。修正仅发生在镜像内的`/opt/umdk-cam82`副本，不改宿主机vendor，不需单独更新GitCode仓库。

开发端发布本实验仓库更新后，在238宿主机执行：

```bash
cd /ssd_2/yuanwenliu/sglang
git status --short --branch
git pull --ff-only origin feat/qwen3-cam-detour-conflict
ls experiments/qwen3_vllm_cam82/patches/umdk-all2all-pregen-prefix.patch
```

有本地改动先处理，不强制覆盖。随后**重跑4.5原来的docker build命令**，代理保持可用；不重新下载基础镜像、vendor或权重，不清缓存、不加`--no-cache`。应先看到`CAM82_PREGEN_PREFIX_PATCH_OK`，再继续CAM编译。该标记仅代表补丁应用成功；必须最终本次构建退出成功且看到Successfully built/tagged，才进入4.6。

补丁应用、前缀选择和pregen文件复制已做CPU回归测试，但完整CANN算子编译及NPU执行仍需在238验证。若脚本校验或补丁检查失败，不删除校验绕过，先回传最早错误。

#### 4.5.3 已定位：CANN runtime环境脚本的grep触发提前退出

238的严格模式定位输出为`ENV_FAIL rc=1 file=.../runtime/bin/setenv.bash line=84 command=grep "${DEP_HAL_NAME}" > /dev/null 2>&1`。因此构建在重复加载CANN环境时，被脚本内返回1的grep中止；此前日志停在`patch generated op_host CMakeLists`，并不表示CMake修补本身失败。grep返回1通常表示没有匹配；这条输出本身不证明缺少哪个HAL库、驱动不兼容或910B4不能使用该版本，不据此更换软件栈。

镜像内补丁`patches/umdk-cann-env-guard.patch`让单算子脚本调用`build_helpers/source_cann_env_compat.sh`：仅在加载SDK环境的短窗口内暂时关闭`errexit`和继承的ERR trap，使SDK脚本按普通shell语义完成内部探测；加载结束立即恢复严格模式与错误诊断，**最终source返回非零仍停止**，并检查ACL头文件、OPP目录、msopgen和cmake是否可用。随后恢复已验证的toolkit运行库优先顺序。这里不是全局关闭`set -e`，也不是`source ... || true`；后续编译、打包、安装失败仍立即停止。没有修改CANN原脚本或宿主机环境。

同时开启ERR trap继承，后续错误会输出文件、行号、命令和返回码。两个补丁均只应用在镜像内的固定UMDK副本，vendor提交仍为9c0bbdad，不需另行更新GitCode分支。

开发端发布本实验仓库更新后，238宿主机执行：

```bash
cd /ssd_2/yuanwenliu/sglang
git status --short --branch
git pull --ff-only origin feat/qwen3-cam-detour-conflict
ls experiments/qwen3_vllm_cam82/patches/umdk-cann-env-guard.patch experiments/qwen3_vllm_cam82/build_helpers/source_cann_env_compat.sh
```

有本地改动先处理。然后重跑4.5的原Docker构建命令，不清缓存、不加`--no-cache`、不重下vendor/基础镜像/权重。预期在补丁标记`CAM82_CANN_ENV_PATCH_OK`之后，单算子构建会输出`CAM82_CANN_ENV_READY`，再输出`run generated build.sh`。这些仅说明越过环境加载，不表示算子或镜像已经构建成功；仍须本次构建最终退出成功且出现Successfully built/tagged，才能进入4.6。CPU回归覆盖了中间grep返回1、SDK最终失败、SDK文件/必要资源缺失及后续编译失败，不替代238上的完整编译和NPU验证。

#### 4.5.4 已定位：cust_opapi编译缺少platform/platform_info.h

238已确认固定基础镜像中存在两个头文件：

```text
/usr/local/Ascend/ascend-toolkit/latest/include/experiment/platform/platform/platform_info.h
/usr/local/Ascend/ascend-toolkit/latest/include/experiment/platform/platform/platform_infos_def.h
```

报错链为pregen的`aclnn_all2_all_detour.cpp → aclnn/opdev/platform.h → platform/platform_info.h`。因此不是头文件包没安装，而是msopgen生成项目的`cust_opapi`未包含这条依赖所需的搜索根目录。要添加的是`$ASCEND_HOME_PATH/include/experiment/platform`：其后拼接`platform/platform_info.h`才对应实际文件，不能直接把最后的`platform`目录也加进根路径。诊断find输出的autofuse符号链接循环与本次编译报错不是同一件事，不要删除这些链接。

新增`patches/umdk-platform-include.patch`，仅在镜像内扩展现有CMake修补函数：保留pregen复制逻辑，并在生成的`op_host/CMakeLists.txt`末尾为`cust_opapi`添加PRIVATE头文件路径；在当前SDK下依次检查`include/experiment/platform`（8.2实测）和`include`（8.5实测），要求两个头文件都存在并输出`CAM82_PLATFORM_INCLUDE=...`。后续验证还补齐该目标的`aclnnInner_*.cpp`编译源，见4.5.6。不修改算子算法/API包装，不复制或伪造头文件，不修改CANN包，也不设置影响所有目标的全局`CPLUS_INCLUDE_PATH`。原文件SHA256校验和此前两个补丁仍保留，vendor提交不变。

开发端发布本实验仓库更新后，238宿主机执行：

```bash
cd /ssd_2/yuanwenliu/sglang
git status --short --branch
git pull --ff-only origin feat/qwen3-cam-detour-conflict
ls experiments/qwen3_vllm_cam82/patches/umdk-platform-include.patch
```

有本地改动先处理。随后重跑4.5的原Docker构建命令，不清缓存、不重新下载、不更新UMDK分支。应先看到`CAM82_PLATFORM_INCLUDE_PATCH_OK`，进入生成项目的CMake后再看到`CAM82_PLATFORM_INCLUDE=.../include/experiment/platform`。本地CPU回归用真实主机编译器复现了未补路径时的头文件错误，补路径后成功编译模拟库，并检查该路径不影响其他目标、修补可重复执行、头文件缺失仍停止。这不是完整Ascend算子编译成功的证明；仍以238本次Docker构建最终退出成功且出现Successfully built/tagged为准，之后才能进入4.6。

#### 4.5.5 已定位：.run已生成，但--install导致只打印Usage并失败

日志已输出`[compile_single_op] output: .../CAM_ascend910b4_all_to_all_detour_ubuntu_aarch64.run`，说明单算子编译和打包已完成；随后出现Usage并返回1，失败的是安装，不是前面的编译。原构建辅助脚本使用了`bash "${packages[0]}" --install`，而本次msopgen自定义算子包列出的参数没有`--install`，不能套用CANN toolkit大安装包的安装命令。

新增`patches/umdk-op-package-install.patch`，将镜像内的安装调用修正为：

```bash
bash "${packages[0]}" --quiet --nox11 --install-path="${ASCEND_OPP_PATH}"
```

这是脚本内部命令，不要在宿主机直接执行上述变量命令。`--quiet`避免交互，`--nox11`不启动图形终端；安装路径显式选择当前CANN的OPP根目录，算子仍进入`$ASCEND_OPP_PATH/vendors/CAM`。自定义算子包直接执行安装、`--install-path=<path>`落在`<path>/vendors/<vendor_name>`的规则见[官方自定义算子部署说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/devaids/optool/atlasopdev_16_0155.html)，参数含义也可对照[Ascend自定义算子包参数表](https://github.com/Ascend/RecSDK/blob/develop/docs/en/torch/torch_rec_v1/recsdk_torch_installation_guide.md)。最终仍以本次生成包的帮助信息和238实测为准，不混用toolkit安装参数。

安装命令成功后，脚本还检查`vendors/CAM/op_api/lib/libcust_opapi.so`及`vendors/CAM/op_api/include/aclnn_all2_all_detour.h`均非空，才输出`CAM82_OP_PACKAGE_INSTALLED`并继续构建Python wheel。安装返回非零或缺少必要文件都会立即停止，不把“打印Usage”或只解压文件视为成功。此修正只改镜像内固定UMDK副本的构建辅助脚本，不改宿主机或CANN原文件，也不需要发布新的UMDK分支。

开发容器隔离验证时可另外指定`CAM_OPP_INSTALL_PATH`，安装位置改为该目录下的`vendors/CAM`，避免覆盖已有CAM安装；不设置该变量时仍使用上述`ASCEND_OPP_PATH`。隔离安装后，调用时要设置对应`ASCEND_CUSTOM_OPP_PATH`和API库路径，具体已验证命令见DEV85_VALIDATION.md。

开发端发布本实验仓库更新后，238宿主机执行：

```bash
cd /ssd_2/yuanwenliu/sglang
git status --short --branch
git pull --ff-only origin feat/qwen3-cam-detour-conflict
ls experiments/qwen3_vllm_cam82/patches/umdk-op-package-install.patch
```

有本地改动先处理。随后重跑4.5的原Docker构建命令。不要清缓存或重新下载；失败的RUN层会重新执行其中的编译与安装步骤，前面成功层仍可复用。先看到补丁标记`CAM82_OP_INSTALL_PATCH_OK`，再检查实际安装后的`CAM82_OP_PACKAGE_INSTALLED`；这两条仍不表示wheel或最终镜像已完成。安装流程的CPU模拟回归已覆盖参数传递、安装失败、返回成功但文件缺失的情况；真实.run安装、后续wheel及最终导入仍需238验证。只有本次构建退出成功且出现Successfully built/tagged，才能进入4.6。

#### 4.5.6 已定位并在8.5容器修正：CAM导入bad_alloc与inner API缺失

238已确认Torch、torch_npu导入成功，CAM扩展导入崩溃；Torch、libcam及扩展均使用CXX11 ABI=1，容器无实际内存限额。该结果不能据此改成ABI=0，也不能把问题归因于32GB HBM容量。Mooncake尚未执行到。

在8.146.225.22实际容器`cam_lyw_dev_85`（CANN8.5、Python3.11.15、Torch2.8.0+cpu、torch_npu2.8.0.post2）中，单独dlopen新编译的libcam也复现同类崩溃。GDB调用栈显示全局`std::regex ipv4_regex`在动态库加载时构造，经正则编译、字符串赋值走到一次约247TB的异常分配请求并抛出bad_alloc。定位到了失败位置，但未据此断言究竟是哪一个底层运行库存在更深层的兼容性问题。

新增`patches/umdk-ipv4-no-static-regex.patch`：移除这处全局正则和regex依赖，IPv4校验改成无动态分配的逐字符检查，保留原规则：4段十进制数、每段1–3位、值0–255、允许前导零、不接受空段/符号/其他字符。通信算法、数据布局、选路策略及Torch ABI不变。原regex与新实现的Linux C++等价回归覆盖54,107个用例，避免简单使用inet_pton改变前导零规则。

扩展加载恢复后，又在主动加载`libcust_opapi.so`时发现`aclnnInnerAll2AllDetour`未定义。自动生成的inner源文件存在，但模板`aclnn_*.cpp`通配符没有包含`aclnnInner_*.cpp`；只生成.so或只import扩展不能发现这个错误。现已扩展现有CMake补丁，把inner源文件一起编译链接，重新打包安装，并检查内外两组All2AllDetour符号均已定义。

新增`build_helpers/verify_cam_import.py`：逐个打印Torch/NPU/CAM导入阶段，检查5个CAM接口的dispatcher schema，再加载实际安装的算子API库并检查两项外部aclnn入口。Dockerfile最终RUN也执行此检查；成功输出`CAM_IMPORT_AND_API_PASS (no NPU execution)`后仍继续原有Mooncake等导入检查，而非绕过原生模块验证。

开发机已生成并验证的产物、日志、环境和复验命令见[8.5算子验证记录](DEV85_VALIDATION.md)。**这些产物不是8.2可用性的认证，也不是NPU多卡通信实测，不要直接复制8.5 wheel/.run到238。** 用户当前要求仅编译算子，未清理服务器已有数据，也未在开发机构建完整8.2镜像。

开发端发布本实验仓库更新后，238宿主机更新实验分支，确认包含`patches/umdk-ipv4-no-static-regex.patch`和`build_helpers/verify_cam_import.py`，再重跑4.5原Docker构建命令。无需更新vendor的9c0bbdad提交或重新下载模型；仍以238本次构建退出成功且最终出现Successfully built/tagged为准，之后才能进入4.6。

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

**228、238都执行（相同命令，宿主机，仅首次创建）：** 使用新容器名，不替换旧实验；容器已存在则跳过创建，直接进入：

```bash
cd /ssd_2/yuanwenliu/sglang
bash experiments/qwen3_vllm_cam82/create_container.sh qwen3-vllm:cam82-v1 qwen3-vllm-cam82 --print-only
bash experiments/qwen3_vllm_cam82/create_container.sh qwen3-vllm:cam82-v1 qwen3-vllm-cam82
```

**228、238都执行（相同命令，各开一个宿主机窗口）：**

```bash
docker exec -it qwen3-vllm-cam82 bash
```

**仅228，在刚进入的容器内加载环境：**

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
```

**仅238，在刚进入的容器内加载环境：**

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
```

**228、238都执行（相同命令，容器内，完成各自source之后）：**

```bash
python experiments/qwen3_vllm_cam82/check.py --device --model
python experiments/qwen3_vllm_cam82/test_cpu.py
```

需`CAM82_CHECK_PASS`和CPU测试成功。包检查不等于CAM/PD运行成功。不能source旧qwen3_cam/env.sh或qwen3_vllm/env.sh；不要设置ASCEND_RT_VISIBLE_DEVICES，物理选择由新worker明确处理。模型离线；228/238的KV与DP通信必须联网。

**两机运行环境修正（228已实测通过，238待验证）：** 新`env.sh`会取消`HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`两项覆盖，让HCCL按默认拓扑选择。旧脚本只设`HCCL_INTRA_ROCE_ENABLE=0`会触发本环境的EI0001参数校验错误；不能再单独设置这一项，也不沿用此前Direct TE的机内RoCE开关。取消覆盖不等于证明流量必走HCCS，实际链路仍需profile/计数核实，所有对照组保持相同配置。

新脚本同时将`${ASCEND_OPP_PATH}/vendors/CAM`加入`ASCEND_CUSTOM_OPP_PATH`，将其`op_api/lib`目录加入`LD_LIBRARY_PATH`，并保持toolkit运行库优先。原因是CAM扩展运行时按文件名`libcust_opapi.so`加载API；仅在构建时按绝对路径加载通过，不代表运行时能够找到。缺少该配置时，会在第一次CAM调用报`aclnnAll2AllDetour ... not in libopapi.so`，即使`CamCommInit`已经成功。

此前针对第6节probe的环境脚本修正，只需重新source，不改算子二进制。**但如果已经进入第7节并遇到D的`Explicit role/mode and full physical visibility required`，必须先执行下方5.1，将两机容器中的Python适配包更新到0.1.1；仅拉仓库或重新source不足以修复这个模型启动问题。** 已运行进程不会自动改变环境或重新加载代码。

适配包在镜像里安装，不会仅因宿主机拉新代码而自动更新。后续纯Python修订可在新CAM82容器内离线执行`python -m pip install --no-deps --no-build-isolation --no-index -e experiments/qwen3_vllm_cam82/adapter`，记录HEAD并重跑关口；涉及CAM/CANN/native库版本变化必须重新构建镜像。不把适配包安装到旧CANN9/SGLang容器。

**228、238都执行（相同命令，宿主机）：** 各查8张NPU IP/链路，无须贴所有listen端口；失败时回传相关卡输出：

```bash
for i in 0 1 2 3 4 5 6 7; do /usr/local/Ascend/driver/tools/hccn_tool -i "$i" -ip -g; /usr/local/Ascend/driver/tools/hccn_tool -i "$i" -link -g; done
```

### 5.1 已有镜像的模型启动修复：两机离线更新Python适配包

适用实测：两机镜像内仍是`site-packages/qwen_cam82/worker.py`旧版本；228的P在创建MoE HCCL通信域时报EI0001环境变量错误，D在worker构造阶段被完整物理可见性检查拒绝。`vllm._C`启动WARNING及`Engine core initialization failed`并非这两处根因，不因此重新安装vLLM/Torch/CANN。

修复逻辑：P/D都经`qwen_cam82.entrypoint`进入原生vLLM CLI，在导入CLI/创建引擎前清除两个HCCL机内覆盖变量。D通过只针对本实验worker和D8/TP4/localDP1布局的进程内适配，在父进程创建DP引擎前阻止自动裁卡；不是在已导入NPU代码的worker内临时unset设备变量。其他worker仍用原生逻辑，原vLLM文件不改写，版本/源码保护保留。worker仍拒绝设备被裁剪，检查8卡可见及当前物理卡，并按角色选择228 P=0–3、228 D=4–7、238 D=0–3。MoE并行域、KV transport、CAM算子二进制不变；该修复需两机实际模型启动验收，CPU/mock测试不代表模型已跑通。

先在各服务前台窗口正常结束自己的proxy/P/D，若在CAM模式则等通信停止后再退出238 relay，确认自己的worker已退出。不要在模型仍运行时覆盖包，不停止他人任务。不需要重建、重新传输镜像，不需要重新编译算子或下载权重。

**228、238都执行（相同命令，宿主机或当前挂载仓库的容器内二选一）：** 联网代理按需先准备；如果Git status显示自己的本地修改，先处理再拉取，不覆盖。下面的`safe.directory`只对本次Git命令信任这个明确目录，用于容器root与宿主机属主不同触发的`dubious ownership`；不设全局通配符、不chown仓库：

```bash
cd /ssd_2/yuanwenliu/sglang
git -c safe.directory=/ssd_2/yuanwenliu/sglang status --short --branch
git -c safe.directory=/ssd_2/yuanwenliu/sglang pull --ff-only origin feat/qwen3-cam-detour-conflict
git -c safe.directory=/ssd_2/yuanwenliu/sglang rev-parse HEAD
```

**228、238都执行（相同命令，宿主机新窗口）：** 已在正确容器内则不重复进入：

```bash
docker exec -it qwen3-vllm-cam82 bash
```

**228、238都执行（相同命令，容器内）：** 仅安装本地纯Python适配包，`--no-deps`不改动推理依赖，`--no-index`不联网：

```bash
cd /ssd_2/yuanwenliu/sglang
python -m pip install --no-deps --no-build-isolation --no-index --force-reinstall experiments/qwen3_vllm_cam82/adapter
python -c 'import importlib.metadata as m, importlib.util as u; v=m.version("qwen-vllm-cam82"); print("CAM82_ADAPTER_VERSION", v); assert v=="0.1.1"; print("CAM82_ENTRYPOINT", u.find_spec("qwen_cam82.entrypoint").origin)'
```

仅228，容器内：

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
```

仅238，容器内：

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
```

**228、238都执行（相同命令，容器内，完成各自source之后）：**

```bash
python experiments/qwen3_vllm_cam82/check.py --device --model
python experiments/qwen3_vllm_cam82/test_cpu.py
```

均通过后，228已通过的第6节probe不必因纯Python启动修复重测；238若仍未通过第6节，先完成它。然后按第7节的完整窗口命令重新开始off基线，保留新时间戳日志，不能拿旧`p-off-20261016-030626.log`代替新一轮结果。每个P/D入口应输出`CAM82_STARTUP_READY ... HCCL_intra_overrides=unset`；两处D还应输出`CAM82_DP_VISIBILITY ... all8_visible=True`。各worker应输出`CAM82_WORKER_DEVICE`，核对物理卡：228 P为0/1/2/3，228 D为4/5/6/7，238 D为0/1/2/3。缺少标记、卡号不符或有新异常时停止后续请求，回传该轮首个实际异常；不取消保护硬跑。

### 5.2 KV连接器类名修正：MooncakeConnector，而非MooncakeConnectorV1

228新一轮06:40日志的实际错误为`AttributeError: module 'vllm_ascend.distributed.mooncake_connector' has no attribute 'MooncakeConnectorV1'`，发生在scheduler通过KVConnectorFactory解析类时。容器导出列表和固定版本源码均确认：实际类名为`MooncakeConnector`，继承`KVConnectorBase_V1`，所以没有V1后缀不意味着使用旧接口。此前本实验启动配置写错类名；不是缺Mooncake包、不是TE建链错误，也不以这个错误判断OOM。

`launch.py`现为P及两端D的所有TP模式统一设置`kv_connector=MooncakeConnector`，保留`kv_connector_module_path=vllm_ascend.distributed.mooncake_connector`及原有KV角色、端口和并行配置。启动及`check.py --device`增加`CAM82_KV_CONNECTOR_OK`检查：通过实际原生工厂解析类，确认来自指定Ascend模块并继承V1基类；若注册表选中其他实现则停止。检查只解析类，不构造scheduler/worker/TransferEngine，也不证明KV已搬运成功。

**228、238都执行（相同命令，宿主机或当前挂载仓库的容器内二选一）：** 先结束自己的失败服务及残留worker，按需准备Git联网代理；有本地改动先处理，不覆盖：

```bash
cd /ssd_2/yuanwenliu/sglang
git -c safe.directory=/ssd_2/yuanwenliu/sglang status --short --branch
git -c safe.directory=/ssd_2/yuanwenliu/sglang pull --ff-only origin feat/qwen3-cam-detour-conflict
```

本次只修改挂载仓库的`launch.py`、`check.py`及文档/测试，不修改已安装适配包。**已完成5.1并确认适配包为0.1.1的两机，无需再安装包、构建镜像、编译算子或传输Docker。** 尚未完成5.1的机器仍需先完成5.1。

**228、238都执行（相同命令，宿主机新窗口）：** 已在实验容器内则跳过这一行：

```bash
docker exec -it qwen3-vllm-cam82 bash
```

仅228，容器内：

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
```

仅238，容器内：

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
```

**228、238都执行（相同命令，容器内，完成各自source之后）：**

```bash
python experiments/qwen3_vllm_cam82/check.py --device --model
python experiments/qwen3_vllm_cam82/test_cpu.py
```

需出现`CAM82_KV_CONNECTOR_OK vllm_ascend.distributed.mooncake_connector.MooncakeConnector`及`CAM82_CHECK_PASS`，然后按第7节原命令重新启动三个服务。新打印的COMMAND中应为`"kv_connector": "MooncakeConnector"`；若仍为V1后缀，运行的是旧启动脚本。不要手工给第三方模块添加别名或修改源码指纹来绕过检查。当前修正针对已确认的类名异常，完整P/D、EP8与真实KV迁移仍待验收。

## 6 先验收CAM三种TP语义，再启动模型

2026-10-16用户反馈：228在取消上述HCCL变量覆盖并补齐CAM库/算子包路径后，HCCL与CAM两项测试退出码均为0。238尚未执行本轮修正后的测试；不能以228结果代替238验收。以下为每台机器独立的机内测试，228/238不需要同时启动。

**228、238都执行（相同命令，各自新开宿主机窗口）：** 进入已创建的实验容器；尚未创建时先完成第5节。228已通过本节可跳过复测，继续为238完成本节：

```bash
docker exec -it qwen3-vllm-cam82 bash
```

**仅228，容器内：**

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
```

**仅238，容器内：**

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
```

**228、238都执行（相同命令，容器内，完成各自source之后）：**

```bash
printenv | grep -E '^HCCL_INTRA_(PCIE|ROCE)_ENABLE='
python -c 'import torch, torch_npu, ctypes; lib=ctypes.CDLL("libcust_opapi.so"); print(lib.aclnnAll2AllDetour); print(lib.aclnnAll2AllDetourGetWorkspaceSize); print("CAM_API_LOOKUP_OK")'
```

`printenv | grep`应无输出（退出码1表示没有这两个变量，正常）；Python检查应打印`CAM_API_LOOKUP_OK`。这是按文件名加载的检查，不进行NPU通信；若失败先停止，不继续probe。不要重新安装其他版本算子或复制开发机8.5产物。

确认自己的旧服务/probe已正常退出，并获得两机16卡实验窗口；不要kill他人任务。**228、238都执行（相同命令，容器内）：** 先测HCCL，**只用于probe，不用于模型启动**：

```bash
set -o pipefail
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29631 experiments/qwen3_vllm_cam82/smoke_tp.py --mode hccl-a2a 2>&1 | tee "experiments/qwen3_vllm_cam82/results/tp-hccl-$(date +%Y%m%d-%H%M%S).log"
echo "HCCL测试退出码=$?"
```

先确认各自退出码0且出现`CAM_TP_SMOKE_PASS mode=hccl-a2a`。**228、238都执行（相同命令，容器内）：** 再测CAM；前一项失败不要继续：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29631 experiments/qwen3_vllm_cam82/smoke_tp.py --mode cam-detour 2>&1 | tee "experiments/qwen3_vllm_cam82/results/tp-cam-$(date +%Y%m%d-%H%M%S).log"
echo "CAM测试退出码=$?"
```

需两台机的两种mode分别出现`CAM_TP_SMOKE_PASS`、退出码0且全部进程退出，才能进入第7节。它同时运行两个TP4子组，测试AllReduce/ReduceScatter/AllGather、fp32/bf16、跨chunk、小输入、维度变换和空输入，对照HCCL结果。发现卡号错配/挂起/不一致，停止启动模型；两组同时使用CAM窗口是重要关口，不用CPU代数验证替代。probe在结束时通过进程退出释放自己的窗口，不手动清其他进程HBM。

## 7 原生P4+D8：TP off对照（MoE保持原生All-to-All）

本节需要228的4个SSH窗口和238的1个SSH窗口；各前台服务保持运行，不要退出。每块命令都写全进入容器和加载环境的步骤：第一行在对应宿主机执行，进入容器后再执行后续行。先不用中继进程，因为off不创建CAM窗口。模型服务**不用torchrun**。启动228的P之后，接着启动两处D，不等待单侧D独立就绪；两处D需互相连接。

228窗口1（P4）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role p-228 --tp-mode off 2>&1 | tee "experiments/qwen3_vllm_cam82/results/p-off-$(date +%Y%m%d-%H%M%S).log"
```

228窗口2（D8 DP rank0，物理4–7）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-228 --tp-mode off 2>&1 | tee "experiments/qwen3_vllm_cam82/results/d228-off-$(date +%Y%m%d-%H%M%S).log"
```

238窗口1（D8 DP rank1，物理0–3）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-238 --tp-mode off 2>&1 | tee "experiments/qwen3_vllm_cam82/results/d238-off-$(date +%Y%m%d-%H%M%S).log"
```

两端D需同时启动建立EP8；不是两个独立D4。没有SGLang PD store/7100，没有Mooncake master；KV使用P2PHANDSHAKE的旧Ascend transport。日志需`NATIVE_QWEN_MOE_ALL2ALL ep=4/8`及物理/TP组映射。32GB HBM先上下文2048、max-seqs4、memory0.75；OOM区分权重峰值、profile临时张量与KV预算，不保证仅缩上下文能解决。

诊断时可在三处serve命令均追加`--max-model-len 1024 --max-num-seqs 2 --gpu-memory-utilization 0.70`，同时缩短客户端输入。原生scheduler可能调整profile/批处理token预算，需记录实际值；启动权重峰值导致的OOM不能仅靠这些参数保证解决。比较TP模式时三组预算保持一致。

228窗口3（新开SSH窗口，检查三处health，全部成功后启动proxy；health未成功不要继续）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
curl --noproxy '*' -fsS http://10.155.205.228:30010/health
curl --noproxy '*' -fsS http://10.155.205.228:30011/health
curl --noproxy '*' -fsS http://10.155.205.238:30011/health
python -u experiments/qwen3_vllm_cam82/launch.py proxy
```

228窗口4发送请求、保存输出与负载（复用CPU客户端，不导入CANN9适配）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
curl --noproxy '*' -fsS http://10.155.205.228:30080/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen3","prompt":"Explain prefill and decode separation.","max_tokens":32,"temperature":0}'
python experiments/qwen3_vllm/compare_outputs.py collect --url http://10.155.205.228:30080 --output experiments/qwen3_vllm_cam82/results/pd8-off-output.json
python experiments/qwen3_vllm/workload.py --requests 20 --rps 0.2 --concurrency 2 --output "experiments/qwen3_vllm_cam82/results/pd8-off-$(date +%Y%m%d-%H%M%S).jsonl"
```

必须确认真实KV转移完成，不仅health/HTTP200；检查P/D connector日志及实际搬运，排除重算回退。若原生Qwen的非整除token/空DP输入失败，停下回传原生错误，本轮不绕过MoE算法。默认原生prefill SP使用原生padding，不能声称覆盖所有模型输入情况。

## 8 换成CAM TP：三处服务＋238中继

先正常结束自己的proxy与三处模型，确认worker退出。两机CAM域的所有owner每轮一起新建，不能沿用上一轮残留窗口。本节需要228的4个SSH窗口和238的2个SSH窗口。下面写全各窗口的命令；如果复用的窗口仍在该容器内，跳过该块第一行`docker exec`，从source开始，不要嵌套执行Docker。

238窗口2先启动四个中继owner（随后等D窗口加入CAM域，暂未打印READY属正常）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
set -o pipefail
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29632 experiments/qwen3_vllm_cam82/relay.py 2>&1 | tee "experiments/qwen3_vllm_cam82/results/relay-$(date +%Y%m%d-%H%M%S).log"
```

228窗口1（P4）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role p-228 --tp-mode cam-detour 2>&1 | tee "experiments/qwen3_vllm_cam82/results/p-cam-$(date +%Y%m%d-%H%M%S).log"
```

228窗口2（D8 DP rank0）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-228 --tp-mode cam-detour 2>&1 | tee "experiments/qwen3_vllm_cam82/results/d228-cam-$(date +%Y%m%d-%H%M%S).log"
```

238窗口1（D8 DP rank1）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-238 --tp-mode cam-detour 2>&1 | tee "experiments/qwen3_vllm_cam82/results/d238-cam-$(date +%Y%m%d-%H%M%S).log"
```

务必启动全部三个模型＋238中继。228的P/D互相提供中继窗口，238由四个owner补齐域。CAM端口为各自host的127.0.0.1:10067，512MiB窗口/卡；不是跨机10067，更不联系8.146。缺owner时bootstrap可等至600秒超时；不能只启动P或只启动一端D并期待CAM就绪。

验收`CAM_RELAY_READY`四次、每模型rank的`CAM_TP_READY`，请求后`CAM_TP_CALL mode=cam-detour op=all_reduce/all_gather/reduce_scatter`以及原生MoE标记。三种调用不要求每个request/每个角色同时出现：prefill SP主要AG/RS、decode主要AR，logits也可能AG。没有调用日志的类型不能当作已测。

228窗口3（检查三处health，全部成功后启动本轮proxy）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
curl --noproxy '*' -fsS http://10.155.205.228:30010/health
curl --noproxy '*' -fsS http://10.155.205.228:30011/health
curl --noproxy '*' -fsS http://10.155.205.238:30011/health
python -u experiments/qwen3_vllm_cam82/launch.py proxy
```

228窗口4（本轮proxy就绪后，收集输出、比较和发送相同小负载）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
python experiments/qwen3_vllm/compare_outputs.py collect --url http://10.155.205.228:30080 --output experiments/qwen3_vllm_cam82/results/pd8-cam-output.json
python experiments/qwen3_vllm/compare_outputs.py compare --native experiments/qwen3_vllm_cam82/results/pd8-off-output.json --alltoall experiments/qwen3_vllm_cam82/results/pd8-cam-output.json
python experiments/qwen3_vllm/workload.py --requests 20 --rps 0.2 --concurrency 2 --output "experiments/qwen3_vllm_cam82/results/pd8-cam-$(date +%Y%m%d-%H%M%S).jsonl"
```

工具的`--alltoall`只是第二个快照的参数名，此处比较的是TP模式，不是改MoE。短prompt一致仅是smoke；归约顺序变化可能导致BF16/greedy差异，差异需核对logits/质量，不自动忽略。至少覆盖短/长prefill、混合batch、D一侧空负载、多轮连续请求后才测性能。

### 8.1 额外对照：hccl-a2a

按第10节退出自己的proxy、P与两端D，确认通信停止后退出238中继及其worker。不启动中继。这组保留相同分块、padding、本地归约及控制barrier，仅交换payload走HCCL，分离分解开销与CAM路径变化。off不是“没有TP通信”，而是原生TP。

228窗口1（P4）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role p-228 --tp-mode hccl-a2a 2>&1 | tee "experiments/qwen3_vllm_cam82/results/p-hccl-a2a-$(date +%Y%m%d-%H%M%S).log"
```

228窗口2（D8 DP rank0）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-228 --tp-mode hccl-a2a 2>&1 | tee "experiments/qwen3_vllm_cam82/results/d228-hccl-a2a-$(date +%Y%m%d-%H%M%S).log"
```

238窗口1（D8 DP rank1）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 238
set -o pipefail
python -u experiments/qwen3_vllm_cam82/launch.py serve --role d8-238 --tp-mode hccl-a2a 2>&1 | tee "experiments/qwen3_vllm_cam82/results/d238-hccl-a2a-$(date +%Y%m%d-%H%M%S).log"
```

228窗口3（检查三处health，全部成功后启动本轮proxy）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
curl --noproxy '*' -fsS http://10.155.205.228:30010/health
curl --noproxy '*' -fsS http://10.155.205.228:30011/health
curl --noproxy '*' -fsS http://10.155.205.238:30011/health
python -u experiments/qwen3_vllm_cam82/launch.py proxy
```

228窗口4（本轮proxy就绪后）：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
python experiments/qwen3_vllm/compare_outputs.py collect --url http://10.155.205.228:30080 --output experiments/qwen3_vllm_cam82/results/pd8-hccl-a2a-output.json
python experiments/qwen3_vllm/compare_outputs.py compare --native experiments/qwen3_vllm_cam82/results/pd8-off-output.json --alltoall experiments/qwen3_vllm_cam82/results/pd8-hccl-a2a-output.json
python experiments/qwen3_vllm/workload.py --requests 20 --rps 0.2 --concurrency 2 --output "experiments/qwen3_vllm_cam82/results/pd8-hccl-a2a-$(date +%Y%m%d-%H%M%S).jsonl"
```

## 9 冲突与影响评估

本轮能测**CAM TP多路径、原生MoE All-to-All、正常P→D KV**是否共享资源并产生干扰；不包含KV Pool恢复。尤其228的TP子组可借另一角色的卡中继，但不能凭逻辑箭头认定物理链路被KV使用。MoE域与TP组不同，不代表物理链路一定不共享。

按off / hccl-a2a / cam-detour分别运行相同请求序列、SP/并行/模型/预算，固定到达率与输出长度，每组至少3轮。先低负载，再增加并发/请求长度。**仅228窗口4执行：** 保持当前轮次的模型/proxy运行；新SSH窗口先执行第一行进入容器，复用已在容器内的窗口则从source开始：

```bash
docker exec -it qwen3-vllm-cam82 bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/env.sh 228
python experiments/qwen3_vllm/workload.py --requests 200 --rps 0.5 --concurrency 4 --prefix-tokens 1024 --output "experiments/qwen3_vllm_cam82/results/load-$(date +%Y%m%d-%H%M%S).jsonl"
python experiments/qwen3_vllm/workload.py --summarize experiments/qwen3_vllm_cam82/results/本轮结果.jsonl
```

记录成功率、TTFT/TPOT P50/P99、吞吐与SLO违反率；200样本仅起步，不能可靠估计极端尾延迟。输入＋输出不得超过2048。观察到后端变快/变慢不等于证明冲突。

device profile仅用于诊断轮，不在纯性能轮开启。在第7/8节对应模型窗口中，**进入容器并source后、运行serve之前**分别执行以下配置（这不会让已经运行的引擎自动开始profile）：

仅228窗口1（P4）：

```bash
export VLLM_TORCH_PROFILER_DIR=/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/profiles/run1-p228
```

仅228窗口2（D8 DP rank0）：

```bash
export VLLM_TORCH_PROFILER_DIR=/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/profiles/run1-d228
```

仅238窗口1（D8 DP rank1）：

```bash
export VLLM_TORCH_PROFILER_DIR=/ssd_2/yuanwenliu/sglang/experiments/qwen3_vllm_cam82/profiles/run1-d238
```

通过每个引擎HTTP端点POST `/start_profile`，短请求窗口后POST `/stop_profile`；proxy不提供这些控制。不同轮次另用新目录保留结果。

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
