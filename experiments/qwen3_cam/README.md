# Qwen3-30B-A3B：A2 两机部署与流量冲突验证

更新：2026-10-13。服务器以228、238指代；两机各8张910B4、32GB HBM，机内FullMesh/HCCS，个人工作目录均为 /ssd_2/yuanwenliu。

执行顺序：拉代码 → 私有配置 → 构建/复制镜像 → 每机创建一个实验容器 → 下载权重 → 原生PD → CAM正确性 → KV Pool → 三模式对照与冲突测量。每个命令块逐行执行；不用输入续行反斜杠。

**网络边界：有卡服务器与无卡开发服务器8.146.225.22不互通。代码通过GitHub/GitCode发布、拉取；运行时只在228/238之间通信，不连接开发服务器。两台有卡服务器彼此仍需主机控制网和NPU数据网互通。**

当前交付为源码、配置和工具。用户已反馈镜像构建、权重准备及228预检查通过；实际P/D启动仍失败：D的NPU Tensor布尔判断错误已修复，P的TransferEngine初始化失败仍需按第5/14节复测。完整通信正确性、模型质量和性能尚未验收，不能将预检查通过当作运行成功。

## 1 实验布局与版本

| 主机 | 物理卡 | 角色 | 并行设置 |
| --- | --- | --- | --- |
| 228 | 0–3 | P4 | Attention TP4、EP4 |
| 228 | 4–7 | D8前半 | D全局启动TP8、Attention DP2×TP4、EP8 |
| 238 | 0–3 | D8后半 | 与228卡4–7组成同一D组；专家内部TP1 |
| 238 | 4–7 | KV Pool4 / CAM中继 | 不加载模型，每卡初始8GB HBM Pool |

P与D的MoE域独立。CAM按本机物理卡号组成8-rank内存资源域：228由P4+D4提供窗口，238由D4+Pool4提供窗口。某次Attention通信仍只有4个参与rank，其余卡提供中继。**容器必须看到全部8卡，不隐藏、不重编号**；模型用base-gpu-id选择运行卡。

- SGLang：基于 f620d73，分支 feat/qwen3-cam-detour-conflict，仓库 https://github.com/COLDTEA-L/qwen3。
- UMDK：基于 all_gather_4_scheduler / bf74c6db，配套分支 feat/qwen3-cam-detour-build，预期HEAD为9c0bbdadfcfe882426cabdce5cdf2d5d2fce92ef。
- 模型：原版 Qwen/Qwen3-30B-A3B BF16；48层、128专家、top-k=8、4 KV heads、head_dim=128。权重仅在A2下载。

两机代码、镜像、模型校验值必须一致，不临时切最新main。

## 2 已确认的软件栈

| 内容 | 已确认值 / 本次处理 |
| --- | --- |
| 宿主机 | aarch64、EulerOS2.0 SP10、driver25.5.2 |
| 主机控制网 | 两机均enp67s0f5；不是NPU RoCE接口/IP |
| 238的基础镜像ImageID | sha256:39c97f336132e02b9e7066428ddde439e763623e4024c1f405049f79b5be4ad6 |
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

| 内容 | 228/238统一使用的路径 |
| --- | --- |
| SGLang | /ssd_2/yuanwenliu/sglang |
| 实验UMDK（不覆盖旧仓） | /ssd_2/yuanwenliu/umdk-qwen3-cam |
| 模型 | /ssd_2/yuanwenliu/models/Qwen3-30B-A3B |
| 下载venv | /ssd_2/yuanwenliu/qwen3-download-venv |
| 镜像tar / 校验清单 | /ssd_2/yuanwenliu/ |
| 私有配置 / 结果 / profile | SGLang实验目录的runtime-config / results / profiles |

除非写“容器内”，均在宿主机执行。代码/权重/镜像准备不加载模型；从第5节起须有获准使用的整套16卡窗口。不要停止他人任务。

### 4.1 228/238拉取已发布的实验源码

本节仅包含实验机器的代码获取操作。SGLang从GitHub拉取，UMDK从GitCode拉取；先确认代码托管站已有所需实验分支及本次修订，再在228/238执行。实验机器只拉取代码，不承担源码发布，也不执行ssh/scp到开发服务器。发布指令只在对话中提供，不放入本文档。

**开发服务器的UMDK已经准备好了**：/home/liuyuanwen/umdk 当前分支为 feat/qwen3-cam-detour-build，HEAD为9c0bbdadfcfe882426cabdce5cdf2d5d2fce92ef，origin为 https://gitcode.com/yuanwenliu/umdk.git。这里“更新”是发布这个实验分支，不是pull最新master；否则可能丢掉需要的detour实现。不需要新建GitHub UMDK仓库。

228、238各执行：

```bash
test -w /ssd_2/yuanwenliu
git clone --branch feat/qwen3-cam-detour-conflict --single-branch https://github.com/COLDTEA-L/qwen3.git /ssd_2/yuanwenliu/sglang
git clone --branch feat/qwen3-cam-detour-build --single-branch https://gitcode.com/yuanwenliu/umdk.git /ssd_2/yuanwenliu/umdk-qwen3-cam
git -C /ssd_2/yuanwenliu/sglang rev-parse HEAD
git -C /ssd_2/yuanwenliu/umdk-qwen3-cam rev-parse HEAD
```

**228旧UMDK不动**；实验另外clone到umdk-qwen3-cam。若GitCode提示找不到分支，先确认所需分支已在代码托管站可见，不改用旧master或别的all-to-all分支。若有卡机器也不能访问GitHub/GitCode，需用个人电脑或获准的中转渠道传输源码，不直接连接开发服务器。

已有本实验目标目录时不重复clone。确认工作区干净、origin正确，再更新：

```bash
git -C /ssd_2/yuanwenliu/sglang status --short --branch
git -C /ssd_2/yuanwenliu/sglang remote -v
git -C /ssd_2/yuanwenliu/sglang switch feat/qwen3-cam-detour-conflict
git -C /ssd_2/yuanwenliu/sglang pull --ff-only
git -C /ssd_2/yuanwenliu/umdk-qwen3-cam status --short --branch
git -C /ssd_2/yuanwenliu/umdk-qwen3-cam remote -v
git -C /ssd_2/yuanwenliu/umdk-qwen3-cam switch feat/qwen3-cam-detour-build
git -C /ssd_2/yuanwenliu/umdk-qwen3-cam pull --ff-only
```

有本地改动、remote不符或ff-only失败就停下，不reset。两机SGLang HEAD一致，UMDK均应为上述9c0bbdad。

### 4.2 两机分别生成运行配置：为什么要做

这一步不是建立“无卡控制器”，也不是复制SSH密钥。它给有卡环境中的服务提供**对端地址、端口、模型路径和容量约束**：

| 文件 | 谁读取 | 作用 |
| --- | --- | --- |
| deployment.env | env.sh、launch.sh、传输命令 | 228/238地址、控制网接口、个人路径；228:7100为PD store，228:29500为D建组 |
| meta.json | 238的MemCache MetaService | 238:5000元数据、:6000配置、:8100指标 |
| pool.json | 238卡4–7 sidecar | 同一元数据域、DEVICE_RDMA；每卡8GB正容量Pool |
| client.json | 228的P HiCache / probe | 找到238的Pool；HBM/DRAM=0，不让P成为额外存储池 |

所有IP都是有卡环境内的端点；**不包含8.146.225.22，不保存密码/token**。Pool的world_size=32是最大接入rank数，不是模型卡数。SSH_USER只供两台有卡机器之间复制镜像/权重使用，不参与推理调度。

为了不依赖私有文件复制，**228和238分别执行同一生成器**，填相同的完整228地址、238地址与账户yuanwenliu：

```bash
cd /ssd_2/yuanwenliu/sglang
python3 experiments/qwen3_cam/configure_runtime.py
```

脚本交互询问地址，不把私网地址写进Git跟踪文件。两机得到相同的四个文件，存放在runtime-config；该目录被Git忽略。只有文件角色不同，不是两份拓扑定义。可用下面命令核对两机校验值相同：

```bash
sha256sum experiments/qwen3_cam/runtime-config/deployment.env experiments/qwen3_cam/runtime-config/meta.json experiments/qwen3_cam/runtime-config/pool.json experiments/qwen3_cam/runtime-config/client.json
```

已用旧版生成过配置时，旧变量名/模型路径不能直接复用。先备份旧目录再重新生成，不覆盖；以下仅在旧目录存在且备份名未占用时执行：

```bash
test -d experiments/qwen3_cam/runtime-config && test ! -e experiments/qwen3_cam/runtime-config.before-path-update && mv experiments/qwen3_cam/runtime-config experiments/qwen3_cam/runtime-config.before-path-update
python3 experiments/qwen3_cam/configure_runtime.py
```

备份目录也被Git忽略。若备份名已存在或生成器报文件已存在，停下核对，不删除旧文件。

### 4.3 238构建镜像

本节在**238宿主机**执行，不是在旧容器里执行。已确认238使用legacy builder，不支持此前命令中的BuildKit日志参数；现改用每条命令的DOCKER_BUILDKIT=0，不安装buildx、不修改Docker全局配置。DEPRECATED提示只是警告，真正的构建错误要单独处理。[Docker旧构建器说明](https://docs.docker.com/reference/cli/docker/image/build/)。

镜像标签不等于已经存在的镜像：先构建core，再构建pool；docker save只导出，docker run才创建容器。用户已完成本轮构建；已有合格实验镜像/容器时不重复构建。本轮第5/14节源码修订通过挂载生效，不需要重新构建镜像或下载权重。**不要一次性复制本节全部命令；每一步成功后才执行下一步。**

不修改旧yuanwen_sglang_a2、不使用opensora:latest。以下标签若已存在，先核对；已是本实验合格镜像可跳过构建，否则用新标签并同步修改后续引用，不覆盖他人标签。

**第一步：检查空间、基础镜像并准备构建文件。**任一步报错就停止：

```bash
df -h /ssd_1/docker /ssd_2/yuanwenliu
docker image inspect sha256:39c97f336132e02b9e7066428ddde439e763623e4024c1f405049f79b5be4ad6 --format '{{.Id}} {{.Architecture}}'
docker tag sha256:39c97f336132e02b9e7066428ddde439e763623e4024c1f405049f79b5be4ad6 qwen3-cam:a2-base-observed
cd /ssd_2/yuanwenliu/sglang
cp python/pyproject_npu.toml experiments/qwen3_cam/docker/source-pyproject_npu.toml
```

**构建前必做：在238宿主机设置代理并测试。**以下采用HTTP/HTTPS CONNECT代理，真实地址由你输入，不提供虚构地址。代理必须在有卡环境可达，不能填不互通的无卡开发服务器；127.0.0.1指238本机，不是你的电脑。仅SOCKS5代理请先使用代理程序的HTTP/mixed端口，不要只设置ALL_PROXY。

先**单独执行下面这一行**，按提示输入完整代理URL并回车，再复制后面的命令。输入不回显，避免含认证信息的URL进入shell历史；不要把代理账号/密码发到聊天或提交到仓库：

```bash
read -r -s -p '238可用的HTTP代理URL（含协议和端口）: ' QWEN_BUILD_PROXY; printf '\n'
```

然后在**同一个238宿主机终端**导出大小写环境变量，兼容apt和pip；NO_PROXY保留已有设置，否则只排除本机地址：

```bash
export HTTP_PROXY="${QWEN_BUILD_PROXY:?请先输入代理URL}" HTTPS_PROXY="$QWEN_BUILD_PROXY" http_proxy="$QWEN_BUILD_PROXY" https_proxy="$QWEN_BUILD_PROXY"
export NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1}}"
export no_proxy="$NO_PROXY"
unset QWEN_BUILD_PROXY
```

用基础镜像检查与构建相同的host网络和代理传递方式，不占NPU、不安装包。两个地址都应返回200，否则先处理代理，不继续构建：

```bash
docker run --rm --network=host --runtime=runc -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY -e http_proxy -e https_proxy -e no_proxy --entrypoint python qwen3-cam:a2-base-observed -c 'import urllib.request; urls=("http://ports.ubuntu.com/ubuntu-ports/dists/jammy/InRelease","https://pypi.org/simple/transformers/"); [print(url,urllib.request.urlopen(url,timeout=20).status) for url in urls]'
```

--network=host让构建容器能连接238本机代理；它与后面推理容器的设备配置无关。构建时用--build-arg传HTTP_PROXY/HTTPS_PROXY/NO_PROXY及小写变量，**只写变量名，不在命令中展开含密码的URL**。这些是Docker预定义代理参数，无需在Dockerfile添加ARG/ENV；本次不把代理永久写入镜像，也不修改Docker daemon全局代理。[Docker代理构建参数说明](https://docs.docker.com/build/building/variables/#proxy-arguments)。

新开宿主机终端要重新设置代理，不能只传一次给旧容器。Ubuntu源、PyPI以及pool wheel所在的files.pythonhosted.org都须经代理可达；403/407通常需要检查访问策略/认证。构建阶段保持TLS校验，证书错误按组织CA处理；第4.5节另提供用户选择的、仅作用于权重下载进程的临时跳过校验方式，不影响Docker构建或pip。

**第二步：只构建core。**这里不会启动模型：

```bash
DOCKER_BUILDKIT=0 docker build --network=host --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy -f experiments/qwen3_cam/docker/Dockerfile.a2-core -t qwen3-cam:a2-core-v1 experiments/qwen3_cam/docker
```

必须看到Successfully tagged qwen3-cam:a2-core-v1，且下面的检查成功，才能进入第三步：

```bash
docker image inspect qwen3-cam:a2-core-v1 --format '{{.Id}} {{.Architecture}}'
```

**第三步：构建pool。**命令先确认core存在；缺失时不执行构建：

```bash
docker image inspect qwen3-cam:a2-core-v1 >/dev/null && DOCKER_BUILDKIT=0 docker build --network=host --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy -f experiments/qwen3_cam/docker/Dockerfile.a2-pool --build-arg A2_CORE_IMAGE=qwen3-cam:a2-core-v1 -t qwen3-cam:a2-pool-v1 experiments/qwen3_cam/docker
```

必须看到Successfully tagged qwen3-cam:a2-pool-v1，并检查成功：

```bash
docker image inspect qwen3-cam:a2-pool-v1 --format '{{.Id}} {{.Architecture}}'
```

core安装编译工具、源码NPU依赖、Router0.3.2；pool固定安装MemCache/MemFabric1.1.0。要求当前Python3.11的Python.h。依赖解算或headers失败，保留错误，不解除Torch/kernel约束或换A3/950 wheel。

**第四步：验证pool包版本。**238回传的磁盘快照为Docker盘剩166G、个人目录所在盘剩2.1T；每次构建前以实际df为准，不代表所有后续构建都保证空间够用。不要docker system prune或删除他人镜像。

```bash
docker run --rm --network=none --runtime=runc --entrypoint python qwen3-cam:a2-pool-v1 -m pip show torch torch-npu sgl-kernel-npu deep-ep memfabric-hybrid memcache-hybrid transformers | awk '/^(Name|Version):/ {print}'
```

预期见第2节；此条无NPU查询用runc。包版本检查通过后才进入第4.4节。后面的模型容器使用ascend。

### 4.4 复制镜像，在228和238各创建一个容器

先完成第4.3节，确认pool镜像及包版本。238执行；镜像不存在或导出失败时，不计算校验值，也不要复制tar：

```bash
docker image inspect qwen3-cam:a2-pool-v1 >/dev/null && docker save -o /ssd_2/yuanwenliu/qwen3-cam-a2-v1.tar qwen3-cam:a2-pool-v1 && sha256sum /ssd_2/yuanwenliu/qwen3-cam-a2-v1.tar
```

上条命令成功后再复制到228：

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_cam/runtime-config/deployment.env
scp /ssd_2/yuanwenliu/qwen3-cam-a2-v1.tar "${SSH_USER}@${HOST_228_IP}:/ssd_2/yuanwenliu/"
```

228核对tar校验值与238相同，再导入：

```bash
sha256sum /ssd_2/yuanwenliu/qwen3-cam-a2-v1.tar
df -h /ssd_1/docker
docker load -i /ssd_2/yuanwenliu/qwen3-cam-a2-v1.tar
```

**两机各执行以下命令**；新的容器名为qwen3-cam-a2-v3，不替换旧容器：

```bash
docker info --format '{{json .Runtimes}}'
docker image inspect qwen3-cam:a2-pool-v1 --format '{{.Id}} {{.Architecture}}'
mkdir -p /ssd_2/yuanwenliu/models/Qwen3-30B-A3B /ssd_2/yuanwenliu/qwen3-download-venv
bash /ssd_2/yuanwenliu/sglang/experiments/qwen3_cam/docker/create_container.sh qwen3-cam:a2-pool-v1 qwen3-cam-a2-v3
docker ps --filter name=qwen3-cam-a2-v3 --format '{{.Names}} {{.Image}} {{.Status}}'
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
| /ssd_2/yuanwenliu/sglang、umdk-qwen3-cam、模型目录、下载venv读写 | 仅挂载本实验工作目录，保持容器内外路径一致 |
| firmware只读 | 宿主存在该目录时保留 |

已删去OpenSora/ModelZoo目录以及整个/home、/data、/var挂载，也不挂整个/ssd_2。`--privileged`仍有很高宿主权限，收窄挂载不等于严格隔离；不要在容器内操作无关宿主资源。[Docker权限说明](https://docs.docker.com/engine/containers/run/#runtime-privilege-and-linux-capabilities)、[Ascend Runtime设备参数](https://www.hiascend.com/document/detail/en/mindcluster/2610/clustersched/schedulingug/docs/en/scheduling/usage/containerization/02_usage_on_the_docker_client.md)。

缺DCMI/日志/设备等路径时脚本停止，不自动创建系统目录。需要先查看实际docker run时，在创建前加第三个参数`--print-only`；该模式只检查并打印，不启动容器。已创建的旧容器无法通过docker exec改变runtime/mount。这版使用新名字qwen3-cam-a2-v3，工作目录/挂载都改到个人目录；旧容器与旧文件不自动移动或删除。已有合格v1镜像可以复用，脚本显式设置新工作目录。

容器先只运行sleep，不加载模型。每个新终端从宿主机进入：

```bash
docker exec -it qwen3-cam-a2-v3 /bin/bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_cam/env.sh 228
```

238的source改为238。env.sh初始化CANN/PYTHONPATH/接口/PD store地址，清除旧OpenSora的RANK等变量。**每个模型/测试终端都source一次；宿主机export不会自动进入docker exec，cd也不能改变Python导入路径。**env.sh只配置PD store地址，不启动该服务；启动见5.2。

当前env.sh会将`SGLANG_HOST_IP`固定为当前机器的配置IP，使MemFabric的session不误选其他网卡；清除大小写HTTP/HTTPS/ALL代理，并设置两机直连的NO_PROXY。推理从本地加载模型，保留offline模式。**下载终端不要source它**；公网代理只在4.5下载会话使用。若要重新下载，应重新传入代理并unset离线变量，不能把推理直连配置误当成下载配置。

### 4.5 238下载权重，复制给228

构建代理不会自动成为运行期环境变量。先回到**238宿主机**，确认当前终端仍有第4.3节的代理设置；新终端先重新设置。仅对本次下载终端传代理，不重新创建容器、不传给后续普通推理终端：

```bash
docker exec -it -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY -e http_proxy -e https_proxy -e no_proxy qwen3-cam-a2-v3 /bin/bash
```

**4.5.1 准备下载环境。**进入后使用Python3.11独立下载环境；不加载模型，不source开启offline模式的env.sh。代理同时供pip安装ModelScope和下载权重使用。已有下载venv且ModelScope安装成功时跳过创建和安装，不必重装：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python -m venv /ssd_2/yuanwenliu/qwen3-download-venv
/ssd_2/yuanwenliu/qwen3-download-venv/bin/python -m pip install modelscope
```

**4.5.2 下载权重：本次实验临时跳过TLS证书校验。**238已出现`CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`；按用户要求，下面只在这一个Python进程里将Requests的发送参数设为`verify=False`，覆盖模型文件列表查询及权重下载（含重定向），不修改已安装包、系统证书、镜像、pip或Git设置。进程结束即失效。**这会失去服务器身份验证，存在中间人篡改模型/配置的风险，仅用于可信实验网络；正式使用应安装组织CA并恢复校验。**保留`InsecureRequestWarning`提醒；它不等于下载失败。[Requests官方说明](https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification)。

在当前238下载容器终端保留代理，执行下面两行。`REQUESTS_CA_BUNDLE`为空不是关闭校验的方法；此处在发送前明确覆盖`verify`，也覆盖先前CA变量的影响：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
/ssd_2/yuanwenliu/qwen3-download-venv/bin/python -c 'import requests; send=requests.sessions.Session.send; requests.sessions.Session.send=lambda self,request,**kwargs: send(self,request,**dict(kwargs,verify=False)); from modelscope import snapshot_download; print(snapshot_download("Qwen/Qwen3-30B-A3B", local_dir="/ssd_2/yuanwenliu/models/Qwen3-30B-A3B"))'
```

ModelScope使用现有下载目录/缓存继续处理，不删除已有文件。下载报错时停在4.5.2，不执行校验、复制或模型启动；`config.json`不存在通常是下载未完成的后续结果，而非需要重建Docker。不要在另一终端同时下载到同一目录。已安装ModelScope后出现的`ms-service-profiler`缺依赖警告，不是这次HTTPS证书错误的原因，不要为此批量升级推理依赖。

后续配置好可信CA时，将上述下载命令替换为正常校验命令即可；无需重建镜像：

```bash
/ssd_2/yuanwenliu/qwen3-download-venv/bin/python -c 'from modelscope import snapshot_download; print(snapshot_download("Qwen/Qwen3-30B-A3B", local_dir="/ssd_2/yuanwenliu/models/Qwen3-30B-A3B"))'
```

**4.5.3 下载命令成功返回后，再校验文件。**下面检查配置和索引中列出的分片是否存在且非空；不能把这项检查或两机SHA256相同当作来源可信的证明，特别是下载和远端校验信息都来自未验证的连接时：

```bash
mkdir -p /ssd_2/yuanwenliu/sglang/experiments/qwen3_cam/results
/ssd_2/yuanwenliu/qwen3-download-venv/bin/python -m pip freeze > /ssd_2/yuanwenliu/sglang/experiments/qwen3_cam/results/download-versions.txt
/ssd_2/yuanwenliu/qwen3-download-venv/bin/python -c 'import json; from pathlib import Path; p=Path("/ssd_2/yuanwenliu/models/Qwen3-30B-A3B"); c=json.loads((p/"config.json").read_text()); expected={"model_type":"qwen3_moe","num_hidden_layers":48,"num_experts":128,"num_experts_per_tok":8,"num_key_value_heads":4,"head_dim":128}; assert all(c.get(k)==v for k,v in expected.items()),c; i=json.loads((p/"model.safetensors.index.json").read_text()); shards=set(i["weight_map"].values()); assert all((p/n).is_file() and (p/n).stat().st_size>0 for n in shards); print("MODEL_FILES_PASS",len(shards))'
```

**4.5.4 用scp从238复制到228，不依赖宿主机rsync。**约60GB BF16权重，不选AWQ/GPTQ/FP8。已有完整权重可跳过下载但保留校验。看到`MODEL_FILES_PASS`后，若仍在下载容器里，先执行`exit`返回238宿主机；这个exec会话的代理变量随会话结束。不要为了复制权重在容器中安装rsync，容器内安装不会给宿主机提供该命令。之后推理按第4.4节普通docker exec进入，228/238业务连接仍须直连，不经公网下载代理。

下面在**238宿主机**逐行执行；先检查ssh/scp和源目录，再确认228目标目录空间。任一步失败就停止，228至少应有完整模型所需空间；登录使用228的SSH密码/密钥，不是代理密码：

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_cam/runtime-config/deployment.env
command -v ssh
command -v scp
du -sh /ssd_2/yuanwenliu/models/Qwen3-30B-A3B
ssh "${SSH_USER}@${HOST_228_IP}" 'mkdir -p /ssd_2/yuanwenliu/models && df -h /ssd_2/yuanwenliu/models'
scp -r /ssd_2/yuanwenliu/models/Qwen3-30B-A3B "${SSH_USER}@${HOST_228_IP}:/ssd_2/yuanwenliu/models/"
```

目的地写**父目录models/**，最终路径是`/ssd_2/yuanwenliu/models/Qwen3-30B-A3B`，避免复制成两层同名模型目录。复制整个目录，不用`*`，保留隐藏文件。228已有同名文件会被覆盖，不删除目标目录中的额外文件；确认该目录只用于本次模型快照，不与其他版本共用。`scp -r`递归复制目录，见[OpenSSH说明](https://man.openbsd.org/scp#r)。

scp不提供rsync式的断点增量续传，中断后重新执行可能重传已复制的文件；不要启动使用未完成权重的服务，也不要同时运行两个复制任务。复制失败时停在4.5.4，先处理网络/登录/空间错误。

**4.5.5 复制成功后校验两机权重。**两机宿主机各生成校验清单，内容应一致：

```bash
cd /ssd_2/yuanwenliu/models/Qwen3-30B-A3B
find . -type f \( -name '*.safetensors' -o -name '*.json' \) -print0 | sort -z | xargs -0 sha256sum > /ssd_2/yuanwenliu/qwen3-model-sha256.txt
```

两机生成清单后，在**238宿主机**把238清单复制到228的独立文件名，并比较。下面命令成功且输出`MODEL_COPY_PASS`才进入4.6；`diff`有输出或文件校验失败时，不启动模型：

```bash
source /ssd_2/yuanwenliu/sglang/experiments/qwen3_cam/runtime-config/deployment.env
scp /ssd_2/yuanwenliu/qwen3-model-sha256.txt "${SSH_USER}@${HOST_228_IP}:/ssd_2/yuanwenliu/qwen3-model-sha256-238.txt"
ssh "${SSH_USER}@${HOST_228_IP}" 'diff -u /ssd_2/yuanwenliu/qwen3-model-sha256-238.txt /ssd_2/yuanwenliu/qwen3-model-sha256.txt && cd /ssd_2/yuanwenliu/models/Qwen3-30B-A3B && sha256sum -c /ssd_2/yuanwenliu/qwen3-model-sha256-238.txt && echo MODEL_COPY_PASS'
```

只下载一次快照再复制，避免两机分别下载漂移的main。权重/私有配置/results/profiles不进入Git。

### 4.6 启动前只查本实验所需资源

两机宿主机执行；确认16卡使用窗口、控制网、指定端口：

```bash
npu-smi info
ip -br addr show enp67s0f5
ss -lnt | awk 'NR==1 || $4 ~ /:(5000|6000|7100|8100|8998|10067|29500|29610|29611|29612|29620|30000|30001|8000)$/'
```

228：7100 PD store，8998 P bootstrap，29500 D建组，30000/30001 P/D HTTP，8000 Router；238：5000/6000/8100 Pool meta/config/metrics。10067为同机CAM，296xx用于torchrun测试/sidecar。启动后属于本实验的监听是正常的，不需要全量listen清单。

AICore=0不代表卡已空闲，检查进程/HBM并与任务所有者确认；不杀他人进程。主机控制网成功不等于NPU RoCE通过，网络失败时才按第14节定向查询。

## 5 先跑原生PD（不开HiCache/Pool服务）

### 5.1 每个窗口重新初始化、检查源码和NPU接口

以下均在新容器内；新开窗口先在宿主机执行`docker exec -it qwen3-cam-a2-v3 /bin/bash`。进入后228执行`source /ssd_2/yuanwenliu/sglang/experiments/qwen3_cam/env.sh 228`，238把参数改为238。应看到`QWEN_ENV_READY`。**每个P、D、store、Router窗口各自source，不能沿用其他窗口的环境。**源码修订后旧worker不会自动更新，先正常结束自己的旧P/D进程再重启。

```bash
python experiments/qwen3_cam/test_cpu.py
python -c 'import sglang; print(sglang.__file__); assert "/ssd_2/yuanwenliu/sglang/python/" in sglang.__file__'
python experiments/qwen3_cam/preflight.py --config experiments/qwen3_cam/configs/p_228.json --model-path "$MODEL_PATH" --mode off --require-pool --import-model
```

228 D的preflight用configs/d_228.json，238用configs/d_238.json。要求19项CPU测试OK、MODEL_IMPORT_OK、NPU_KVCACHEIO_API_OK、DEVICES_OK 8、POOL_API_OK、PREFLIGHT_PASS。输出中的SGLANG_HOST_IP必须分别为228/238的配置地址。检查只覆盖接口和文件，未加载权重或执行设备传输；池API通过不等于远端存取通过。

`Ignore import error ...qwen3_5...fused_sigmoid_mul`是扫描其他模型时跳过其不兼容模块的警告；本实验是Qwen3-30B-A3B，不是Qwen3.5。不要为了消除警告盲目升级kernel。若最终异常指向Qwen3自身或NPU_KVCACHEIO_API_FAILED，停止并回传完整traceback。

### 5.2 228先启动独立PD config store

旧版步骤仅export了ASCEND_MF_STORE_URL，却没有明确创建store。官方TransferEngine示例会调用`create_config_store`，而当前SGLang适配器只调用engine.initialize；因此增加一个明确的前台store，避免初始化时没有配置存储服务。[MemFabric示例](https://github.com/Ascend/memfabric_hybrid/blob/master/examples/transfer/test_transfer_engine.py)、[API](https://github.com/Ascend/memfabric_hybrid/blob/master/doc/pythonAPI.md)。这是**P端必须先排除的部署遗漏**，不代表已经证明是现有native初始化错误的唯一原因。

**228独立store终端**：进入实验容器并source env.sh 228后执行，保留该终端：

```bash
python -u experiments/qwen3_cam/start_pd_store.py
```

要求看到`PD_STORE_READY store=tcp://228的配置IP:7100`。此进程只创建配置存储，不加载模型、不分配模型/NPU缓存；保留SDK的TLS默认设置，不继承权重下载的跳过证书校验补丁。7100不是HTTP服务，不用curl访问。不与238上MemCache的5000/6000服务混用。若7100已被占用，脚本报错停止：核对是不是自己的旧store，不自动接管或杀掉任何进程。后续所有P/D重启仍需保留这个store终端。

store启动后，228 P窗口检查：

```bash
python experiments/qwen3_cam/preflight.py --config experiments/qwen3_cam/configs/p_228.json --model-path "$MODEL_PATH" --mode off --require-pool --import-model --check-pd-store
```

228 D/238 D各自换成对应config执行同一检查，均应输出`PD_STORE_TCP_OK`和`PREFLIGHT_PASS`。TCP成功只证明配置服务端口可达，不证明DEVICE_RDMA/HCCL正确；失败时先不要加载模型。

### 5.3 启动P4和两端D8，保留日志

三个**独立前台终端**分别启动；每个终端先docker exec和source。脚本固定D的回退备份为`cpu_tensor`，先避免原生PD验收阶段自动建立非目标Host备份池；TP8/EP8、Attention DP2×TP4、DEVICE_RDMA和P→D真实传输不变。D回退用CPU Tensor备份，不是取消KV迁移，也不是启用远端KV Pool。已经修复Tensor兼容性，不应只靠此参数掩盖P端后续HiCache的验证。

**228终端1：P**

```bash
export SGLANG_CAM_DETOUR_MODE=off
set -o pipefail
bash experiments/qwen3_cam/launch.sh p-228 --context-length 4096 --max-running-requests 4 --chunked-prefill-size 1024 2>&1 | tee "experiments/qwen3_cam/results/p-228-startup-$(date +%Y%m%d-%H%M%S).log"
```

**228终端2：D主节点**

```bash
export SGLANG_CAM_DETOUR_MODE=off
set -o pipefail
bash experiments/qwen3_cam/launch.sh d-228 --context-length 4096 --max-running-requests 4 2>&1 | tee "experiments/qwen3_cam/results/d-228-startup-$(date +%Y%m%d-%H%M%S).log"
```

**238终端1：D第二节点**

```bash
export SGLANG_CAM_DETOUR_MODE=off
set -o pipefail
bash experiments/qwen3_cam/launch.sh d-238 --context-length 4096 --max-running-requests 4 2>&1 | tee "experiments/qwen3_cam/results/d-238-startup-$(date +%Y%m%d-%H%M%S).log"
```

D两端都启动才完成建组。SGLang自己起模型worker，**不套16进程torchrun**。检查P/D日志中的`ASCEND_TRANSFER_INIT`地址/物理卡号和`ASCEND_TRANSFER_READY`，待权重加载/EP/PD初始化成功再启动Router；有失败先看14.1，不因看到HTTP错误而重新下载权重。228终端3（同样进入容器并source env.sh 228）启动Router：

```bash
python -m sglang_router.launch_router --pd-disaggregation --prefill "http://${HOST_228_IP}:30000" --decode "http://${HOST_228_IP}:30001" --host 0.0.0.0 --port 8000
```

228终端4发送请求：

```bash
curl --fail-with-body http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d '{"text":"Explain tensor parallelism briefly.","sampling_params":{"max_new_tokens":32,"temperature":0}}'
```

验收：合理文本、P→D迁移成功；日志/trace确认D EP8=228卡4–7+238卡0–3、Attention两组同机TP4。OOM先排查其他进程/上下文/临时内存，不能悄悄退化EP1或假传输。最终环境已将MemFabric升级到1.1.0，必须在这个环境通过PD，不沿用旧镜像结论。

结束本步：在自己的终端Ctrl+C，先Router、再P/D，确认子进程释放卡。PD store不占模型卡，继续后续实验可保留；结束整套实验时最后停止它。不要pkill python、stop原SGLang容器或停止别人的任务。

## 6 编译CAM并做单机8卡正确性测试

P/D/Pool全部停下。228/238分别在新容器中source env.sh后：

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
python experiments/qwen3_cam/preflight.py --config experiments/qwen3_cam/configs/p_228.json --model-path "$MODEL_PATH" --mode cam-detour --require-pool --import-model
```

238改configs/d_238.json；要求CAM_IMPORT_OK和PREFLIGHT_PASS。

## 7 验证真实KV Pool，再给P接入HiCache

### 7.1 独立远端put/get

模型全部停下，228卡0供probe独占。每个终端先source env.sh。

**238 Meta终端：**

```bash
python -m sglang.srt.mem_cache.storage.npu_memcache.start_meta_service --config_path experiments/qwen3_cam/runtime-config/meta.json
```

**238 Pool终端：**

```bash
torchrun --nproc_per_node=4 --master_port=29612 experiments/qwen3_cam/pool_sidecar.py --config experiments/qwen3_cam/configs/d_238.json --pool-config experiments/qwen3_cam/runtime-config/pool.json --mode off
```

要求238卡4、5、6、7各打印KV_POOL_READY和RELAY_READY。只有这四个服务有正HBM容量。

**228 probe终端：**

```bash
python experiments/qwen3_cam/pool_roundtrip.py --client-config experiments/qwen3_cam/runtime-config/client.json --device 0 --mib 16 --iterations 20
curl --fail-with-body "http://${HOST_238_IP}:8100/metrics"
```

要求20次checksum PASS，238卡4–7容量/占用及Pool日志/metrics证明远端读写。仅checksum不能证明placement。若零容量客户端或DEVICE_RDMA初始化失败，停下保存错误，不改客户端为正容量后仍声称Pool4。

### 7.2 P接HiCache，验证远端恢复

保留238 Meta和off sidecar。P的启动改为以下命令；D和Router仍用第5节：

```bash
export SGLANG_CAM_DETOUR_MODE=off
CLIENT_JSON=$(tr -d '\n' < experiments/qwen3_cam/runtime-config/client.json)
bash experiments/qwen3_cam/launch.sh p-228 --context-length 4096 --max-running-requests 4 --chunked-prefill-size 1024 --enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-storage-backend npu_memcache --hicache-mem-layout page_first_kv_split --hicache-storage-backend-extra-config "$CLIENT_JSON"
```

先短请求，再两次相同约2K历史前缀。第一次查存储写入；第二次查远端Pool read/bytes或hit。仅cached_tokens或本地Radix命中不足以证明Pool恢复。可用不同前缀压力逐出本地缓存；仅在off模式下也可尝试重启P再读，但须证明L3键/域仍有效。

当前路径为Pool→host staging→P的模型KV→D，不是D直接Pool恢复。若只看到本地命中，先解决远端命中证据，不继续把它当KV Pool恢复实验。

## 8 多路径与三模式对照

每轮都在同一镜像、相同权重/并行/HiCache/负载下运行。切模式时先Router→P/D→sidecar停止；Meta可保持。某个CAM owner退出后不能单独重启复用旧IPC引用，先结束完整本机CAM域，再重建。

**238 sidecar终端**先启动：

```bash
torchrun --nproc_per_node=4 --master_port=29612 experiments/qwen3_cam/pool_sidecar.py --config experiments/qwen3_cam/configs/d_238.json --pool-config experiments/qwen3_cam/runtime-config/pool.json --mode cam-detour
```

它可能等待238的D owner，不要求先独立完成CAM就绪。然后三个模型终端各source对应228/238，并设置：

```bash
export SGLANG_CAM_DETOUR_MODE=cam-detour
export SGLANG_CAM_TRACE_MOE=1
```

**228的P终端：**

```bash
export SGLANG_CAM_DETOUR_TRACE='experiments/qwen3_cam/results/cam-p-228-{rank}.jsonl'
CLIENT_JSON=$(tr -d '\n' < experiments/qwen3_cam/runtime-config/client.json)
bash experiments/qwen3_cam/launch.sh p-228 --context-length 4096 --max-running-requests 4 --chunked-prefill-size 1024 --enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-storage-backend npu_memcache --hicache-mem-layout page_first_kv_split --hicache-storage-backend-extra-config "$CLIENT_JSON"
```

**228的D终端：**

```bash
export SGLANG_CAM_DETOUR_TRACE='experiments/qwen3_cam/results/cam-d-228-{rank}.jsonl'
bash experiments/qwen3_cam/launch.sh d-228 --context-length 4096 --max-running-requests 4
```

**238的D终端：**

```bash
export SGLANG_CAM_DETOUR_TRACE='experiments/qwen3_cam/results/cam-d-238-{rank}.jsonl'
bash experiments/qwen3_cam/launch.sh d-238 --context-length 4096 --max-running-requests 4
```

不要复制第5/7节的MODE=off。228的P4+D4、238的D4+Pool4全部owner启动后再启动Router，发送短请求，检查CAM_READY/CAM_CALL。JSONL是host提交日志，不是device执行时间。

off、hccl-a2a、cam-detour分别完整重启：P/D的MODE与sidecar的mode一致，真实Pool均在线。每轮先固定输入、temperature=0、seed检查token/logprob/代表性准确率，再压测；明显输出错误不进入性能结果。

## 9 综合serving负载

先把P/D上下文从4096提高到16384，并发从4逐步提高：可先P=8、D=16，再扫P=16、D=64；其他设置不变。32GB HBM同时要容纳权重、静态KV、临时张量、CAM512MiB，Pool卡另加8GB池。

背景请求持续decode，恢复burst使用共享长历史前缀。固定到达计划，先小负载，再扫输入4K/8K、并发8/16/32/64和缓存冷热。16K前缀加输出会超16384上下文，需扩大上下文或用15360前缀。理论完整BF16 KV每token96KiB，8K约768MiB；实际以传输字节为准。

在228的新容器另开客户端终端、source env.sh 228后：

```bash
python experiments/qwen3_cam/serving_workload.py --model-path "$MODEL_PATH" --base-url "http://${HOST_228_IP}:8000" --duration 30 --background-rps 0.5 --burst-every 10 --burst-size 8 --prefix-tokens 8192 --background-output-tokens 512 --restore-output-tokens 128 --max-concurrency 64 --run-id cam-r1 --output experiments/qwen3_cam/results/serving-cam-r1.jsonl
```

每种模式同一参数，到达计划不变；run-id/输出文件改为本轮唯一名称，不能混写。记录TTFT、TPOT/SSE间隔、缓存信息、失败和client_queue_ms。客户端限流会改变实际负载；SSE批量返回token时，脚本间隔不是device逐token TPOT。热前缀仍须证明远端Pool读取。

## 10 找共享资源与真实重叠

| 组合 | 可能共享 | 必须证明 |
| --- | --- | --- |
| 多路径+MoE | D卡HCCS边、HBM/DMA | 原生MoE实际使用该资源且device区间重叠 |
| MoE+跨机KV | NIC/跨机交换机资源 | 实际路由、端口计数器与时间重叠 |
| 多路径+Pool读取 | 中继Pool卡HBM/DMA或真实同机搬运边 | Pool真实路径，不虚构KV绕行 |
| 三类共同影响 | 上述资源集合 | 联合时间线、独立/并发对照 |

例如绕行238卡0→238卡4→238卡1访问238卡4 HBM；若KV从238卡4自己的NIC出机，它不经过238卡4→238卡1，只可能共享238卡4的HBM/DMA。三类不一定同时堵同一条物理边。关闭TBO时，同一batch的Attention/MoE通常有依赖，不能预设两者并发。

在P/D分别启动profile，不依赖Router广播：

```bash
curl -X POST "http://${HOST_228_IP}:30000/start_profile" -H 'Content-Type: application/json' -d '{"output_dir":"/ssd_2/yuanwenliu/sglang/experiments/qwen3_cam/profiles/p","start_step":3,"num_steps":30,"activities":["CPU","GPU"]}'
curl -X POST "http://${HOST_228_IP}:30001/start_profile" -H 'Content-Type: application/json' -d '{"output_dir":"/ssd_2/yuanwenliu/sglang/experiments/qwen3_cam/profiles/d","start_step":3,"num_steps":30,"activities":["CPU","GPU"]}'
```

这里GPU是通用设备活动名称，NPU后端使用对应profiler。发送workload后检查所有worker的device task开始/结束、CAM_DETOUR、NATIVE_MOE、HCCL与KV事件。CPU范围不能代替device重叠证据；跨机时钟需校准。缺RDMA事件时补MemFabric完成指标或当前CANN支持的系统采集。

记录实际工具支持的收发字节、busy/queue/拥塞或重传、带宽、HBM容量/带宽。HBM容量不等于带宽；共享资源+重叠仍只是可能冲突，必须与独立/串行对照比较。

## 11 可选：16卡受控回放

只有真实serving未观察到需要的重叠时再做。先停全部模型/sidecar，脚本占用16卡，不需要权重。MoE是两次HCCL EP8 All-to-All代理，KV是HCCL P2P代理，不是DeepEP实际token路由或MemFabric Pool，因此不能当成真实服务结果。

228/238各在新容器source正确env.sh，然后同时启动：

```bash
# 228
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr="$HOST_228_IP" --master_port=29620 experiments/qwen3_cam/replay_conflict.py --mode cam-detour --warmup 20 --iterations 100 --output experiments/qwen3_cam/results/replay-cam-228 --kv-target 228
# 238
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr="$HOST_228_IP" --master_port=29620 experiments/qwen3_cam/replay_conflict.py --mode cam-detour --warmup 20 --iterations 100 --output experiments/qwen3_cam/results/replay-cam-238 --kv-target 228
```

自动测单类、双类、三类七种组合；仍需profile证明device重叠。可扫tp-mib=1/8/32、moe-mib=1/8/32/128、kv-mib=16/64/256，kv-target 228/238对应跨机/同机目标；再测hccl-a2a，同参数至少5轮，结果目录每轮独立。

把238结果目录复制到228同名路径（先核对不要覆盖旧结果），228汇总：

```bash
python experiments/qwen3_cam/analyze_replay.py experiments/qwen3_cam/results/replay-cam-228 experiments/qwen3_cam/results/replay-cam-238 --output experiments/qwen3_cam/results/replay-summary.json
```

脚本按rank/payload/模式算p50/p95/p99和slowdown，不平均掉最慢rank；缺同条件standalone时拒绝计算。置信区间需要从跨轮原始样本另算，不能把少量样本p99当可靠尾延迟。

## 12 判断是否冲突、影响多少

至少汇报通信p95 slowdown、TTFT/TPOT p95、tokens/s、KV完成时间、SLO miss rate和共享资源指标。

通信slowdown=同实现并发p95÷独立p95。先预定报告分档，例如<5%且在噪声内记未见显著退化、5–10%轻度、10–30%明显、>30%严重；这些是实验阈值，不是行业标准。至少5轮paired比较，置信区间含1则不做确定性退化结论。

同资源并发退化，错峰/串行后恢复，并有queue/busy回落，才支持链路冲突解释。只有HBM/DMA指标变化则归类内存/搬运竞争；没有device重叠则是依赖串行。不能把实现额外开销、计算竞争、网络拥塞混为一谈。

每条结论附版本/拓扑/路径、device时间线、独立/并发/干预数值和重复轮次。允许结果为“链路冲突”“HBM/DMA竞争但无同边冲突”“当前负载未发现冲突”，不填写假想百分比。

## 13 代码同步边界：实验机器只拉取

本文档仅指导228/238获取和运行已发布的实验源码，不包含源码发布命令。发布在开发侧单独处理，相关指令仅在对话中提供；不要在实验机器执行发布操作。

实验机器按第4.1节从GitHub/GitCode克隆或ff-only更新，核对两机版本一致。有本地改动时停止并保留，不强行覆盖。代码托管站只负责源码流转，不承担任何服务运行期通信。

## 14 失败定位与安全停止

| 问题 | 先查 |
| --- | --- |
| ascend runtime缺失、挂载路径缺失 | docker info与宿主路径；请管理员处理，不改Docker全局配置、不切旧opensora镜像 |
| apt超时，随后找不到zip/rsync等包 | 先修复源索引访问；按4.3传构建代理与host网络，不通过删除包名绕过问题 |
| pip或wheel下载失败 | 构建代理是否传入；PyPI/files.pythonhosted.org访问策略、认证及CA；不关闭TLS校验 |
| 源码导入失败 | 是否用/ssd_2/yuanwenliu/sglang/python、CANN环境、新源码要求的kernel API |
| Ignore import error ...qwen3_5...fused_sigmoid_mul | 模型扫描警告；检查真正退出的traceback是否来自目标Qwen3，不直接升级kernel |
| Boolean value of Tensor ... ambiguous | NPU Tensor与Host缓存初始化的兼容性问题，更新本分支修复；不是OOM |
| Ascend Transfer Engine initialization failed | 第5.2的PD store、明确的session IP，以及新版日志ret/native_error；不盲目退回SDMA |
| EP/PD/Pool网络失败 | 当前后端DEVICE_RDMA、接口/端口与各NPU网络健康 |
| CAM等待 | 全部8 owner、10067、comm_id7、物理rank映射/IPC权限 |
| 有日志无CAM_CALL | MODE/配置及融合是否绕过接入点 |
| Pool只有本地hit | 存储写入、远端read/placement、键与域；不把Radix hit当Pool hit |

网络问题时才在宿主机定向查（工具不在PATH时用driver/tools中的实际路径）：

```bash
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i "$i" -ip -g; done
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i "$i" -net_health -g; done
```

停止仅限自己的实验终端：Router→P/D→sidecar→Meta→PD store（最后）。切回off前确认旧CAM域退出；不pkill python、不docker prune、不清共享Pool。probe只删除自己的UUID key。未授权时不删除或替换旧容器。

### 14.1 2026-10-13 P/D启动失败：结论、修订和复测

| 日志/现象 | 已确认的原因或边界 | 本轮处理 |
| --- | --- | --- |
| 228/238 D：pool_host/mha.py 的Tensor布尔值错误 | NPUMHATokenToKVPool默认用连续的全层Tensor；通用代码误用`buffers or ()` | 对None显式判断，并逐层遍历Tensor或列表，不复制缓存、不改变设备传输算子；覆盖空值、列表、元组、Tensor形式回归测试 |
| 未显式开启HiCache却进入Host池 | 当前Decode回退策略会自动选host_pool，引发本地Host池构建；不是远端Pool4已经运行 | 原生PD脚本固定cpu_tensor回退备份；P端远端HiCache仍须第7节实测 |
| P：TransferEngine.initialize返回非零 | 日志仅给通用异常，无法确认具体store/地址/RDMA/驱动原因；旧流程未明确创建PD store | 增加前台PD store、两机TCP预检查、明确SGLANG_HOST_IP、原生ret/native_error日志；尚未证明实机初始化成功 |
| HTTPConnectionPool | 这份附件未给host/port/URL，不能断定公网下载；也可能是P退出后的服务连接失败或残留代理 | 推理source时清理下载代理；保留完整HTTP行，按目标区分公网与228/238服务 |
| fused_sigmoid_mul缺失且前缀Ignore import error | qwen3_5在模型扫描时被跳过；不是Qwen3-30B本身的致命异常 | 不升级kernel，不把警告当根因；预检查增加目标NPU KV IO API检查 |
| 怀疑OOM | 附件这几段没有OutOfMemory异常，也不能证明不存在更早native内存错误 | 不凭这些traceback改变P/D配比、量化或降EP；保留第一条native错误及npu-smi快照 |

**已有NPU适配的位置：**`hardware_backend/npu/memory_pool_npu.py`负责NPU连续KV缓存；`mem_cache/pool_host/mha.py`的NPU分支调用`sgl_kernel_npu.kvcacheio.transfer_kv_dim_exchange`；`disaggregation/ascend/conn.py`和`transfer_engine.py`接入MemFabric的PD传输。不是没有NPU实现，而是新通用Host缓存逻辑与NPU表示之间出现回归，加上部署/诊断步骤不完整。这次保留NPU路径，不用CUDA算子替代，不承诺整个新源码与旧kernel完全兼容。

**已有容器的恢复步骤：**

1. 确认代码托管站已有本次SGLang修订；本轮只更新SGLang，UMDK保持现有实验版本。本文档不包含开发侧发布操作。
2. 正常结束自己的旧P/D和Router，避免旧worker继续使用旧源码。不要停他人的任务。
3. 228、238宿主机各执行下面更新，确认HEAD相同。GitHub访问仍按获准方式设置宿主代理；有本地改动就停下，不reset、不清理。容器挂载该源码目录，所以无需重建Docker、重装kernel或重新下载权重。

```bash
cd /ssd_2/yuanwenliu/sglang
git status --short --branch
git remote get-url origin
git switch feat/qwen3-cam-detour-conflict
git pull --ff-only origin feat/qwen3-cam-detour-conflict
git rev-parse HEAD
```

4. 每个新实验终端从宿主机docker exec进入，再source对应机器的env.sh；重新运行5.1预检查。228、238都应通过，不能仅看CPU测试OK。
5. 按5.2先启动228的PD store并保持前台，三处preflight加`--check-pd-store`。7100监听由自己的store提供；如果现有服务占用，先核对归属，不能kill或偷偷换端口。
6. 按5.3重新启动P、两端D，使用带tee的命令保留日志；待P/D都就绪再启动Router。成功后再进入CAM/Pool，不跳过原生PD。

**P仍失败时的最小取证：**每个P/D容器终端source后将MF日志级别设为0再按5.3重启；不要反复并行启动新worker。该变量是本分支的适配开关，调用SDK的`set_log_level(0)`，不是假定存在某个官方环境变量：

```bash
export SGLANG_ASCEND_MF_LOG_LEVEL=0
```

保存日志中第一次`ASCEND_TRANSFER_INIT`到第一条失败（含`ret`和`native_error`）前后的原生日志，不只贴最后Scheduler traceback。若报错在DEVICE_RDMA的前置all_gather，取证HCCL建组错误；若已进入engine.initialize，依据native错误排查store连接、session地址、NPU RDMA初始化和库版本。get_last_err_msg只是辅助信息，native完整日志仍是依据。两台宿主机补充`npu-smi info`，以及第14节hccn_tool的IP/健康查询；无需全量listen清单。不要把PD store的TCP成功当作RoCE网络验证。

**本地权重与HTTP排查：**日志应显示本地model_path与offline配置；env.sh禁止推理使用下载代理。`huggingface.co/modelscope.cn`属于公网模型查询，需排查哪个组件触发；`228:8998`属于PD bootstrap，`228:30000/30001/8000`属于推理/Router接口。首次HTTP请求发生在哪个进程、到哪里、是否在另一进程失败后出现，决定它是根因还是后续症状；远端权重已准备好，不因此重新下载。PD store的7100使用TCP协议，不归入HTTP接口。

**验证边界：**19项CPU/模拟回归测试和命令语法检查通过，仅验证Tensor表示处理、环境设置、参数构造和错误信息传递；没有在228/238实际执行NPU Host缓存搬运、启动RDMA引擎或验证推理质量。D实际启动、P/D迁移以及第7节远端Pool都仍需实机验收。

**Tensor修复的专项实机复测：**cpu_tensor基线不会覆盖原先触发错误的D host_pool分支。原生PD通过后，可正常结束自己的Router和两端D，再在228/238的D启动命令末尾分别追加`--disaggregation-decode-retraction-backup host_pool`，重启两端D并检查Host池初始化；需要支持回退机制时再验证真实备份/恢复。这项复测应单独记录，不与cpu_tensor基线混用。第7节P的HiCache还必须通过真实远端put/get和恢复，不能用一次启动成功替代搬运正确性测试。初次只排查启动时，先保持cpu_tensor基线。

## 15 依据与验证边界

- 你提供的A2/OpenSora docker run：ascend、privileged、设备、DCMI、驱动与日志挂载；不采用OpenSora应用挂载。
- [Ascend Docker Runtime](https://www.hiascend.com/document/detail/en/mindcluster/2610/clustersched/schedulingug/docs/en/scheduling/usage/containerization/02_usage_on_the_docker_client.md)、[Docker运行参数](https://docs.docker.com/reference/cli/docker/container/run/)。
- [SGLang Qwen3](https://github.com/sgl-project/sglang/blob/f620d73/python/sglang/srt/models/qwen3_moe.py)、[PD](https://github.com/sgl-project/sglang/blob/f620d73/docs/docs/advanced_features/pd_disaggregation.mdx)、[NPU MemCache](https://github.com/sgl-project/sglang/blob/f620d73/python/sglang/srt/mem_cache/storage/npu_memcache/README.md)及本仓库NPU pyproject。
- [MemCache API版本说明](https://raw.githubusercontent.com/Ascend/memcache/master/doc/install_whl.md)、[1.1.0包元数据](https://pypi.org/pypi/memcache-hybrid/1.1.0/json)、[MemFabric1.1.0元数据](https://pypi.org/pypi/memfabric-hybrid/1.1.0/json)；wheel和SHA256见docker/requirements-pool.txt。

本轮通过19项CPU/模拟回归测试和Python/shell语法检查；覆盖个人路径、228/238启动角色、代理清理与明确的session IP、ascend/八卡/IPC参数、NPU Tensor/list缓冲区遍历、PD store封装及原生初始化错误保留。这些不等于目标实机验收；测试中store和TransferEngine为模拟对象。镜像构建、权重准备及部分预检查由用户在A2执行并反馈；助手没有直接连接228/238、分配NPU或完成模型运行。
