# 开发服务器 CAM 算子验证记录

日期：2026-10-16（Asia/Shanghai）。按用户最终要求，只编译验证算子，不清理数据、不构建整套8.2镜像。

## 1 环境与边界

- 服务器：8.146.225.22，arm64，无可用于本实验的NPU。
- 实际容器：`cam_lyw_dev_85`，镜像`cam-dev:a3_8.5_open`；已启动并保留。
- CANN：`/usr/local/Ascend/cann-8.5.0`。
- Python：`/opt/conda/envs/cam_py311_pt28/bin/python`，3.11.15。
- Torch：2.8.0+cpu；torch_npu：2.8.0.post2；CXX11 ABI=True。
- UMDK基线：9c0bbdadfcfe882426cabdce5cdf2d5d2fce92ef，另应用本实验仓库的5项补丁及环境加载辅助脚本。
- 独立工作区：`/home/liuyuanwen/cam85-verify.skIr2x`，源码在其`umdk`目录。
- 没有切换或覆盖`/home/liuyuanwen/umdk`原分支，原有未跟踪build目录也保留。
- 没有覆盖容器原有`$ASCEND_OPP_PATH/vendors/CAM`；新包安装在工作区的`opp-install`，wheel安装在工作区的`python`目录。
- 容器内为定位问题新增GDB及必要依赖；未升级已有包，未清理任何已有镜像、容器或用户数据。

这组环境不同于228/238的CANN8.2+Torch2.7，产物不能跨版本直接替换。此记录证明编译、安装、导入和动态库符号验证通过，不证明NPU执行、数值正确性或多卡通信性能。

## 2 问题定位与修正

1. 用ctypes单独加载新编译的libcam，即复现bad_alloc，无需加载模型或建立通信域。
2. GDB定位全局IPv4正则构造，出现约247TB的异常分配请求。保留日志，不把它解释成正常的物理内存/显存需求。
3. 改为无分配IPv4校验后，同一容器、同一Torch/ABI、同一加载顺序成功输出`FIXED_LIBCAM_LOAD_OK`。
4. 重新链接wheel，导入成功，AllToAll/AllGather/ReduceScatter及通信域管理接口schema可查询。
5. 主动加载算子API库时发现inner API实现未进入链接，修正源文件glob后重新编译、打包、隔离安装。
6. 最终检查成功输出`CAM_OP_API_SYMBOLS_OK`和`CAM_IMPORT_AND_API_PASS (no NPU execution)`。

## 3 产物与日志

工作区前缀统一为`/home/liuyuanwen/cam85-verify.skIr2x`。

| 内容 | 工作区下的位置 |
| --- | --- |
| .run | `umdk/output/cam/comm_operator/run/CAM_ascend910b4_all_to_all_detour_ubuntu_aarch64.run` |
| wheel | `umdk/output/cam/qwen3-all2all/wheel/umdk_cam_op_lib-208.1.0b1-cp311-cp311-linux_aarch64.whl` |
| 原生通信库 | `umdk/output/cam/qwen3-all2all/comm-build/libcam.so`、同目录`libcam_static.a` |
| 隔离安装的API库 | `opp-install/vendors/CAM/op_api/lib/libcust_opapi.so` |
| 隔离安装的扩展 | `python/umdk_cam_op_lib.cpython-311-aarch64-linux-gnu.so` |
| 初次完整编译日志 | `logs/build85.log` |
| 崩溃C++调用栈 | `logs/libcam-badalloc-gdb.log` |
| 通信库修正后验证 | `logs/libcam-fixed-build-import.log` |
| inner API修正后编译安装 | `logs/inner-api-rebuild.log` |
| 最终导入及API验证 | `logs/wheel-import-check-final.log` |
| 最终CPU回归（33项，1项可选源码检查跳过） | `logs/cpu-regression-final.log` |

产物SHA256（仅适用于本次8.5验证构建）：

```text
a2cee914ee1368254dc2fdbd7660aefa324d1081aa46bd860043b25b509fcfed  CAM_ascend910b4_all_to_all_detour_ubuntu_aarch64.run
29f5906005b8506a981c368c8b1c931499f68e33150b620e20cc68deed922450  umdk_cam_op_lib-208.1.0b1-cp311-cp311-linux_aarch64.whl
```

## 4 在现有容器中复验

开发服务器宿主机进入容器：

```bash
docker exec -it cam_lyw_dev_85 bash
```

以下在容器中执行，只影响当前shell，不改原有Python环境或CAM安装：

```bash
export PATH=/opt/conda/envs/cam_py311_pt28/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PYTHONPATH=/home/liuyuanwen/cam85-verify.skIr2x/python
export ASCEND_CUSTOM_OPP_PATH=/home/liuyuanwen/cam85-verify.skIr2x/opp-install/vendors/CAM
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/aarch64-linux/lib64:/home/liuyuanwen/cam85-verify.skIr2x/opp-install/vendors/CAM/op_api/lib:${SHMEM_HOME_PATH}/shmem/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:$ASCEND_HOME_PATH/aarch64-linux/devlib"
python -u /home/liuyuanwen/cam85-verify.skIr2x/verify_cam_import.py --op-api /home/liuyuanwen/cam85-verify.skIr2x/opp-install/vendors/CAM/op_api/lib/libcust_opapi.so
```

应输出`CAM_IMPORT_AND_API_PASS (no NPU execution)`。检查脚本的版本化副本是本目录`build_helpers/verify_cam_import.py`。

## 5 对实验机的意义

主实验Dockerfile已经纳入修正和新增原生验证，但没有在开发机生成8.2完整镜像。228/238应从修正后的源码在各自固定8.2栈中构建，再执行原README的NPU/TP/PD关口；不能因为本记录成功而跳过这些步骤。
