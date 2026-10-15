# Official v0.10.2rc1 arm64 child manifest, verified through Quay on 2026-10-14.
ARG BASE_IMAGE=quay.io/ascend/vllm-ascend@sha256:610df8ee1ba35261cd324e2ecb161710f34a1f0c32875bc454872a8a64291f12
FROM ${BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY adapter /opt/cam82-adapter
RUN python -m pip install --no-deps --no-build-isolation --no-index /opt/cam82-adapter && \
    python -c 'import platform; from qwen_cam82.policy import check_installation; assert platform.machine()=="aarch64"; assert platform.python_version_tuple()[:2]==("3","11"); print("BASE_SOURCE_CHECK",check_installation())'
# Docker's predefined HTTP_PROXY/HTTPS_PROXY build arguments are used only for
# preparation; never put authenticated proxy URLs in ENV or source files.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates pkg-config libssl-dev libnuma-dev \
    libibverbs-dev librdmacm-dev libgoogle-glog-dev libgflags-dev libjsoncpp-dev \
    libyaml-cpp-dev libasio-dev libcurl4-openssl-dev mpich libmpich-dev iproute2 unzip && apt-get clean
COPY vendor/yalantinglibs /opt/yalantinglibs
RUN cmake -S /opt/yalantinglibs -B /opt/ylt-build -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF && \
    cmake --build /opt/ylt-build -j4 && cmake --install /opt/ylt-build
COPY vendor/Mooncake /opt/Mooncake
RUN test "$(git -C /opt/Mooncake rev-parse HEAD)" = a00f75742469d6cc408bc5393b73c501e92dc74a && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    cmake -S /opt/Mooncake -B /opt/mooncake-build -DUSE_ASCEND=ON -DUSE_ASCEND_DIRECT=OFF \
      -DWITH_STORE=OFF -DUSE_ETCD=OFF -DUSE_CUDA=OFF -DBUILD_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF && \
    cmake --build /opt/mooncake-build -j4 && cmake --install /opt/mooncake-build && \
    find /opt/mooncake-build -name libascend_transport_mem.so -exec install -m 755 {} /usr/local/lib/ \; && ldconfig
COPY vendor/umdk /opt/umdk-cam82
# Keep vendor pinned; patch only the image copy's build helpers.
COPY patches/umdk-all2all-pregen-prefix.patch /opt/cam82/umdk-all2all-pregen-prefix.patch
COPY patches/umdk-cann-env-guard.patch /opt/cam82/umdk-cann-env-guard.patch
COPY patches/umdk-platform-include.patch /opt/cam82/umdk-platform-include.patch
COPY build_helpers/source_cann_env_compat.sh /opt/umdk-cam82/build/cam/comm_operator/source_cann_env_compat.sh
RUN test "$(git -C /opt/umdk-cam82 rev-parse HEAD)" = 9c0bbdadfcfe882426cabdce5cdf2d5d2fce92ef && \
    test "$(sha256sum /opt/umdk-cam82/build/cam/comm_operator/compile_single_op.sh | cut -d ' ' -f1)" = ce9d1a7821b2956c2b56f6d4f4e8518abf99238676494c6d323dc9f2fca9c889 && \
    test -s /opt/umdk-cam82/src/cam/comm_operator/ascend_kernels/pregen/build_out/autogen/aclnn_all2_all_detour.cpp && \
    test -s /opt/umdk-cam82/src/cam/comm_operator/ascend_kernels/pregen/build_out/autogen/aclnn_all2_all_detour.h && \
    git -C /opt/umdk-cam82 apply --check /opt/cam82/umdk-all2all-pregen-prefix.patch && \
    git -C /opt/umdk-cam82 apply /opt/cam82/umdk-all2all-pregen-prefix.patch && \
    echo CAM82_PREGEN_PREFIX_PATCH_OK && \
    git -C /opt/umdk-cam82 apply --check /opt/cam82/umdk-cann-env-guard.patch && \
    git -C /opt/umdk-cam82 apply /opt/cam82/umdk-cann-env-guard.patch && \
    echo CAM82_CANN_ENV_PATCH_OK && \
    git -C /opt/umdk-cam82 apply --check /opt/cam82/umdk-platform-include.patch && \
    git -C /opt/umdk-cam82 apply /opt/cam82/umdk-platform-include.patch && \
    echo CAM82_PLATFORM_INCLUDE_PATCH_OK && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/aarch64-linux/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:${ASCEND_HOME_PATH}/aarch64-linux/devlib" && \
    python -c 'import torch, torch_npu; print("CAM82_BUILD_TORCH_IMPORT_OK", torch.__version__, torch_npu.__version__)' && \
    BUILD_JOBS=4 bash /opt/umdk-cam82/build/cam/comm_operator/build_qwen3_all2all_detour.sh ascend910b4 && \
    wheels=(/opt/umdk-cam82/output/cam/qwen3-all2all/wheel/*.whl) && test "${#wheels[@]}" = 1 && \
    python -m pip install --no-deps --no-index "${wheels[0]}" && \
    install -m 755 /opt/umdk-cam82/output/cam/qwen3-all2all/comm-build/libcam.so /usr/local/lib/libcam.so && ldconfig
COPY check.py /opt/cam82/check.py
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/aarch64-linux/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:${ASCEND_HOME_PATH}/aarch64-linux/devlib" && \
    python /opt/cam82/check.py && \
    python -c 'import torch, torch_npu, umdk_cam_op_lib; from mooncake.engine import TransferEngine; print("CAM82_NATIVE_IMPORT_OK", torch.__version__, torch_npu.__version__)' && \
    python -m pip freeze > /opt/cam82/packages.txt
WORKDIR /ssd_2/yuanwenliu/sglang
CMD ["/bin/bash"]
