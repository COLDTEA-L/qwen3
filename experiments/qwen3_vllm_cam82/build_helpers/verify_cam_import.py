"""Check CAM module/API loading and schemas without initializing an NPU/comm."""
import argparse
import ctypes
import importlib
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-api", help="Override the CAM operator API shared library")
    args = parser.parse_args()
    modules = {}
    for name in ("torch", "torch_npu", "umdk_cam_op_lib"):
        print("CAM_IMPORT_BEGIN", name, flush=True)
        modules[name] = importlib.import_module(name)
        print("CAM_IMPORT_OK", name, getattr(modules[name], "__file__", None), flush=True)
    torch = modules["torch"]
    op = modules["umdk_cam_op_lib"]
    for name in ("all2_all_detour", "all_gather_detour", "reduce_scatter_detour", "cam_get_comm", "cam_free_comm"):
        if not hasattr(op, name):
            raise RuntimeError("Missing CAM extension API: " + name)
        schema = torch._C._dispatch_find_schema_or_throw("umdk_cam_op_lib::" + name, "").schema()
        print("CAM_SCHEMA_OK", schema, flush=True)
    library = Path(args.op_api) if args.op_api else Path(os.environ["ASCEND_OPP_PATH"]) / "vendors/CAM/op_api/lib/libcust_opapi.so"
    if not library.is_file():
        raise RuntimeError("CAM operator API library missing: " + str(library))
    print("CAM_OP_API_LOAD_BEGIN", library, flush=True)
    native = ctypes.CDLL(str(library))
    for name in ("aclnnAll2AllDetour", "aclnnAll2AllDetourGetWorkspaceSize"):
        if not hasattr(native, name):
            raise RuntimeError("Missing installed operator symbol: " + name)
    print("CAM_OP_API_SYMBOLS_OK", flush=True)
    print("CAM_IMPORT_AND_API_PASS (no NPU execution)", flush=True)


if __name__ == "__main__":
    main()
