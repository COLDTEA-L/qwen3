"""Qualification guards around the pinned, unmodified Ascend Mooncake worker.

The paired aligned allocator provides the exact views used by native attention
binding and native registration/data movement. These guards check their actual
addresses and reject failed transfers rather than assuming allocator alignment.
"""
import functools
import math
import threading

ALIGNMENT = 2 * 1024 * 1024
BLOCK_SIZE = 64
NUM_BLOCKS = 2048
BLOCK_BYTES = 64 * 1 * 128 * 2  # Qwen3-30B-A3B, BF16, attention TP4.
NUM_LAYERS = 48


def validate_config(config):
    cache = config.cache_config
    if (cache.block_size != BLOCK_SIZE
            or cache.num_gpu_blocks_override != NUM_BLOCKS):
        raise RuntimeError("CAM82 native KV requires --block-size 64 and "
                           "--num-gpu-blocks-override 2048 on P and both D engines")


def validate_regions(kv_caches):
    """Check the exact K/V views that attention and the connector both use."""
    if len(kv_caches) != NUM_LAYERS:
        raise RuntimeError(f"CAM82 native KV expects {NUM_LAYERS} attention layers")
    regions = []
    for layer, cache in kv_caches.items():
        if len(cache) != 2:
            raise RuntimeError(f"CAM82 native KV requires a K/V pair: {layer}")
        for name, tensor in zip(("K", "V"), cache):
            shape = tuple(tensor.shape)
            if (shape != (NUM_BLOCKS, BLOCK_SIZE, 1, 128)
                    or str(tensor.dtype) not in ("torch.bfloat16", "bfloat16")
                    or tensor.element_size() != 2 or not tensor.is_contiguous()):
                raise RuntimeError(f"CAM82 native KV unexpected layout: {layer}.{name} shape={shape}")
            ptr = tensor.data_ptr()
            size = tensor.numel() * tensor.element_size()
            if size != math.prod(shape) * 2 or ptr <= 0 or ptr % ALIGNMENT:
                raise RuntimeError(f"CAM82_KV_ALIGNMENT_FAILED {layer}.{name} "
                                   f"ptr=0x{ptr:x} bytes={size}; require a 2MiB-aligned "
                                   "registered base; do not round registration pointers")
            regions.append((ptr, ptr + size))
    ordered = sorted(regions)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise RuntimeError("CAM82 native KV regions overlap")
    return len(regions)


class NativeKVGuard:
    def __init__(self, worker):
        self.worker = worker
        self._lock = threading.Lock()
        self._failure = None

    def fail(self, meta, error):
        with self._lock:
            if self._failure is None:
                self._failure = f"request={meta.get('request_id', '?')} {type(error).__name__}: {error}"
                print(f"CAM82_KV_TRANSFER_FAILED {self._failure}", flush=True)

    def check(self):
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("CAM82_KV_FAIL_STOP: refuse decode after KV transfer failure; " + failure)

    def install_receiver(self):
        receiver = self.worker.kv_recv_thread
        if receiver is None:
            return  # P keeps its original native cleanup/acknowledgement logic.
        if getattr(receiver, "_cam82_kv_guard", None) is self:
            return
        native_transfer = receiver._transfer_kv_cache

        @functools.wraps(native_transfer)
        def transfer(meta):
            try:
                self.check()
                local = meta["local_block_ids"]
                remote = meta["remote_block_ids"]
                if local and (len(local) != len(remote)
                              or any(not 0 <= i < NUM_BLOCKS for i in (*local, *remote))):
                    raise RuntimeError("CAM82 native KV invalid block count/index")
                return native_transfer(meta)
            except Exception as error:
                self.fail(meta, error)
                raise

        receiver._transfer_kv_cache = transfer
        receiver._cam82_kv_guard = self


def install(connector):
    """Scope wrappers to this experiment's connector instance, not global classes."""
    if (type(connector).__module__ != "vllm_ascend.distributed.mooncake_connector"
            or type(connector).__name__ != "MooncakeConnector"):
        raise RuntimeError("CAM82 native KV guard requires the pinned Ascend MooncakeConnector")
    worker = connector.connector_worker
    if worker is None:
        raise RuntimeError("CAM82 native KV guard must be installed on a WORKER connector")
    if hasattr(worker, "_cam82_native_kv_guard"):
        return worker._cam82_native_kv_guard
    validate_config(worker.vllm_config)
    guard = NativeKVGuard(worker)
    native_register = worker.register_kv_caches
    native_finished = worker.get_finished

    @functools.wraps(native_register)
    def register(kv_caches):
        count = validate_regions(kv_caches)  # Before any native registration/thread startup.
        result = native_register(kv_caches)
        guard.install_receiver()
        print(f"CAM82_KV_REGIONS_OK engine={worker.engine_id} regions={count} "
              f"blocks={NUM_BLOCKS} block_tokens={BLOCK_SIZE} "
              "bytes_per_region=33554432 alignment=2097152 native_transport=True", flush=True)
        return result

    @functools.wraps(native_finished)
    def finished(*args, **kwargs):
        # Native _handle_request marks done even on failure. Never let that
        # completion reach the scheduler. Check after collection as well, since
        # the receiving thread can fail concurrently with get_finished().
        guard.check()
        result = native_finished(*args, **kwargs)
        guard.check()
        return result

    worker.register_kv_caches = register
    worker.get_finished = finished
    worker._cam82_native_kv_guard = guard
    return guard
