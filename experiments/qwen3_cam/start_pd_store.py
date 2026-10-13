"""Foreground MemFabric PD config store. No model or NPU allocation.

Run once on 228, before P/D. This is not the MemCache pool's config store.
It preserves the installed SDK's TLS defaults; it does not bypass certificates.
"""

import argparse
import os
import signal
import socket
import threading
from urllib.parse import urlparse


def store_address(url):
    parsed = urlparse(url)
    if (parsed.scheme != "tcp" or not parsed.hostname or not parsed.port
            or parsed.username or parsed.password or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment):
        raise ValueError("Require tcp://local-ip:port without credentials/path/query")
    return parsed.hostname, parsed.port


def start_store(url, mf):
    host, port = store_address(url)
    # Do not reuse/replace a listener of unknown ownership.
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.bind((host, port))
    setter = getattr(mf, "set_log_level", None)
    if setter is not None:
        setter(0)
    create = getattr(mf, "create_config_store", None)
    if create is None:
        raise RuntimeError("Installed MemFabric has no create_config_store API")
    ret = create(url)
    if ret != 0:
        getter = getattr(mf, "get_last_err_msg", None)
        detail = getter() if getter is not None else "unavailable"
        raise RuntimeError(f"PD_STORE_FAILED ret={ret} store={url} native_error={detail}")
    with socket.create_connection((host, port), timeout=5):
        pass
    print(f"PD_STORE_READY store={url}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-url", default=os.getenv("ASCEND_MF_STORE_URL"))
    args = parser.parse_args()
    if os.getenv("QWEN_EXPERIMENT_HOST_ROLE") != "228":
        parser.error("Source env.sh 228 in the 228 container first")
    if not args.store_url:
        parser.error("Missing ASCEND_MF_STORE_URL / --store-url")
    host, _ = store_address(args.store_url)
    if host != os.getenv("HOST_228_IP"):
        parser.error("PD store must bind the configured HOST_228_IP")
    import memfabric_hybrid as mf

    start_store(args.store_url, mf)
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    done.wait()


if __name__ == "__main__":
    main()
