"""Create private deployment/Pool configuration; never overwrite existing files."""
import argparse
import ipaddress
import json
import os
import shlex
from pathlib import Path


def render(ip_228, ip_238, ssh_user):
    for value in (ip_228, ip_238):
        if ipaddress.ip_address(value).version != 4:
            raise ValueError("Use the mutually reachable IPv4 host addresses")
    if ip_228 == ip_238:
        raise ValueError("228 and 238 must be different hosts")
    if not ssh_user or any(c.isspace() for c in ssh_user):
        raise ValueError("SSH account must be nonempty and contain no whitespace")
    task_root = "/ssd_2/yuanwenliu"
    values = {"HOST_228_IP": ip_228, "HOST_238_IP": ip_238, "SSH_USER": ssh_user,
              "HOST_IF_228": "enp67s0f5", "HOST_IF_238": "enp67s0f5",
              "EXPERIMENT_ROOT": task_root,
              "UMDK_PATH": f"{task_root}/umdk-qwen3-cam",
              "MODEL_PATH": f"{task_root}/models/Qwen3-30B-A3B"}
    endpoint = {"meta_service_url": f"tcp://{ip_238}:5000",
                "config_store_url": f"tcp://{ip_238}:6000"}
    meta = {**endpoint, "metrics_url": f"http://{ip_238}:8100", "log_level": "info"}
    pool = {**endpoint, "world_size": 32, "protocol": "device_rdma",
            "dram_size": 0, "hbm_size": "8GB", "max_hbm_size": "8GB", "log_level": "info"}
    client = {**pool, "hbm_size": 0, "max_hbm_size": 0}
    result = {"deployment.env": "".join(f"export {k}={shlex.quote(v)}\n" for k,v in values.items())}
    result.update({name: json.dumps(data, indent=2)+"\n" for name,data in
                   (("meta.json",meta),("pool.json",pool),("client.json",client))})
    return result


def write_configs(directory, configs):
    directory = Path(directory)
    if any((directory / name).exists() for name in configs):
        raise FileExistsError("Deployment files already exist; review them, do not overwrite")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, content in configs.items():
        fd = os.open(directory/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(content)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip-228", dest="ip_228")
    p.add_argument("--ip-238", dest="ip_238")
    p.add_argument("--ssh-user")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "runtime-config")
    args = p.parse_args()
    configs = render(args.ip_228 or input("228 host IPv4: ").strip(),
                     args.ip_238 or input("238 host IPv4: ").strip(),
                     args.ssh_user or input("SSH account for 228/238: ").strip())
    write_configs(args.output_dir, configs)
    print("PRIVATE_CONFIG_READY", args.output_dir, "(not committed to Git)")


if __name__ == "__main__":
    main()
