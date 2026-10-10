"""Create private deployment/Pool configuration; never overwrite existing files."""
import argparse
import ipaddress
import json
import os
import shlex
from pathlib import Path


def render(a_ip, b_ip, ssh_user):
    for value in (a_ip, b_ip):
        if ipaddress.ip_address(value).version != 4:
            raise ValueError("Use the mutually reachable IPv4 host addresses")
    if a_ip == b_ip:
        raise ValueError("A and B must be different hosts")
    if not ssh_user or any(c.isspace() for c in ssh_user):
        raise ValueError("SSH account must be nonempty and contain no whitespace")
    values = {"A_IP": a_ip, "B_IP": b_ip, "A2_SSH_USER": ssh_user,
              "HOST_IF_A": "enp67s0f5", "HOST_IF_B": "enp67s0f5",
              "MODEL_PATH": "/ssd_2/models/Qwen3-30B-A3B"}
    endpoint = {"meta_service_url": f"tcp://{b_ip}:5000",
                "config_store_url": f"tcp://{b_ip}:6000"}
    meta = {**endpoint, "metrics_url": f"http://{b_ip}:8100", "log_level": "info"}
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
    p.add_argument("--a-ip")
    p.add_argument("--b-ip")
    p.add_argument("--ssh-user")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "runtime-config")
    args = p.parse_args()
    configs = render(args.a_ip or input("A host IPv4: ").strip(),
                     args.b_ip or input("B host IPv4: ").strip(),
                     args.ssh_user or input("A2 SSH account: ").strip())
    write_configs(args.output_dir, configs)
    print("PRIVATE_CONFIG_READY", args.output_dir, "(not committed to Git)")


if __name__ == "__main__":
    main()
