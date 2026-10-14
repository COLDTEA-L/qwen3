"""Run in a separate download venv on the host, never in the inference image."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--insecure", action="store_true", help="Explicitly disable TLS verification for this download only")
    parser.add_argument("--directory", type=Path, default=Path("/ssd_2/yuanwenliu/models/Qwen3-30B-A3B"))
    args = parser.parse_args()
    import huggingface_hub
    if huggingface_hub.__version__ != "0.34.4":
        raise RuntimeError("This downloader requires huggingface_hub==0.34.4 in an isolated venv")
    if args.insecure:
        import requests
        from huggingface_hub import configure_http_backend
        def factory():
            session = requests.Session()
            session.verify = False
            return session
        configure_http_backend(backend_factory=factory)
        print("WARNING: TLS verification disabled ONLY in this downloader; trusted network required", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen3-30B-A3B", local_dir=str(args.directory), max_workers=2,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    print("WEIGHTS_DOWNLOAD_COMPLETE", args.directory)


if __name__ == "__main__":
    main()
