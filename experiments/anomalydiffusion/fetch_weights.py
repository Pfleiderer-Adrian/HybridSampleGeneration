"""Download and verify the public original LDM checkpoint, without credentials."""
import argparse
from pathlib import Path
import urllib.request

from .prepare import sha256, write_json

URL = "https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt"
SHA256 = "7c95fc34febdbf15fea1481a87db43c935ecadeb1e300dd6bc3de3f5f663e298"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(URL, temporary)
        if sha256(temporary) != SHA256:
            raise ValueError("Downloaded checkpoint checksum differs from the verified original.")
        temporary.replace(path)
    if sha256(path) != SHA256:
        raise ValueError("Existing checkpoint checksum differs from the verified original.")
    write_json(path.with_suffix(".json"), {"url": URL, "sha256": SHA256, "bytes": path.stat().st_size})
    print(path)


if __name__ == "__main__":
    main()
