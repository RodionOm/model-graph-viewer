"""Download pinned OpenVINO TinyLlama IR (2.20 GB), verifying every SHA256."""

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(output_dir: Path) -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "examples/tinyllama-source.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, metadata in manifest["files"].items():
        destination = output_dir / name
        if destination.is_file():
            if (
                destination.stat().st_size == metadata["size_bytes"]
                and digest(destination) == metadata["sha256"]
            ):
                print(f"Verified existing {destination}", flush=True)
                continue
            raise ValueError(
                f"Existing file failed verification: {destination}; move it and retry."
            )
        url = (
            f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
        )
        temporary = None
        print(f"Downloading {name}: {metadata['size_bytes']:,} bytes", flush=True)
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                tempfile.NamedTemporaryFile(
                    dir=output_dir, prefix=f".{name}.", suffix=".part", delete=False
                ) as target,
            ):
                temporary = Path(target.name)
                checksum = hashlib.sha256()
                size = 0
                reported = 0
                while chunk := response.read(4 * 1024 * 1024):
                    target.write(chunk)
                    checksum.update(chunk)
                    size += len(chunk)
                    if size - reported >= 128 * 1024 * 1024:
                        print(f"  {size / metadata['size_bytes']:.0%}", flush=True)
                        reported = size
            if size != metadata["size_bytes"] or checksum.hexdigest() != metadata["sha256"]:
                raise ValueError(f"Downloaded file failed size/SHA256 verification: {name}")
            os.replace(temporary, destination)
            print(f"Verified {destination}", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/tinyllama"))
    args = parser.parse_args()
    download(args.output_dir)


if __name__ == "__main__":
    main()
