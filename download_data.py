"""Download and verify the recorded DeepLense archive; preserve existing files."""
import argparse
import hashlib
from pathlib import Path

FILE_ID = "1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5"
SHA256 = "ddfe91d32f79da92c5610295f3dd3475b7e40d5788e16cd866cb3e9487cb7b9c"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/deeplense_observational.zip"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    path = args.output
    if not args.verify_only:
        partial = path.with_name(path.name + ".partial")
        if path.exists() or partial.exists():
            parser.error("Output or partial download exists; choose another path or use --verify-only.")
        import gdown
        path.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(id=FILE_ID, output=str(partial), quiet=False, use_cookies=False)
        if result is None:
            raise RuntimeError("Download failed. Use the manual Drive link in DATASET.md.")
        actual = digest(partial)
        if actual != SHA256:
            raise ValueError(f"Archive hash mismatch: {actual}. Partial file preserved for inspection.")
        partial.rename(path)
    actual = digest(path)
    if actual != SHA256:
        raise ValueError(f"Archive hash mismatch: {actual}")
    print(f"Verified {path}: SHA-256 {actual}")


if __name__ == "__main__":
    main()
