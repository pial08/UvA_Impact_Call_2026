"""Extract the BPE tokenizer used to count tokens in the pilot.

The pilot counts tokens with the legacy Claude tokenizer that shipped inside the
anthropic Python SDK up to version 0.34.x (MIT licence). Shares of tokens are
insensitive to the tokenizer, so any modern BPE tokenizer gives nearly the same
percentages; this script only makes the published numbers exactly reproducible.

Usage:  python get_tokenizer.py --out ../data
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="../data")
    p.add_argument("--version", default="0.34.2")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "pip", "download", f"anthropic=={args.version}",
                        "--no-deps", "--only-binary=:all:", "-d", tmp], check=True)
        wheel = glob.glob(os.path.join(tmp, "anthropic-*.whl"))[0]
        with zipfile.ZipFile(wheel) as z:
            name = next(n for n in z.namelist() if n.endswith("tokenizer.json"))
            target = os.path.join(args.out, "claude_legacy_tokenizer.json")
            with open(target, "wb") as f:
                f.write(z.read(name))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
