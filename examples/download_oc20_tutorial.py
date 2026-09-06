#!/usr/bin/env python
"""Download the official 100-train/20-validation OC20 tutorial S2EF data.

Source: facebookresearch/fairchem, fairchem_core-1.0.0,
tests/core/conftest.py::tutorial_dataset_path. Only the two S2EF LMDBs are
extracted. Data are licensed CC BY 4.0; cite the OC20 dataset paper.
"""

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/tutorial_data.tar.gz"
SHA256 = "81fe6511f87d435f463be34eccae9ac1dbe1e3e86789d4904966cb70e5a3077a"
MEMBERS = ("./s2ef/train_100/data.lmdb", "./s2ef/val_20/data.lmdb")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/oc20_tutorial"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name.removeprefix("./")).exists() for name in MEMBERS):
        parser.error(
            "Output already contains tutorial LMDBs; choose a fresh output directory."
        )
    with tempfile.TemporaryDirectory(prefix="tinnet-oc20-") as directory:
        path = Path(directory) / "tutorial.tar.gz"
        urlretrieve(URL, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SHA256:
            raise ValueError(
                "Tutorial archive checksum changed; verify the upstream source before using it."
            )
        with tarfile.open(path) as archive:
            for name in MEMBERS:
                member = archive.getmember(name)
                if not member.isfile():
                    raise ValueError(f"Expected a regular LMDB file: {name}")
                destination = args.output / name.removeprefix("./")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open(
                    "wb"
                ) as target:
                    import shutil

                    shutil.copyfileobj(source, target)
    (args.output / "source.json").write_text(
        json.dumps(
            {
                "url": URL,
                "sha256": digest,
                "license": "CC BY 4.0",
                "paper": "https://doi.org/10.1021/acscatal.0c04525",
                "note": "Tutorial fixture for smoke tests; not the OC20 benchmark.",
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
