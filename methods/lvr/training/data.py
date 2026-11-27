"""Load the authors' annotations and materialize the matching Visual CoT images."""

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

ANNOTATIONS = {
    "viscot_363k_lvr_formatted.json": "6c0b40838309e50db5695052ef40b261a6a62106fa0b5563981bfc684db4cf77",
    "viscot_sroie_dude_lvr_formatted.json": "5448ff8c659dcff28ebd53b7c324cbe07b624adace4643866d95f9b6ce9562e5",
}
IMAGE_REVISION = "223d2d8c1146fda2bb918801b8276c587b78b61c"
ANNOTATION_URL = "https://drive.usercontent.google.com/download?id=1RUbKMQU3H7u8iWDqpDv8aiYDMlUaDIlE&export=download&confirm=t"


def load_rows(root):
    root.mkdir(parents=True, exist_ok=True)
    directory = root / "lvr_data"
    with FileLock(root / ".lvr-data.lock"):
        if not all((directory / name).is_file() for name in ANNOTATIONS):
            with urllib.request.urlopen(ANNOTATION_URL, timeout=120) as response:
                archive = response.read()
            if (
                hashlib.sha256(archive).hexdigest()
                != "605d23f7ba626fdd6c7b4e667ffa98defae9d30a411ac29322a4f9b9fec4c484"
            ):
                raise ValueError("LVR annotation archive checksum mismatch")
            directory.mkdir(exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
                for name in ANNOTATIONS:
                    tmp = directory / (name + ".tmp")
                    tmp.write_bytes(zipped.read("lvr_data/" + name))
                    tmp.replace(directory / name)
    rows = []
    for name, digest in ANNOTATIONS.items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Unexpected revision of {name}")
        rows.extend(json.loads(data))
    return rows


def prepare_images(root, rows):
    directory = root / "images" / "viscot"
    with FileLock(root / ".lvr-data.lock"):
        if directory.is_dir():
            return
        needed = {
            Path(row["image"][0]).relative_to("viscot").as_posix() for row in rows
        }
        source = Path(
            snapshot_download(
                "deepcs233/Visual-CoT",
                repo_type="dataset",
                revision=IMAGE_REVISION,
                allow_patterns=["cot_images_tar_split/cot_images_*"],
            )
        )
        parts = [source / f"cot_images_tar_split/cot_images_{i:02}" for i in range(13)]
        directory.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory.parent) as tmp:
            stage = Path(tmp) / "viscot"
            with subprocess.Popen(["cat", *parts], stdout=subprocess.PIPE) as stream:
                with tarfile.open(fileobj=stream.stdout, mode="r|*") as tar:
                    for member in tar:
                        relative = member.name.removeprefix("./").removeprefix(
                            "cot_image_data/"
                        )
                        if member.isfile() and relative in needed:
                            target = stage / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with (
                                tar.extractfile(member) as src,
                                target.open("wb") as dst,
                            ):
                                shutil.copyfileobj(src, dst)
                            needed.remove(relative)
                # Tar readers stop at the end marker before archive padding.
                stream.stdout.close()
                if stream.wait() not in (0, -13) or needed:
                    raise ValueError(
                        f"Incomplete Visual CoT archive ({len(needed)} missing images)"
                    )
            stage.rename(directory)
