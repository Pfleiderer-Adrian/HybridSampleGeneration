"""Download and prepare study-local DTD textures when no path is configured."""

import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

from PIL import Image
from tqdm import tqdm

from examples.common.image_io import IMAGE_EXTENSIONS
from .draem.synthesis import TextureSynthesizer


DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DTD_IMAGE_COUNT = 5640


def _download(destination):
    with urlopen(DTD_URL, timeout=60) as response, destination.open("wb") as file:
        size = response.headers.get("Content-Length")
        expected = int(size) if size else None
        received = 0
        with tqdm(total=expected, desc="Downloading DTD", unit="B", unit_scale=True) as progress:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
                received += len(chunk)
                progress.update(len(chunk))
        if expected is not None and received != expected:
            raise OSError(f"Incomplete DTD download: {received}/{expected} bytes.")


def _extract_images(archive, destination):
    """Copy regular images only; never extract links, devices or traversal paths."""
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError(f"Unsafe DTD archive path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported DTD archive entry: {member.name}")
        images = [member for member in members if member.isfile()
                  and PurePosixPath(member.name).parts[:2] == ("dtd", "images")
                  and PurePosixPath(member.name).suffix.lower() in IMAGE_EXTENSIONS]
        if len(images) != DTD_IMAGE_COUNT or len({m.name for m in images}) != len(images):
            raise ValueError(f"Expected {DTD_IMAGE_COUNT} unique DTD images, found {len(images)}.")
        for member in tqdm(images, desc="Preparing DTD textures", unit="image"):
            if member.size > 32 * 1024 * 1024:
                raise ValueError(f"Unexpected DTD image size: {member.name}")
            relative = PurePosixPath(member.name).relative_to("dtd")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            with Image.open(target) as image:
                image.verify()


def prepare_textures(study_folder, data):
    """Resolve and persist the texture path; do nothing for hybrid-only training."""
    if data.hybrid_fraction >= 1:
        return None
    if data.texture_root is not None:
        TextureSynthesizer(data.texture_root)
        return Path(data.texture_root)

    cache = Path(study_folder).resolve() / "downstream" / "textures"
    destination = cache / "dtd"
    images = destination / "images"
    marker = destination / "prepared.json"
    if destination.exists():
        if not marker.is_file():
            raise ValueError(f"Incomplete DTD directory: {destination}. Move it aside before retrying.")
        info = json.loads(marker.read_text())
        count = len(TextureSynthesizer(images).paths)
        if info.get("url") != DTD_URL or info.get("image_count") != DTD_IMAGE_COUNT or count != DTD_IMAGE_COUNT:
            raise ValueError(f"Invalid DTD cache: {destination}. Move it aside before retrying.")
        print(f"[DRAEM] Reusing DTD textures: {images}")
    else:
        cache.mkdir(parents=True, exist_ok=True)
        lock = cache / "download.lock"
        try:
            lock_file = lock.open("x")
        except FileExistsError as exc:
            raise RuntimeError(f"DTD preparation is locked: {lock}. If no download is running, remove this stale lock.") from exc
        try:
            with lock_file, tempfile.TemporaryDirectory(prefix="dtd-", dir=cache) as temporary:
                temporary = Path(temporary)
                archive = temporary / "dtd.tar.gz"
                staged = temporary / "prepared"
                print(f"[DRAEM] Preparing DTD in {destination}")
                _download(archive)
                _extract_images(archive, staged)
                (staged / "prepared.json").write_text(json.dumps({"url": DTD_URL, "image_count": DTD_IMAGE_COUNT}, indent=2) + "\n")
                staged.rename(destination)
        finally:
            lock.unlink(missing_ok=True)
    data.texture_root = str(images)
    return images
