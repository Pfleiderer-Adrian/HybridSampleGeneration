"""MVTec DTD preparation without network access or a full dataset download."""

import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from examples.mvtec_ad2.downstream.configuration import DataConfiguration
from examples.mvtec_ad2.downstream.textures import prepare_textures


def archive_at(path, name="dtd/images/banded/image.png", link=False):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    payload = buffer.getvalue()
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo(name)
        if link:
            entry.type = tarfile.SYMTYPE
            entry.linkname = "/tmp/outside"
            archive.addfile(entry)
        else:
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))


class TexturePreparationTests(unittest.TestCase):
    def test_download_prepare_and_reuse(self):
        with tempfile.TemporaryDirectory() as folder, patch("examples.mvtec_ad2.downstream.textures.DTD_IMAGE_COUNT", 1):
            with patch("examples.mvtec_ad2.downstream.textures._download", side_effect=archive_at) as download:
                data = DataConfiguration()
                images = prepare_textures(folder, data)
                self.assertEqual(images, Path(folder)/"downstream"/"textures"/"dtd"/"images")
                self.assertTrue((images/"banded"/"image.png").is_file())
                self.assertEqual(data.texture_root, str(images))
                self.assertEqual(prepare_textures(folder, DataConfiguration()), images)
                download.assert_called_once()
            self.assertEqual(sorted(p.name for p in images.parent.parent.iterdir()), ["dtd"])

    def test_unused_or_explicit_textures_do_not_download(self):
        with tempfile.TemporaryDirectory() as folder, patch("examples.mvtec_ad2.downstream.textures._download") as download:
            study = Path(folder)/"study"
            self.assertIsNone(prepare_textures(study, DataConfiguration(hybrid_fraction=1)))
            self.assertFalse(study.exists())
            Image.new("RGB", (8, 8)).save(Path(folder)/"texture.png")
            self.assertEqual(prepare_textures(study, DataConfiguration(texture_root=folder)), Path(folder))
            with self.assertRaises(ValueError):
                prepare_textures(study, DataConfiguration(texture_root=str(Path(folder)/"missing")))
            download.assert_not_called()

    def test_failed_download_cleans_up_and_can_retry(self):
        with tempfile.TemporaryDirectory() as folder, patch("examples.mvtec_ad2.downstream.textures.DTD_IMAGE_COUNT", 1):
            data = DataConfiguration()
            with patch("examples.mvtec_ad2.downstream.textures._download", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    prepare_textures(folder, data)
            self.assertIsNone(data.texture_root)
            self.assertEqual(list((Path(folder)/"downstream"/"textures").iterdir()), [])
            with patch("examples.mvtec_ad2.downstream.textures._download", side_effect=archive_at):
                self.assertTrue(prepare_textures(folder, data).is_dir())

    def test_unsafe_archive_is_rejected(self):
        for name, link in (("../escape.png", False), ("dtd/images/link.png", True)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                with patch("examples.mvtec_ad2.downstream.textures._download", side_effect=lambda path: archive_at(path, name, link)):
                    with self.assertRaises(ValueError):
                        prepare_textures(folder, DataConfiguration())
                self.assertFalse((Path(folder)/"downstream"/"textures"/"dtd").exists())
