"""Tests for the image-folder and NIfTI loaders of the bundled examples."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from examples.image_2d.image_dataloader import ImageDataloader
from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator

HAS_NIBABEL = importlib.util.find_spec("nibabel") is not None


def _ingest(config, loader):
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        return HybridDataGenerator(config).ingest_dataset(loader)


class ImageDataloaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.images, self.masks = root / "images", root / "masks"
        self.images.mkdir()
        self.masks.mkdir()
        rng = np.random.default_rng(0)
        # "defect" has a mask with a suffix, "clean" an empty mask, "unlabelled" none.
        for name in ("defect", "clean", "unlabelled"):
            Image.fromarray(rng.integers(0, 255, (12, 16, 3), dtype=np.uint8), mode="RGB").save(
                self.images / f"{name}.png"
            )
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[3:7, 4:9] = 255
        Image.fromarray(mask, mode="L").save(self.masks / "defect_mask.png")
        Image.fromarray(np.zeros((12, 16), dtype=np.uint8), mode="L").save(self.masks / "clean_seg.png")
        # A mask without an image and a file that is no image are ignored.
        Image.fromarray(mask, mode="L").save(self.masks / "orphan_label.png")
        (self.images / "notes.txt").write_text("not an image")

    def tearDown(self):
        self.temporary.cleanup()

    def _loader(self, **kwargs):
        return ImageDataloader(str(self.images), str(self.masks), **kwargs)

    def test_images_and_masks_are_paired_by_name(self):
        samples = {name: (image, mask) for image, mask, name in self._loader()}

        self.assertEqual(set(samples), {"defect.png", "clean.png", "unlabelled.png"})
        image, mask = samples["defect.png"]
        self.assertEqual(image.shape, (3, 12, 16))
        self.assertEqual(image.dtype, np.float32)
        self.assertEqual(mask.shape, (1, 12, 16))
        self.assertEqual(int(np.count_nonzero(mask)), 20)
        self.assertEqual(int(np.count_nonzero(samples["clean.png"][1])), 0)
        self.assertIsNone(samples["unlabelled.png"][1])

    def test_dataset_properties_are_discovered(self):
        info = self._loader().sample_infos
        self.assertEqual((info.height, info.width, info.channels), (12, 16, 3))
        self.assertEqual(info.ndim, 3)

    def test_normalization_and_returned_paths(self):
        raw = {name: image for image, _, name in self._loader()}
        self.assertGreater(float(raw["defect.png"].max()), 1.0)
        for image, mask, name, image_path, mask_path in self._loader(normalize=True, return_paths=True):
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)
            self.assertEqual(Path(image_path).name, name)
            self.assertEqual(mask_path is None, mask is None)

    def test_modality_suffix_is_part_of_the_image_name(self):
        for name in ("defect", "clean", "unlabelled"):
            (self.images / f"{name}.png").rename(self.images / f"{name}_t1.png")
        samples = {name: mask for _, mask, name in self._loader(modality="t1")}
        self.assertEqual(set(samples), {"defect_t1.png", "clean_t1.png", "unlabelled_t1.png"})
        self.assertIsNotNone(samples["defect_t1.png"])
        self.assertIsNone(samples["unlabelled_t1.png"])

    def test_color_masks_are_reduced_to_one_channel_unless_kept(self):
        color = np.zeros((12, 16, 3), dtype=np.uint8)
        color[3:7, 4:9] = 255
        Image.fromarray(color, mode="RGB").save(self.masks / "defect_mask.png")
        reduced = {name: mask for _, mask, name in self._loader()}
        kept = {name: mask for _, mask, name in self._loader(keep_mask_channels=True)}
        self.assertEqual(reduced["defect.png"].shape, (1, 12, 16))
        self.assertEqual(kept["defect.png"].shape, (3, 12, 16))

    def test_typed_samples_carry_their_source_paths(self):
        samples = list(self._loader().iter_input_samples())
        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertIsInstance(sample, InputSample)
            self.assertEqual(Path(sample.source_image_path).name, sample.source_name)
        by_name = {sample.source_name: sample for sample in samples}
        self.assertTrue(by_name["defect.png"].source_segmentation_path.endswith("defect_mask.png"))
        self.assertIsNone(by_name["unlabelled.png"].segmentation)

    def test_empty_folders_are_reported(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(FileNotFoundError, "Could not discover dataset"):
                ImageDataloader(empty, empty)

    def test_loader_feeds_the_pipeline(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration("image-loader", study_folder=str(Path(root) / "study"))
            config.extraction.anomaly_size = (3, 8, 8)
            summary = _ingest(config, self._loader())
        self.assertEqual(summary.total_samples, 3)
        self.assertEqual(summary.anomalous_samples, 1)
        self.assertEqual(summary.control_samples, 2)
        self.assertEqual(summary.unannotated_samples, 1)
        self.assertEqual(summary.spatial_dimensions, 2)


@unittest.skipUnless(HAS_NIBABEL, "nibabel is required for the NIfTI example loader")
class NiftiDataloaderTests(unittest.TestCase):
    def setUp(self):
        import nibabel

        self.nibabel = nibabel
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.images, self.masks = root / "images", root / "masks"
        self.images.mkdir()
        self.masks.mkdir()
        rng = np.random.default_rng(0)
        self.affine = np.diag([1.0, 2.0, 3.0, 1.0])
        # NIfTI stores (H, W, D); the loader returns (C, D, H, W).
        self.volume = rng.random((10, 12, 6)).astype(np.float32)
        self.mask = np.zeros((10, 12, 6), dtype=np.uint8)
        self.mask[2:5, 3:7, 1:4] = 1
        self._save(self.volume, self.images / "defect.nii.gz")
        self._save(self.mask, self.masks / "defect.nii.gz")
        self._save(rng.random((10, 12, 6)).astype(np.float32), self.images / "unlabelled.nii.gz")

    def tearDown(self):
        self.temporary.cleanup()

    def _save(self, array, path):
        self.nibabel.save(self.nibabel.Nifti1Image(array, self.affine), str(path))

    def _loader(self, **kwargs):
        from examples.nifti_3d.nifti_dataloader import NiftiDataloader

        return NiftiDataloader(str(self.images), str(self.masks), "t1", **kwargs)

    def test_volumes_are_returned_channel_and_depth_first(self):
        samples = {name: (image, mask) for image, mask, name in self._loader()}

        self.assertEqual(set(samples), {"defect", "unlabelled"})
        image, mask = samples["defect"]
        self.assertEqual(image.shape, (1, 6, 10, 12))
        self.assertEqual(mask.shape, (1, 6, 10, 12))
        np.testing.assert_allclose(image[0], self.volume.transpose(2, 0, 1), atol=1e-6)
        np.testing.assert_array_equal(mask[0] > 0, self.mask.transpose(2, 0, 1) > 0)
        self.assertIsNone(samples["unlabelled"][1])

    def test_affines_and_typed_samples(self):
        for image, mask, name, image_affine, mask_affine in self._loader(return_affine=True):
            np.testing.assert_allclose(image_affine, self.affine)
            self.assertEqual(mask_affine is None, mask is None)
        samples = {sample.source_name: sample for sample in self._loader().iter_input_samples()}
        self.assertEqual(set(samples), {"defect", "unlabelled"})
        self.assertTrue(samples["defect"].source_image_path.endswith("defect.nii.gz"))
        self.assertEqual(samples["defect"].image.shape, (1, 6, 10, 12))
        self.assertIsNone(samples["unlabelled"].segmentation)

    def test_array_layouts_are_converted_to_channel_depth_height_width(self):
        from examples.nifti_3d.nifti_dataloader import ensure_cdhw, minmax01

        self.assertEqual(ensure_cdhw(np.zeros((10, 12))).shape, (1, 1, 10, 12))
        self.assertEqual(ensure_cdhw(np.zeros((10, 12, 6))).shape, (1, 6, 10, 12))
        self.assertEqual(ensure_cdhw(np.zeros((10, 12, 6, 2)), channels_hint=2).shape, (2, 6, 10, 12))
        with self.assertRaises(ValueError):
            ensure_cdhw(np.zeros(5))

        scaled = minmax01(np.stack([np.arange(8.0).reshape(2, 2, 2), np.full((2, 2, 2), 3.0)]))
        self.assertAlmostEqual(float(scaled[0].min()), 0.0)
        self.assertAlmostEqual(float(scaled[0].max()), 1.0, places=5)
        self.assertEqual(float(scaled[1].max()), 0.0)

    def test_saved_volume_can_be_loaded_again(self):
        from examples.nifti_3d.nifti_dataloader import ensure_cdhw, save_cdhw_as_nifti

        volume = ensure_cdhw(self.volume)
        target = Path(self.temporary.name) / "output" / "saved.nii.gz"
        save_cdhw_as_nifti(volume, self.affine, str(target))

        loaded = self.nibabel.load(str(target))
        np.testing.assert_allclose(loaded.affine, self.affine)
        np.testing.assert_allclose(np.asarray(loaded.dataobj), self.volume, atol=1e-6)
        with self.assertRaises(ValueError):
            save_cdhw_as_nifti(volume[0], self.affine, str(target))
        with self.assertRaises(ValueError):
            save_cdhw_as_nifti(volume, np.eye(3), str(target))

    def test_loader_feeds_the_pipeline(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration("nifti-loader", study_folder=str(Path(root) / "study"))
            config.extraction.anomaly_size = (1, 4, 4, 4)
            config.model.set_model("cVAE_ConvNeXt_3D")
            summary = _ingest(config, self._loader())
        self.assertEqual(summary.total_samples, 2)
        self.assertEqual(summary.anomalous_samples, 1)
        self.assertEqual(summary.unannotated_samples, 1)
        self.assertEqual(summary.spatial_dimensions, 3)


if __name__ == "__main__":
    unittest.main()
