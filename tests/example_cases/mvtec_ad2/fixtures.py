"""Shared MVTec AD 2 test fixtures: a tiny category and a seeded hybrid study."""
import numpy as np
from PIL import Image

from examples.mvtec_ad2.data.dataset import MVTecAD2Dataloader
from examples.mvtec_ad2.data.splits import manifest_samples
from hybrid_sample_generator import Configuration, HybridDataGenerator
from hybrid_sample_generator.domain.records import HybridSample, Placement, RealAnomaly, SyntheticAnomaly


def create_images(root):
    for label in ("good", "bad"):
        folder = root / "can" / "test_public" / label
        folder.mkdir(parents=True)
        for index in range(8):
            image = np.full((64, 64, 3), 80 + index, dtype=np.uint8)
            if label == "bad":
                image[20:40, 20:40] = 220
            Image.fromarray(image).save(folder / f"{index}.png")
            if label == "bad":
                masks = root / "can" / "test_public" / "ground_truth" / "bad"
                masks.mkdir(parents=True, exist_ok=True)
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[20:40, 20:40] = 255
                Image.fromarray(mask).save(masks / f"{index}.png")

def generator_config(folder):
    config = Configuration("mvtec-test", study_folder=folder)
    config.extraction.anomaly_size = (3, 64, 64)
    return config

def seed_hybrid(config, manifest):
    generator = HybridDataGenerator(config)
    generator.ingest_dataset(MVTecAD2Dataloader(manifest_samples(manifest, "train")))
    repo, store = generator.repository, generator.artifact_store
    healthy = repo.list_original_samples(has_anomaly=False)[0]
    donor = repo.list_original_samples(has_anomaly=True)[0]
    target = store.load_array(healthy.image_path)
    image = target.copy()
    image[:, 20:40, 20:40] = 220
    mask = np.zeros((1, 64, 64), dtype=np.float32)
    mask[:, 20:40, 20:40] = 1
    image_path = store.save_entity_array("hybrid_samples", "hybrid", "image", image)
    mask_path = store.save_entity_array("hybrid_samples", "hybrid", "segmentation", mask)
    real = RealAnomaly("real", donor.id, 0, donor.image_path, donor.segmentation_path,
                       donor.image_path, donor.segmentation_path, 2, None, .5, .5)
    repo.upsert_real_anomaly(real)
    repo.upsert_synthetic_anomaly(SyntheticAnomaly("synth", real.id, 0, image_path, mask_path, 42))
    repo.upsert_hybrid_sample(HybridSample("hybrid", healthy.id, 0, image_path, mask_path, "generated"))
    repo.upsert_placement(Placement("placement", "hybrid", "synth", 0, 2, None, .5, .5))
    return repo, store
