"""Reject invalid batch selections before writing splits or starting work."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.mvtec_ad2.discovery import normalize_categories
from examples.mvtec_ad2.pipeline import run_existing_studies, run_study
from examples.mvtec_ad2.review import review_studies
from examples.mvtec_ad2.studies import prepare_studies
from .test_downstream import create_images


class StudySelectionTests(unittest.TestCase):
    def test_category_aliases_and_invalid_selections(self):
        self.assertEqual(normalize_categories((' CAN ', 'wall plugs')), ['can', 'wallplugs'])
        for names in ((), ('can', 'can'), ('wall plugs', 'wallplugs'), ('can', 'unknown')):
            with self.subTest(names=names), self.assertRaises(ValueError):
                normalize_categories(names)
        with self.assertRaises(TypeError):
            normalize_categories((None,))

    def test_invalid_later_category_does_not_write_earlier_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            output = root / 'studies'
            for categories, error in ((('can', 'unknown'), ValueError),
                                      (('can', 'fabric'), FileNotFoundError),
                                      (('can', 'can'), ValueError)):
                with self.subTest(categories=categories), self.assertRaises(error):
                    prepare_studies(root, categories, save_path=output)
                self.assertFalse(output.exists())

    def test_duplicate_paths_rejected_before_open_or_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'study'
            same = folder / '..' / 'study'
            with patch('examples.mvtec_ad2.pipeline.open_study') as opened:
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    run_existing_studies([folder, same], steps=('export',))
                opened.assert_not_called()
            with patch('examples.mvtec_ad2.review.open_study') as opened:
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    review_studies([folder, same])
                opened.assert_not_called()

    def test_explicit_empty_loader_is_not_replaced_by_manifest_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            study = prepare_studies(root, 'can', save_path=root / 'studies')[0]
            study = replace(study, sample_dataloader=[])
            with self.assertRaisesRegex(ValueError, 'no usable original samples'):
                run_study(study, steps=('ingest',))
