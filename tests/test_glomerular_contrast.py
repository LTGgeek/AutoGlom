import csv
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use('Agg')

from uhdog import uhdog
from uhdog.utils import calculate_cortex_background, calculate_glomerular_contrasts


class CortexBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((5, 5, 5), 0.1)
        self.labels = np.zeros(self.image.shape, dtype=int)
        self.labels[2, 2, 2] = 1
        self.image[2, 2, 2] = 0.2
        self.cortex = np.ones(self.image.shape, dtype=bool)

    def test_uses_distant_cortex_voxels_and_excludes_equal_or_darker_voxels(self):
        self.image[0, 0, 0] = 0.6
        self.image[4, 4, 4] = 0.8
        self.image[1, 2, 2] = 0.2
        result = calculate_cortex_background(self.image, self.labels, self.cortex)
        self.assertAlmostEqual(result['mean_glomerular_intensity'], 0.2)
        self.assertEqual(result['cortex_background_voxel_count'], 2)
        self.assertAlmostEqual(result['mean_cortex_background_intensity'], 0.7)

    def test_excludes_voxels_outside_cortex_and_all_glomeruli(self):
        self.image[0, 0, 0] = 0.6
        self.image[4, 4, 4] = 1.0
        self.cortex[4, 4, 4] = False
        self.labels[2, 2, 3] = 1
        self.image[2, 2, 3] = 0.8
        result = calculate_cortex_background(self.image, self.labels, self.cortex)
        self.assertAlmostEqual(result['mean_glomerular_intensity'], 0.5)
        self.assertEqual(result['cortex_background_voxel_count'], 1)
        self.assertAlmostEqual(result['mean_cortex_background_intensity'], 0.6)

    def test_threshold_is_mean_of_all_glomerular_voxels(self):
        self.labels.fill(0)
        self.labels[1, 1, 1] = 1
        self.labels[1, 1, 2] = 1
        self.labels[3, 3, 3] = 2
        self.image[1, 1, 1] = 0.1
        self.image[1, 1, 2] = 0.3
        self.image[3, 3, 3] = 0.8
        self.image[0, 0, 0] = 0.45
        result = calculate_cortex_background(self.image, self.labels, self.cortex)
        self.assertEqual(result['glomerular_intensity_voxel_count'], 3)
        self.assertAlmostEqual(result['mean_glomerular_intensity'], 0.4)
        self.assertEqual(result['cortex_background_voxel_count'], 1)
        self.assertAlmostEqual(result['mean_cortex_background_intensity'], 0.45)

    def test_no_glomeruli_has_no_threshold_or_background_mean(self):
        self.labels.fill(0)
        result = calculate_cortex_background(self.image, self.labels, self.cortex)
        self.assertIsNone(result['mean_glomerular_intensity'])
        self.assertIsNone(result['mean_cortex_background_intensity'])
        self.assertEqual(result['cortex_background_voxel_count'], 0)

    def test_no_eligible_cortex_voxels_has_no_background_mean(self):
        result = calculate_cortex_background(self.image, self.labels, self.cortex)
        self.assertIsNone(result['mean_cortex_background_intensity'])
        self.assertEqual(result['cortex_background_voxel_count'], 0)

    def test_empty_cortex_has_no_background_mean(self):
        self.cortex.fill(False)
        result = calculate_cortex_background(self.image, self.labels, self.cortex)
        self.assertIsNone(result['mean_cortex_background_intensity'])
        self.assertEqual(result['cortex_background_voxel_count'], 0)

    def test_zero_glomerular_intensities_are_included_in_threshold(self):
        self.image[2, 2, 2] = 0
        result = calculate_cortex_background(self.image, self.labels, self.cortex)
        self.assertEqual(result['mean_glomerular_intensity'], 0)
        self.assertEqual(result['cortex_background_voxel_count'], 124)

    def test_nonfinite_intensities_are_excluded_from_both_means(self):
        self.labels[2, 2, 3] = 1
        self.image[2, 2, 3] = np.nan
        self.image[0, 0, 0] = np.inf
        self.image[4, 4, 4] = 0.6
        result = calculate_cortex_background(self.image, self.labels, self.cortex)
        self.assertEqual(result['glomerular_intensity_voxel_count'], 1)
        self.assertAlmostEqual(result['mean_glomerular_intensity'], 0.2)
        self.assertEqual(result['cortex_background_voxel_count'], 1)
        self.assertAlmostEqual(result['mean_cortex_background_intensity'], 0.6)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            calculate_cortex_background(self.image, self.labels, self.cortex[:, :, :2])
        with self.assertRaises(ValueError):
            calculate_cortex_background(self.image, self.labels[:, :, :2], self.cortex)


class GlomerularContrastTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((5, 5, 5), 0.6)
        self.labels = np.zeros(self.image.shape, dtype=int)
        self.labels[2, 2, 2] = 1
        self.image[2, 2, 2] = 0.2
        self.cortex = np.ones(self.image.shape, dtype=bool)

    def measure(self, image=None):
        return calculate_glomerular_contrasts(
            self.image if image is None else image, self.labels, self.cortex
        )

    def test_cortex_background_and_per_glomerulus_formula(self):
        records, background = self.measure()
        record, = records
        self.assertEqual((record['row'], record['column'], record['slice']), (2, 2, 2))
        self.assertEqual(record['background_voxel_count'], 124)
        self.assertAlmostEqual(record['background_mean_intensity'], 0.6)
        self.assertAlmostEqual(record['contrast'], 2.0)
        self.assertEqual(background['mean_glomerular_intensity'], 0.2)
        self.assertEqual(record['status'], 'ok')

    def test_uses_minimum_voxel_even_when_it_is_not_first(self):
        self.labels[2, 2, 3] = 1
        self.image[2, 2, 3] = 0.01
        records, background = self.measure()
        record, = records
        self.assertEqual(record['slice'], 3)
        self.assertEqual(record['glom_intensity'], 0.01)
        self.assertEqual(record['background_voxel_count'], 123)
        self.assertAlmostEqual(background['mean_glomerular_intensity'], 0.105)
        self.assertAlmostEqual(record['contrast'], 59.0)

    def test_excludes_brighter_voxels_in_own_and_other_glomeruli(self):
        self.image.fill(0.8)
        self.image[2, 2, 2] = 0.2
        self.labels[2, 2, 3] = 1
        self.image[2, 2, 3] = 0.9
        self.labels[2, 2, 1] = 2
        self.image[2, 2, 1] = 0.9
        records, _ = self.measure()
        for record in records:
            self.assertEqual(record['background_voxel_count'], 122)
            self.assertAlmostEqual(record['background_mean_intensity'], 0.8)
        self.assertAlmostEqual(records[0]['contrast'], 3.0)

    def test_shared_background_can_give_negative_individual_contrast(self):
        self.labels[4, 4, 4] = 2
        self.image[4, 4, 4] = 0.8
        records, background = self.measure()
        self.assertAlmostEqual(background['mean_glomerular_intensity'], 0.5)
        self.assertAlmostEqual(records[0]['contrast'], 2.0)
        self.assertAlmostEqual(records[1]['contrast'], -0.25)
        self.assertEqual(records[1]['status'], 'ok')

    def test_minimum_voxel_is_found_with_irregular_bounds(self):
        self.labels.fill(0)
        self.image.fill(0.6)
        self.labels[1, 2, 3] = 1
        self.labels[2, 1, 1] = 1
        self.image[1, 2, 3] = 0.3
        self.image[2, 1, 1] = 0.01
        records, _ = self.measure()
        record, = records
        self.assertEqual((record['row'], record['column'], record['slice']), (2, 1, 1))
        self.assertAlmostEqual(record['contrast'], 59.0)

    def test_minimum_ties_use_first_voxel_in_array_order(self):
        self.labels[2, 2, 3] = 1
        self.image[2, 2, 2:4] = 0.1
        records, _ = self.measure()
        record, = records
        self.assertEqual((record['row'], record['column'], record['slice']), (2, 2, 2))
        self.assertEqual(record['glom_intensity'], 0.1)

    def test_minimum_search_excludes_background_and_other_labels(self):
        self.labels.fill(0)
        self.labels[1, 2, 3] = 1
        self.labels[2, 1, 1] = 1
        self.labels[1, 1, 2] = 2
        self.image[1, 2, 3] = 0.4
        self.image[2, 1, 1] = 0.2
        self.image[1, 1, 2] = 0.01
        self.image[1, 1, 1] = 0
        records, _ = self.measure()
        self.assertEqual(records[0]['glom_intensity'], 0.2)
        self.assertEqual((records[0]['row'], records[0]['column'], records[0]['slice']), (2, 1, 1))

    def test_zero_minimum_at_later_voxel_is_excluded(self):
        self.labels[2, 2, 3] = 1
        self.image[2, 2, 3] = 0
        records, _ = self.measure()
        record, = records
        self.assertEqual(record['slice'], 3)
        self.assertEqual(record['glom_intensity'], 0)
        self.assertIsNone(record['contrast'])
        self.assertEqual(record['status'], 'zero_intensity')

    def test_minimum_uses_finite_voxels_in_mixed_glomerulus(self):
        self.labels[2, 2, 3] = 1
        self.image[2, 2, 3] = 0.1
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                image = self.image.copy()
                image[2, 2, 2] = value
                records, _ = self.measure(image)
                record, = records
                self.assertEqual(record['slice'], 3)
                self.assertEqual(record['glom_intensity'], 0.1)
                self.assertAlmostEqual(record['contrast'], 5.0)
                self.assertEqual(record['status'], 'ok')

    def test_edge_glomerulus_uses_the_whole_cortex(self):
        self.labels.fill(0)
        self.labels[0, 0, 0] = 1
        self.image.fill(0.6)
        self.image[0, 0, 0] = 0.2
        records, _ = self.measure()
        record, = records
        self.assertEqual(record['background_voxel_count'], 124)
        self.assertAlmostEqual(record['contrast'], 2.0)

    def test_zero_intensity_is_recorded_without_division(self):
        self.image[2, 2, 2] = 0
        records, _ = self.measure()
        record, = records
        self.assertEqual(record['background_voxel_count'], 124)
        self.assertIsNone(record['contrast'])
        self.assertEqual(record['status'], 'zero_intensity')

    def test_no_background_cortex_voxels_is_recorded(self):
        self.labels.fill(1)
        records, _ = self.measure()
        record, = records
        self.assertEqual(record['background_voxel_count'], 0)
        self.assertIsNone(record['background_mean_intensity'])
        self.assertIsNone(record['contrast'])
        self.assertEqual(record['status'], 'no_eligible_cortex_voxels')

    def test_nonfinite_intensities_are_excluded(self):
        for location, value in (((2, 2, 2), np.nan),
                                ((2, 2, 2), np.inf)):
            with self.subTest(location=location, value=value):
                image = self.image.copy()
                image[location] = value
                records, _ = self.measure(image)
                record, = records
                self.assertIsNone(record['contrast'])
                self.assertEqual(record['status'], 'nonfinite_intensity')

    def test_nonfinite_background_voxels_are_not_used(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                image = self.image.copy()
                image[1, 2, 2] = value
                records, _ = self.measure(image)
                record, = records
                self.assertEqual(record['background_voxel_count'], 123)
                self.assertAlmostEqual(record['contrast'], 2.0)

    def test_no_brighter_cortex_voxels_excludes_the_glomerulus(self):
        for intensity in (0.1, 0.2):
            with self.subTest(intensity=intensity):
                self.image.fill(intensity)
                self.image[2, 2, 2] = 0.2
                records, _ = self.measure()
                record, = records
                self.assertEqual(record['background_voxel_count'], 0)
                self.assertIsNone(record['background_mean_intensity'])
                self.assertIsNone(record['contrast'])
                self.assertEqual(record['status'], 'no_eligible_cortex_voxels')

    def test_empty_segmentation(self):
        self.labels.fill(0)
        records, background = self.measure()
        self.assertEqual(records, [])
        self.assertIsNone(background['mean_glomerular_intensity'])

    def test_missing_label_ids_preserve_actual_ids(self):
        self.labels[4, 4, 4] = 3
        records, _ = self.measure()
        self.assertEqual([record['glom_id'] for record in records], [1, 3])

    def test_rejects_mismatched_shapes_and_non_3d_images(self):
        with self.assertRaises(ValueError):
            calculate_glomerular_contrasts(self.image, self.labels[:, :, :2], self.cortex)
        with self.assertRaises(ValueError):
            calculate_glomerular_contrasts(
                self.image[:, :, 0], self.labels[:, :, 0], self.cortex[:, :, 0]
            )


class ContrastOutputTests(unittest.TestCase):
    def run_analysis(self, folder, empty=False, no_eligible_background=False,
                     mask_exclusions=False, use_blackdot_mask=False, multi_voxel_glomerulus=False):
        image = np.full((11, 11, 11), 0.1 if no_eligible_background or mask_exclusions else 0.6)
        labels = np.zeros(image.shape, dtype=int)
        medulla = np.zeros(image.shape)
        kidney = np.ones(image.shape)
        blackdots = np.zeros(image.shape)
        if not empty:
            for glom_id, location, intensity in ((1, (2, 2, 2), 0.2),
                                                 (2, (5, 5, 5), 0.3),
                                                 (3, (8, 8, 8), 0.0)):
                labels[location] = glom_id
                image[location] = intensity
        if multi_voxel_glomerulus:
            labels[2, 2, 3] = 1
            image[2, 2, 3] = 0.1
        if mask_exclusions:
            image[0, 0, :3] = 1.0
            kidney[0, 0, 0] = 0
            medulla[0, 0, 1] = 1
            blackdots[0, 0, 2] = 1
            image[4, 4, 4] = 0.6
        config = {
            'folder_path': folder, 'kidney': 'test', 'sslice': 0, 'eslice': 10,
            'med_start_slice': 0, 'med_end_slice': 10, 'dist_boundary': 0,
            'inten_thre': 0.1, 'perct': 0.2, 'unet_mask_threshold': 0.5,
            'x_space': 0.01, 'y_space': 0.01, 'z_space': 0.01,
            'use_blackdot_mask': use_blackdot_mask,
        }
        volumes = np.array([]) if empty else np.array([0.0002, 0.0, 0.0])
        annotation_masks = [medulla, kidney]
        if use_blackdot_mask:
            annotation_masks.append(blackdots)
        with patch.object(uhdog, 'load_images_to_3d_array', return_value=image), \
                patch.object(uhdog, 'load_unet_images_to_3d_array', return_value=np.zeros(image.shape)), \
                patch.object(uhdog, 'load_annotations_to_3d_array',
                             side_effect=annotation_masks), \
                patch.object(uhdog, 'dog_search', return_value=(None, labels)), \
                patch.object(uhdog, 'inten_extraction', return_value=(np.full(3, 0.2), None)), \
                patch.object(uhdog, 'get_all_vol', return_value=volumes):
            uhdog.run_uhdog(config)
        results_dir = os.path.join(folder, 'test_results')
        with open(os.path.join(results_dir, 'results.json')) as f:
            results = json.load(f)
        self.assertFalse(os.path.exists(os.path.join(results_dir, 'glomerular_contrasts.csv')))
        with open(os.path.join(results_dir, 'glomerular_volumes.csv'), newline='') as f:
            volume_rows = list(csv.DictReader(f))
        return results, volume_rows

    def test_adjusted_kidney_contrast_and_volume_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            results, volumes = self.run_analysis(folder)
            self.assertEqual(results['n_glom'], 3)
            self.assertEqual(results['n_glom_contrast'], 2)
            self.assertEqual(results['n_glom_contrast_excluded'], 1)
            self.assertAlmostEqual(results['mean_glomerular_contrast'], 1.95)
            self.assertEqual(results['contrast_method'], 'cortex_background')
            self.assertEqual(results['glomerular_intensity_method'], 'minimum_voxel')
            self.assertAlmostEqual(results['mean_glomerular_intensity'], 1.0 / 6.0)
            self.assertEqual(results['glomerular_intensity_voxel_count'], 3)
            self.assertAlmostEqual(results['mean_cortex_background_intensity'], 0.6)
            self.assertEqual(results['cortex_background_voxel_count'], 1328)
            self.assertEqual([row['glom_id'] for row in volumes], ['1'])
            self.assertTrue(os.path.isfile(os.path.join(folder, 'test_results', 'vs_histogram.png')))

    def test_minimum_voxel_is_used_in_saved_kidney_average(self):
        with tempfile.TemporaryDirectory() as folder:
            results, volumes = self.run_analysis(folder, multi_voxel_glomerulus=True)
            self.assertEqual(results['n_glom'], 3)
            self.assertAlmostEqual(results['mean_glomerular_contrast'], 3.45)
            self.assertEqual(results['cortex_background_voxel_count'], 1327)
            self.assertEqual([row['glom_id'] for row in volumes], ['1'])

    def test_no_eligible_cortex_background_writes_null_mean(self):
        with tempfile.TemporaryDirectory() as folder:
            results, _ = self.run_analysis(folder, no_eligible_background=True)
            self.assertEqual(results['n_glom_contrast'], 0)
            self.assertEqual(results['n_glom_contrast_excluded'], 3)
            self.assertIsNone(results['mean_glomerular_contrast'])
            self.assertEqual(results['cortex_background_voxel_count'], 0)
            self.assertIsNone(results['mean_cortex_background_intensity'])

    def test_cortex_excludes_medulla_artifacts_and_outside_kidney(self):
        with tempfile.TemporaryDirectory() as folder:
            results, _ = self.run_analysis(folder, mask_exclusions=True, use_blackdot_mask=True)
            self.assertEqual(results['cortex_background_voxel_count'], 1)
            self.assertAlmostEqual(results['mean_cortex_background_intensity'], 0.6)
            self.assertAlmostEqual(results['mean_glomerular_contrast'], 1.95)

    def test_empty_results_write_volume_header_and_null_mean(self):
        with tempfile.TemporaryDirectory() as folder:
            results, volumes = self.run_analysis(folder, empty=True)
            self.assertEqual(results['n_glom'], 0)
            self.assertEqual(results['n_glom_contrast'], 0)
            self.assertEqual(results['n_glom_contrast_excluded'], 0)
            self.assertIsNone(results['mean_glomerular_contrast'])
            self.assertEqual(volumes, [])


if __name__ == '__main__':
    unittest.main()
