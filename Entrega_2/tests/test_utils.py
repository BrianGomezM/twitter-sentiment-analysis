import os
import sys
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

# Ensure src is on the path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np

# Import module under test
import utils


class DummyModel:
    def __init__(self, y_pred_proba):
        self._y_pred_proba = np.asarray(y_pred_proba)

    def predict(self, X, verbose=0):  # noqa: N803 (match signature used)
        # Ignore X and return the configured predictions
        return self._y_pred_proba


class DummyHistory:
    def __init__(self, history_dict):
        self.history = history_dict


class DummyEncoder:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)


class TestEvaluateModel(unittest.TestCase):
    @patch('utils.plt.show')
    def test_returns_report_with_expected_content(self, _mock_show):
        # three classes, balanced
        classes = ['negative', 'neutral', 'positive']
        encoder = DummyEncoder(classes)
        # y_test one-hot: [0,1,2,0,1,2]
        y_true_idx = np.array([0, 1, 2, 0, 1, 2])
        y_test = np.eye(3)[y_true_idx]
        # predictions are correct for all
        y_pred_proba = np.eye(3)[y_true_idx]
        model = DummyModel(y_pred_proba)

        report = utils.evaluate_model(model, X_test=np.zeros((6, 5)), y_test=y_test,
                                      encoder=encoder, model_name='UT_MODEL')

        self.assertIsInstance(report, str)
        # Header and class names should be in the report
        self.assertIn('precision', report)
        for cls in classes:
            self.assertIn(cls, report)
        # Perfect metrics should appear (formatted with 4 decimals)
        self.assertIn('1.0000', report)

    @patch('utils.plt.show')
    def test_saves_confusion_and_report_when_save_path(self, _mock_show):
        classes = ['negative', 'neutral', 'positive']
        encoder = DummyEncoder(classes)
        y_true_idx = np.array([0, 1, 2, 0, 1, 2])
        y_test = np.eye(3)[y_true_idx]
        # Introduce some errors
        y_pred_idx = np.array([0, 2, 2, 1, 1, 0])
        y_pred_proba = np.eye(3)[y_pred_idx]
        model = DummyModel(y_pred_proba)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = 'UT_MODEL_IO'
            utils.evaluate_model(model, X_test=np.zeros((6, 5)), y_test=y_test,
                                 encoder=encoder, model_name=model_name, save_path=tmpdir)
            # Files should exist
            confusion_path = os.path.join(tmpdir, f'{model_name}_confusion.png')
            report_path = os.path.join(tmpdir, f'{model_name}_report.txt')
            self.assertTrue(os.path.exists(confusion_path))
            self.assertTrue(os.path.exists(report_path))
            # Report file should contain the extra metrics block
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn('Métricas Resumen', content)

    @patch('utils.plt.show')
    def test_prints_per_class_accuracy(self, _mock_show):
        classes = ['negative', 'neutral', 'positive']
        encoder = DummyEncoder(classes)
        y_true_idx = np.array([0, 1, 2, 0, 1, 2])
        y_test = np.eye(3)[y_true_idx]
        y_pred_idx = np.array([0, 1, 2, 0, 0, 2])  # one error for class 'neutral'
        y_pred_proba = np.eye(3)[y_pred_idx]
        model = DummyModel(y_pred_proba)

        buf = io.StringIO()
        with redirect_stdout(buf):
            utils.evaluate_model(model, X_test=np.zeros((6, 5)), y_test=y_test,
                                 encoder=encoder, model_name='UT_MODEL')
        out = buf.getvalue()
        # Should include accuracy per class lines
        for cls in classes:
            self.assertIn(cls, out)
            self.assertRegex(out, rf"\u2022\s+{cls}: ")  # bullet dot and class label


class TestPlotEnhancedResults(unittest.TestCase):
    @patch('utils.plt.show')
    def test_saves_analysis_and_metrics_without_lr(self, _mock_show):
        # Build a small history without lr
        history = DummyHistory({
            'loss': [1.0, 0.8, 0.6, 0.5],
            'val_loss': [1.1, 0.9, 0.7, 0.55],
            'accuracy': [0.5, 0.6, 0.7, 0.8],
            'val_accuracy': [0.45, 0.55, 0.65, 0.75],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = 'UT_PLOT_NO_LR'
            utils.plot_enhanced_results(history, model_name, save_path=tmpdir)
            analysis_path = os.path.join(tmpdir, f'{model_name}_analysis.png')
            metrics_path = os.path.join(tmpdir, f'{model_name}_metrics.json')
            self.assertTrue(os.path.exists(analysis_path))
            self.assertTrue(os.path.exists(metrics_path))
            # Validate metrics content
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            self.assertEqual(metrics['epochs_trained'], 4)
            self.assertAlmostEqual(metrics['final_train_loss'], 0.5, places=6)
            self.assertEqual(metrics['best_epoch'], int(np.argmin(history.history['val_loss']) + 1))

    @patch('utils.plt.show')
    def test_saves_analysis_and_metrics_with_lr(self, _mock_show):
        # Build a small history with lr schedule
        history = DummyHistory({
            'loss': [0.9, 0.7, 0.65, 0.6],
            'val_loss': [1.0, 0.8, 0.7, 0.62],
            'accuracy': [0.55, 0.65, 0.72, 0.78],
            'val_accuracy': [0.50, 0.60, 0.68, 0.74],
            'lr': [1e-3, 5e-4, 2e-4, 1e-4],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = 'UT_PLOT_WITH_LR'
            utils.plot_enhanced_results(history, model_name, save_path=tmpdir)
            analysis_path = os.path.join(tmpdir, f'{model_name}_analysis.png')
            metrics_path = os.path.join(tmpdir, f'{model_name}_metrics.json')
            self.assertTrue(os.path.exists(analysis_path))
            self.assertTrue(os.path.exists(metrics_path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
