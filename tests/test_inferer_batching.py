"""CPU-only regression tests for video batching, without loading model weights.

Load the actual gaze_inferer class definition with lightweight video/eye-model
collaborators. NumPy buffers, process(), and _infer_batch() execute unchanged.
Run with: python -m unittest discover -s tests -v
"""
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


def load_inferer():
    path = Path(__file__).resolve().parents[1] / 'deepvog' / 'inferer.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == 'gaze_inferer']
    namespace = {'np': np, 'VideoManager': FakeVideoManager}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['gaze_inferer']


class FakeVideoManager:
    def __init__(self, **kwargs):
        self.output_video_flag = False
        self.records = []

    def write_results(self, **record):
        self.records.append(record)


class RecordingModel:
    def __init__(self):
        self.batches = []

    def predict(self, frames):
        self.batches.append(frames[:, 0, 0, 0].tolist())
        return np.zeros((len(frames), 1, 1, 2))


class BatchingTests(unittest.TestCase):
    def run_video(self, count, batch_size, mode='Infer', reported_count=None):
        cls = load_inferer()
        inferer = cls.__new__(cls)
        inferer.model = RecordingModel()
        inferer.flen = 1
        inferer.mm2px_scaling = 1
        inferer.infer_gaze_flag = False
        inferer.eyefitter = SimpleNamespace(
            unproject_single_observation=lambda prediction:
                (None, None, None, None, (None, None, None, None, None, None, None)),
            fit_projected_eye_centre=lambda **kwargs: None,
            estimate_eye_sphere=lambda: (None, None),
            eye_centre=np.zeros((3, 1)), aver_eye_radius=1)
        reader = SimpleNamespace(nextFrame=lambda: iter(range(1, count + 1)))
        inferer._get_video_info = lambda path: (
            'test', '.mp4', reader,
            (count if reported_count is None else reported_count, 240, 320, 3),
            True, 1)
        # A unique scalar fills every pixel in each real NumPy frame buffer.
        inferer._preprocess_image = lambda frame, shape_correct: frame
        with contextlib.redirect_stdout(io.StringIO()):
            inferer.process('test.mp4', mode, batch_size=batch_size)
        expected_batches = [list(range(start, min(start + batch_size, count + 1)))
                            for start in range(1, count + 1, batch_size)]
        self.assertEqual(inferer.model.batches, expected_batches)
        if mode == 'Infer':
            self.assertEqual([row['frame_id'] for row in inferer.vid_manager.records],
                             list(range(1, count + 1)))
        return inferer

    def test_empty_video(self):
        self.run_video(0, 4)

    def test_single_frame(self):
        self.run_video(1, 4)

    def test_video_shorter_than_batch(self):
        self.run_video(3, 4)

    def test_exactly_one_batch(self):
        self.run_video(4, 4)

    def test_one_frame_remainder(self):
        self.run_video(5, 4)

    def test_multiple_frame_remainder(self):
        self.run_video(7, 4)

    def test_exactly_two_batches(self):
        self.run_video(8, 4)

    def test_multiple_batches_and_remainder(self):
        self.run_video(11, 4)

    def test_batch_size_one(self):
        self.run_video(3, 1)

    def test_documented_batch_size_exact(self):
        self.run_video(32, 32)

    def test_documented_batch_size_remainder(self):
        self.run_video(35, 32)

    def test_fit_mode_processes_all_frames(self):
        for count in (1, 3, 4, 5, 7, 8, 11):
            with self.subTest(count=count):
                self.run_video(count, 4, mode='Fit')

    def test_decoded_frames_override_short_metadata_count(self):
        self.run_video(7, 4, reported_count=4)

    def test_decoded_frames_override_long_metadata_count(self):
        self.run_video(3, 4, reported_count=8)


if __name__ == '__main__':
    unittest.main()
