import cv2
import pytest

from traffic_detection import TrafficDetector


@pytest.fixture
def example_clip_path(tmp_path):
    return tmp_path / 'clip.mp4'


def test_dry_run_prediction(example_clip_path):
    td = TrafficDetector(example_clip_path)
    td.detect_vehicles([])

    assert hasattr(td, 'confs')
    assert hasattr(td, 'boxes')
    assert hasattr(td, 'class_names')


def test_model_loading(example_clip_path):
    td = TrafficDetector(example_clip_path)

    td.model is cv2.dnn_DetectionModel
