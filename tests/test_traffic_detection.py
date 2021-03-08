import cv2
import pytest

from traffic_detection import TrafficDetector


@pytest.fixture
def dummy_clip_path(tmp_path):
    return tmp_path / 'dummy_clip.mp4'


def test_model_loading(dummy_clip_path):
    td = TrafficDetector(dummy_clip_path)

    assert isinstance(td.model, cv2.dnn_DetectionModel)
