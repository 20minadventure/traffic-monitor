import cv2
import pytest
from pathlib import Path

from traffic_monitor.detection import TrafficDetector


@pytest.fixture
def dummy_clip_path(tmp_path):
    return tmp_path / 'dummy_clip.mp4'


@pytest.fixture
def real_clip_path():
    return Path('stream_data', 'streamlink_20210303_145004.mp4')


def test_model_loading(dummy_clip_path):
    td = TrafficDetector(dummy_clip_path)

    assert isinstance(td.model, cv2.dnn_DetectionModel)


def test_rasing_error_when_clip_doesnt_exist(dummy_clip_path):
    td = TrafficDetector(dummy_clip_path)

    with pytest.raises(FileNotFoundError):
        next(td.iter_clip_frames())


@pytest.mark.parametrize(
   "count, step",
   [(None, 100), (10, 5), (0, 1)]
)
def test_iter_clip_frames(real_clip_path, count, step):
    td = TrafficDetector(real_clip_path, count=count, step=step)
    if count is None:
        count = 331

    frames = [frame for frame, _ in td.iter_clip_frames()]

    assert frames == list(range(0, count, step))

