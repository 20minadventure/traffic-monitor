import cv2
import pytest
from pathlib import Path

from traffic_monitor.detection import TrafficDetector
from traffic_monitor.detection import Coco, CocoItem


@pytest.fixture
def dummy_clip_path(tmp_path):
    return tmp_path / 'dummy_clip.mp4'


@pytest.fixture
def black_clip_path():
    return Path('tests', 'data', 'black_clip.mp4')


@pytest.fixture
def real_clip_path():
    return Path('stream_data', 'streamlink_20210303_145004.mp4')


def test_model_loading(dummy_clip_path):
    td = TrafficDetector(dummy_clip_path)

    assert isinstance(td.model, cv2.dnn_DetectionModel)


def test_coco_items():
    coco = Coco()

    assert coco.CAR.name == 'car'
    assert coco.CAR.id == 2
    assert isinstance(coco.CAR.color, tuple)
    assert coco.STOP_SIGN == CocoItem('stop sign', 11)


def test_getting_coco_by_id():
    coco = Coco()
    other = coco.get_by_id(99)

    assert coco.CAR == coco.get_by_id(2)
    assert CocoItem('stop sign', 11) == coco.get_by_id(11)
    assert coco.OTHER == coco.get_by_id(99)


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


def test_fps_retrieving(real_clip_path, black_clip_path):
    real_td = TrafficDetector(real_clip_path)
    black_td = TrafficDetector(black_clip_path)

    assert real_td.fps == 30
    assert black_td.fps is None


def test_total_frame_count_retrieving(real_clip_path, black_clip_path):
    real_td = TrafficDetector(real_clip_path)
    black_td = TrafficDetector(black_clip_path)

    assert real_td.frame_count == 331
    assert black_td.frame_count is None


def test_detection(real_clip_path):
    td = TrafficDetector(real_clip_path, count=1)

    td.detect_vehicles()

    assert len(td.classes_ids) == 1
    assert len(td.scores) == 1
    assert len(td.boxes) == 1
    assert len(td.classes_ids[0])
