import cv2
import numpy as np
import pytest
from copy import deepcopy
from pathlib import Path
from copy import deepcopy

from traffic_monitor.detection import TrafficDetector
from traffic_monitor.detection import Coco, CocoItem


@pytest.fixture
def dummy_clip_path(tmp_path):
    return tmp_path / 'dummy_clip.mp4'

@pytest.fixture
def example_dump_path(tmp_path):
    return tmp_path / 'dump'

@pytest.fixture
def example_class_names_path(tmp_path):
    return tmp_path / 'class_names.mp4'  


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


def test_detection_on_black_clip(black_clip_path):
    td = TrafficDetector(black_clip_path, count=1)

    td.detect_vehicles()

    assert len(td.classes_ids) == 1
    assert len(td.scores) == 1
    assert len(td.boxes) == 1
    assert len(td.classes_ids[0]) == 0

def test_predictions_saving_to_json(real_clip_path, example_dump_path):
    td = TrafficDetector(real_clip_path, count=3)
    td.detect_vehicles()

    boxes = deepcopy(td.boxes)
    classes_ids = deepcopy(td.classes_ids)
    scores = deepcopy(td.scores)

    td.dump_detections(example_dump_path, format='json')
    td.load_detections(str(example_dump_path) + '.json')

    assert all([isinstance(box, np.ndarray)] for box in td.boxes)
    assert len(boxes) == len(td.boxes)
    assert all([np.array_equal(old_box, new_box) for old_box, new_box in zip(boxes, td.boxes)])

    assert all([isinstance(class_id, np.ndarray)] for class_id in td.classes_ids)
    assert len(classes_ids) == len(td.classes_ids)
    assert all(
        [np.array_equal(old_class, new_class) for old_class, new_class in zip(classes_ids, td.classes_ids)]
        )  

    assert all([isinstance(score, np.ndarray)] for score in td.scores)
    assert len(scores) == len(td.scores)
    assert all(
        [np.array_equal(old_score, new_score) for old_score, new_score in zip(scores, td.scores)]
        )   

def test_predictions_saving_to_pickle(real_clip_path, example_dump_path):
    td = TrafficDetector(real_clip_path, count=3)
    td.detect_vehicles()

    boxes = deepcopy(td.boxes)
    classes_ids = deepcopy(td.classes_ids)
    scores = deepcopy(td.scores)

    td.dump_detections(example_dump_path, format='pbz2')
    td.load_detections(str(example_dump_path) + '.pbz2')

    assert all([isinstance(box, np.ndarray)] for box in td.boxes)
    assert len(boxes) == len(td.boxes)
    assert all([np.array_equal(old_box, new_box) for old_box, new_box in zip(boxes, td.boxes)])

    assert all([isinstance(class_id, np.ndarray)] for class_id in td.classes_ids)
    assert len(classes_ids) == len(td.classes_ids)
    assert all(
        [np.array_equal(old_class, new_class) for old_class, new_class in zip(classes_ids, td.classes_ids)]
        )  

    assert all([isinstance(score, np.ndarray)] for score in td.scores)
    assert len(scores) == len(td.scores)
    assert all(
        [np.array_equal(old_score, new_score) for old_score, new_score in zip(scores, td.scores)]
        )    

