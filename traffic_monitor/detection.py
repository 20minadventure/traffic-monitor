import cv2
import numpy as np
from math import ceil
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import namedtuple
from tqdm import tqdm
import json
import pickle
import bz2

CocoItem = namedtuple(
    'CocoItem',
    ['name', 'id', 'color'],
    defaults=((0, 0, 0), )
)


@dataclass(init=False, frozen=True)
class Coco:
    PERSON: CocoItem = CocoItem('person', 0)
    BICYCLE: CocoItem = CocoItem('bicycle', 1)
    CAR: CocoItem = CocoItem('car', 2, (0, 0, 255))
    MOTORBIKE: CocoItem = CocoItem('motorbike', 3)
    AEROPLANE: CocoItem = CocoItem('aeroplane', 4)
    BUS: CocoItem = CocoItem('bus', 5, (0, 255, 0))
    TRAIN: CocoItem = CocoItem('train', 6)
    TRUCK: CocoItem = CocoItem('truck', 7, (255, 0, 0))
    BOAT: CocoItem = CocoItem('boat', 8)
    TRAFFIC_LIGHT: CocoItem = CocoItem('traffic light', 9)
    FIRE_HYDRANT: CocoItem = CocoItem('fire hydrant', 10)
    STOP_SIGN: CocoItem = CocoItem('stop sign', 11)
    OTHER: CocoItem = CocoItem('other', -1, (255, 255, 255))

    def __init__(self):
        by_id_dict = {}
        for name, coco_item in asdict(self).items():
            by_id_dict[coco_item.id] = coco_item
        object.__setattr__(self, 'by_id_dict', by_id_dict)

    def get_by_id(self, coco_id):
        return self.by_id_dict.get(coco_id, self.OTHER)


def patch_generator(image, patch_size, patches=None, min_padding=0,
        start_point=(0, 0), end_point=None):
    if end_point is None:
        end_point = image.shape[1], image.shape[0]
    width, height = end_point[0] - start_point[0], end_point[1] - start_point[1]
    if min(width, height) < patch_size or min_padding >= patch_size:
        return

    if patches is None:
        p = np.ceil(
            (width - min_padding) / (patch_size - min_padding)
        ).astype(int)
        q = np.ceil(
            (height - min_padding) / (patch_size - min_padding)
        ).astype(int)
        min_padding_x = min_padding_y = min_padding
    else:
        q, p = patches
        if q * patch_size < height or p * patch_size < width:
            return
        min_padding_x, min_padding_y = (0, 0)
        if p != 1:
            min_padding_x = int((p * patch_size - width) / (p - 1))
        if q != 1:
            min_padding_y = int((q * patch_size - height) / (q - 1))

    x0, y0 = start_point
    for i in range(q):
        for j in range(p):
            xn = x0 + j * (patch_size - min_padding_x)
            yn = y0 + i * (patch_size - min_padding_y)

            xn -= max((xn + patch_size - x0) - width, 0)
            yn -= max((yn + patch_size - y0) - height, 0)

            patch = image[
                yn : yn + patch_size,
                xn : xn + patch_size
            ]
            yield patch, (xn, yn)

            
class TrafficDetector:
    def __init__(self, path, count=None, step=1):
        self.path = path
        self._model = None
        self.count = count
        self.step = step
        self._fps = None
        self._frame_count = None
        self.classes_ids = []
        self.scores = []
        self.boxes = []

        self.CONFIDENCE_THRESHOLD = 0.3
        self.NMS_THRESHOLD = 0.6

    @property
    def model(self):
        if self._model is None:
            nn_weights_path = Path('yolov4', 'yolov4.weights')
            nn_cfg_path = Path('yolov4', 'yolov4.cfg')
            net = cv2.dnn.readNet(str(nn_weights_path), str(nn_cfg_path))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            model = cv2.dnn_DetectionModel(net)
            model.setInputParams(size=(512, 512), scale=1/255)
            self._model = model
        return self._model

    @property
    def fps(self):
        if self._fps is None:
            clip = cv2.VideoCapture(str(self.path))
            success, _ = clip.read()
            while success:
                success, _ = clip.read()
                msec = clip.get(cv2.CAP_PROP_POS_MSEC)
                if round(msec) >= 1000:
                    frame_nr = int(clip.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    self._fps = frame_nr
                    break
            clip.release()
        return self._fps

    @property
    def frame_count(self):
        if self._frame_count is None and self.fps is not None:
            clip = cv2.VideoCapture(str(self.path))
            buggy_frame_count = clip.get(cv2.CAP_PROP_FRAME_COUNT)
            buggy_fps = clip.get(cv2.CAP_PROP_FPS)
            clip.release()
            self._frame_count = round(buggy_frame_count / buggy_fps * self.fps)
        return self._frame_count


    def iter_clip_frames(self):
        if not self.path.is_file():
            raise FileNotFoundError
        clip = cv2.VideoCapture(str(self.path))
        frame_nr = 0
        success, frame = clip.read()
        while success:
            if self.count is None or frame_nr < self.count:
                if frame_nr % self.step == 0:
                    yield frame_nr, frame
            else:
                break
            frame_nr = int(clip.get(cv2.CAP_PROP_POS_FRAMES))
            success, frame = clip.read()
        clip.release()

    def detect_vehicles(self):
        frames = self.iter_clip_frames()
        n = self.frame_count if self.count is None else self.count
        n = ceil(n / self.step)
        for frame, img in tqdm(frames, total=n):
            ids = np.zeros((0, 1), dtype=np.int32)
            boxes = np.zeros((0, 4), dtype=np.int32)
            scores = np.zeros((0, 1), dtype=np.float32)
            empty_prediction = (ids, scores, boxes)
            for im, coord in patch_generator(img, 512, start_point=(0, 208)):
                prediction = self.model.detect(
                    im,
                    self.CONFIDENCE_THRESHOLD,
                    self.NMS_THRESHOLD
                )
                if len(prediction[0]) == 0:
                    prediction = empty_prediction
                shift = coord[0], coord[1], 0, 0
                ids = np.concatenate([ids, prediction[0]])
                scores = np.concatenate([scores, prediction[1]])
                boxes = np.concatenate([boxes, prediction[2] + np.array(shift)])
            self.classes_ids.append(ids)
            self.boxes.append(boxes)
            self.scores.append(scores)

    def draw_boxes(self):
        coco = Coco()
        frames = self.iter_clip_frames()
        results_iter = zip(frames, self.classes_ids, self.boxes, self.scores)
        for (fr, img), fr_classes, fr_boxes, fr_scores in results_iter:
            instances_iter = zip(fr_boxes, fr_classes, fr_scores)
            for (x, y, w, h), class_id, score in instances_iter:
                pt0, pt1 = (x, y), (x + w, y + h)
                coco_item = coco.get_by_id(class_id.item())
                if coco_item.name in ['car', 'truck', 'bus', 'train']:
                    pb0, pb1 = (x + 1, y + int(h * score)), (x + 4, y + h)
                    cv2.rectangle(img, pt0, pt1, coco_item.color, 1)
                    cv2.rectangle(img, pb0, pb1, coco_item.color, -1)
            yield img


    def dump_detections(self, path=Path('example_dump'), format='pbz2'):

        if format == "json":
            data = {
                'boxes': list(map(lambda x: x.tolist(), self.boxes)),
                'scores': list(map(lambda x: x.tolist(), self.scores)),
                'classes_ids': list(map(lambda x: x.tolist(), self.classes_ids))
                }

            with open(str(path) + '.json', 'w') as file:
                json.dump(data, file)

        elif format == "pbz2":
            data = {
                'boxes': self.boxes,
                'scores': self.scores,
                'classes_ids': self.classes_ids
                }

            with bz2.BZ2File(str(path) + '.pbz2', 'w') as file:
                pickle.dump(data, file)

        else:
            raise ValueError


    def load_detections(self, path):
        ext = ''.join(Path(path).suffix)

        if ext == ".json":
            with open(path, 'r') as file:
                data = json.load(file)
            
            self.boxes = list(map(np.array, data['boxes']))
            self.scores = list(map(np.array, data['scores']))
            self.classes_ids = list(map(np.array, data['classes_ids']))

        elif ext == ".pbz2":
            with bz2.BZ2File(path, 'rb') as file:
                data = pickle.load(file)

            self.boxes = data['boxes']
            self.scores = data['scores']
            self.classes_ids = data['classes_ids'] 

        else:
            raise ValueError
