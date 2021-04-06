import cv2
import numpy as np
from math import ceil
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import namedtuple


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

    def get_by_id(self, coco_id):
        for name, coco_item in asdict(self).items():
            if coco_id == coco_item.id:
                return coco_item


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

    @property
    def model(self):
        if self._model is None:
            nn_weights_path = Path('yolov4', 'yolov4.weights')
            nn_cfg_path = Path('yolov4', 'yolov4.cfg')
            net = cv2.dnn.readNet(str(nn_weights_path), str(nn_cfg_path))
            model = cv2.dnn_DetectionModel(net)
            model.setInputParams(size=(512, 512), scale=1/255)
            self._model = model
        return self._model

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
