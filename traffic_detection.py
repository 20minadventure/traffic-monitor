import cv2
import numpy as np
from pathlib import Path


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
    def __init__(self, path):
        self.path = path
        self._model = None

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

    def iter_clip_frames(self, count=None, step=1):
        clip = cv2.VideoCapture(str(self.path))
        success, frame = clip.read()
        frame_nr = 0
        while success:
            frame_nr += 1
            if count is None or frame_nr <= count:
                if frame_nr % step == 0:
                    yield frame
            else:
                break
            success, frame = clip.read()
        clip.release()
