import cv2
from pathlib import Path


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

    def iter_clip_frames(self):
        clip = cv2.VideoCapture(str(self.path))
        while True:
            success, frame = clip.read()
            if success:
                yield frame
            else:
                break
        clip.release()
