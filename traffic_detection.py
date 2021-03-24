import cv2
import numpy as np
import json
from pathlib import Path


class TrafficDetector:
    def __init__(self, path):
        self.path = path

        self._model = None
        self.confs = []
        self.boxes = []
        self.class_names = []

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
        clip = cv2.VideoCapture(str(path))
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

    def detect_vehicles(self, frames):
        CONFIDENCE_THRESHOLD = 0.3
        NMS_THRESHOLD = 0.4
        for img in frames:
            boxes = np.zeros((0, 4), dtype=np.int32)
            ids = np.zeros((0, 1), dtype=np.int32)
            # for im, coord in patch_generator(img2[208:, ...], padding=128):
            for im, coord in [(img[-512:, :512, :], (0, 208))]:
                prediction = self.model.detect(im, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
                shift = coord[0], coord[1], 0, 0
                boxes = np.concatenate([boxes, prediction[2] + np.array(shift)])
                ids = np.concatenate([ids, prediction[0]])

            self.boxes.append(boxes)
            self.class_names.append(ids)
    
    def dump_detections(self, boxes_path='boxes.json', class_names_path="class_names.json"):
        with open(boxes_path, 'w') as boxes_file:
            serializable = list(map(lambda x: x.tolist(), self.boxes))
            json.dump(serializable, boxes_file)
        
        with open(class_names_path, 'w') as class_names_file:
            serializable = list(map(lambda x: x.tolist(), self.class_names))
            json.dump(serializable, class_names_file)

    def load_detections(self, boxes_path='boxes.json', class_names_path="class_names.json"):
        with open(boxes_path, 'r') as boxes_file:
            raw = json.load(boxes_file)
            self.boxes = list(map(np.array, raw))

        with open(class_names_path, 'r') as class_names_file:
            raw = json.load(class_names_file)
            self.class_names = list(map(np.array, raw))     

    def draw_boxes(self, frames):
        for img, boxes, class_names in zip(frames, self.boxes, self.class_names):
            for (x, y, w, h), id in zip(boxes, class_names):
                start_point = (x, y)
                end_point = (x + w, y + h)
                if id == 2:
                    cv2.rectangle(img, start_point, end_point, color=(0, 0, 255), thickness=2)
                elif id == 7:
                    cv2.rectangle(img, start_point, end_point, color=(255, 0, 0), thickness=2)
                elif id == 5:
                    cv2.rectangle(img, start_point, end_point, color=(0, 255, 0), thickness=2)
            yield img



if __name__ == '__main__':
    from pathlib import Path
    from utils import show, rgb

    path = Path('stream_data', 'streamlink_20210304_165003.mp4')
    td = TrafficDetector(path)
    frames = td.iter_clip_frames(count=42, step=2)
    td.detect_vehicles(frames)
    td.dump_detections()

    frames = td.iter_clip_frames(count=42, step=2)
    boxes = td.draw_boxes(frames)

    pred_imgs = lambda x: rgb(next(boxes))
    import moviepy.editor as mpy
    clip = mpy.VideoClip(pred_imgs, duration=2)  # fps * duration < len(frames)
    clip.write_videofile("circle3.mp4", fps=10)
