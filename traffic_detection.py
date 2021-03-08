import cv2


class TrafficDetector:
    def __init__(self, path):
        self.path = path
        self.confs = None
        self.boxes = None
        self.class_names = None

    def iter_clip_frames(self):
        clip = cv2.VideoCapture(str(self.path))
        clip = cv2.VideoCapture(str(path))
        while True:
            success, frame = clip.read()
            if success:
                yield frame
            else:
                break
        clip.release()
