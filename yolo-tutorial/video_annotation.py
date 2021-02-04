import cv2
import numpy as np
from sort import Sort, convert_x_to_bbox

input_path = 'videos/streamlink_capture_part.mp4'
nn_cfg_path = 'yolov4/yolov4.cfg'
nn_weights_path = 'yolov4/yolov4.weights'
labels_path = 'yolov4/coco.names'

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4

class_names = []
with open(labels_path, "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet(nn_weights_path, nn_cfg_path)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(512, 512), scale=1/255)

def annotate(image, classes, scores, boxes, ids):
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    for (classid, score, box, id) in zip(classes, scores, boxes, ids):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f (%f)" % (class_names[classid[0]], score, int(id))
        cv2.rectangle(image, box, color, 2)
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

vs = cv2.VideoCapture(input_path)

while True:

    grabbed, frame = vs.read()

    if not grabbed:
        break
    
    tracker = Sort()

    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    bboxes = np.empty_like(boxes)
    bboxes[:,:2] = boxes[:,:2]
    bboxes[:,2:] = boxes[:,:2] + boxes[:,2:]

    print(bboxes[0,:])
    print(boxes[0,:])
    print("")

    ids = tracker.update(np.concatenate((bboxes, scores), axis=1))[:,-1]

    annotated_frame = annotate(frame, classes, scores, boxes, ids)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()