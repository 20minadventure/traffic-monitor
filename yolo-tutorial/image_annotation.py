
# %%
import cv2

preview = True

# %%
image = cv2.imread('images/snapshot2.jpg')

# %%
net = cv2.dnn.readNet("yolov4/yolov4.weights", "yolov4/yolov4.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(512, 512), scale=1/255)

# %%
class_names = []
with open("yolov4/coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# %%
def annotate(image, classes, scores, boxes, show=True):
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(image, box, color, 2)
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image
   


# %%
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
print(boxes)
# %%
annotated_image = annotate(image, classes,scores, boxes)

if preview:
    cv2.imshow("output", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.imwrite('images/snapshot2_yolov4.jpg', annotated_image)


