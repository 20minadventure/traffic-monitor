import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_movie(path):
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            print("Error?")
            break
    cap.release()

def draw_boxes(img, boxes, class_ids):
    bbox = img.copy()
    for (x, y, w, h), id in zip(boxes, class_ids):
        start_point = (x, y)
        end_point = (x + w, y + h)
        if id == 2:
            cv2.rectangle(bbox, start_point, end_point, color=(0, 0, 255), thickness=2)
        elif id == 7:
            cv2.rectangle(bbox, start_point, end_point, color=(255, 0, 0), thickness=2)
        elif id == 5:
            cv2.rectangle(bbox, start_point, end_point, color=(0, 255, 0), thickness=2)

    return bbox

    # x, y = leaf_side.coord
    # font['color'] = (0, 0, 180)
    # for jajo, score in [(b, s) for b, s in leaf_side.jaja['box_score'] if s > score]:
    #     start_point = jajo[0] - x, jajo[1] - y
    #     end_point = jajo[2] - x, jajo[3] - y
    #     text_point = start_point[0], start_point[1] - 3
    #     cv2.rectangle(bbox, start_point, end_point, color=(0, 0, 255), thickness=2)
    #     bbox = cv2.putText(bbox, f"{score:.2}", text_point, **font)


def make_movie(path, imgs, shape):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 15, shape)
    for img in imgs:
        out.write(img)
    out.release()


def show(imgs, height='600px', background='k', title=None):
    if title is not None:
        print(plt.fignum_exists(title))
        # TODO zrobić, że jak to będzie tru to wtedy zrobić nowe oknot bo wpp nowe się nei pojawiają jakos....
    if isinstance(imgs, list):
        if isinstance(imgs[0], list):
            lt = 'many'
            rows, cols = len(imgs), len(imgs[0])
        else:
            lt = 'single_row'
            rows, cols = 1, len(imgs)
            if len(imgs) == 1:
#                 imgs = imgs[0]
                lt = 'img'
    else:
        lt = 'img'
        rows, cols = 1, 1
    fig, axs = plt.subplots(rows, cols, num=title, figsize=(30,35))

    for i, (ax, img) in enumerate(zip(ravel_list([axs]), ravel_list([imgs], depth_type=np.ndarray))):
        if img is not None:
            ax.imshow(img)
        ax.axis('off')

    fig.set_facecolor(background)
    # fig.canvas.layout.height = height
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)


def ravel_list(list_like, depth=10, depth_type=None):
    if depth_type is None:
        depth_type = ()
    if depth > 0 and not isinstance(list_like, depth_type):
        for item in list_like:
            try:
                for i in ravel_list(item, depth=depth-1, depth_type=depth_type):
                    yield i
            except TypeError:
                yield item
    else:
        yield list_like


def rgb(img, r=1):
    if r != 1:
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None, fx=r, fy=r)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



# Convert layers_result to bbox, confs and classes
def get_final_predictions(outputs, img, threshold, nms_threshold):
    height, width = img.shape[0], img.shape[1]
    boxes, confs, class_ids = [], [], []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > threshold:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)

    merge_boxes_ids = cv2.dnn.NMSBoxes(boxes, confs, threshold, nms_threshold)

    # Filter only the boxes left after nms
    boxes = [boxes[int(i)] for i in merge_boxes_ids]
    confs = [confs[int(i)] for i in merge_boxes_ids]
    class_ids = [class_ids[int(i)] for i in merge_boxes_ids]
    return boxes, confs, class_ids

def patch_generator(image, patch_size=512, padding=0):
    width = max(image.shape[1], patch_size)
    height = max(image.shape[0], patch_size)

    p = np.ceil((width  - padding) / (patch_size - padding)).astype(int) # patche w x
    q = np.ceil((height  - padding) / (patch_size - padding)).astype(int)  # patche wzdłuż y

    for i in range(q):
        for j in range(p):
            xn = j * (patch_size - padding)
            yn = i * (patch_size - padding)

            # zwiększ nakładanie się patchy, gdy brakuje miejsca
            xn -= max((xn + patch_size) - width, 0)
            yn -= max((yn + patch_size) - height, 0)

            patch = image[
                yn : yn + patch_size,
                xn : xn + patch_size
            ]
            yield patch, (xn, yn)
