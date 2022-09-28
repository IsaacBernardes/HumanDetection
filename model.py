import cv2
import numpy as np

class Classifier:
    def __init__(self):
        self.model = cv2.HOGDescriptor()
        self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        try:
            bounding_box_cordinates, weights = self.model.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

            if len(bounding_box_cordinates) > 0:
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bounding_box_cordinates])
                bounding_box_cordinates = non_max_suppression_fast(rects, overlap_thresh=0.65)

            person = 0
            for x, y, w, h in bounding_box_cordinates:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'person {person+1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                person += 1

            return person, frame

        except:
            return 0, None


def non_max_suppression_fast(boxes, overlap_thresh):

    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")