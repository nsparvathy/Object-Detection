import cv2
import numpy as np
import urllib.request

url = "http://192.168.1.7/cam-hi.jpg"

cap = cv2.VideoCapture(url)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile = "coco.names"
classNames = []
with open(classesfile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")
# print(classNames)

modelConfig = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    found_chair = False
    found_book = False
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        className = classNames[classIds[i]]

        # Draw bounding box based on class
        color = (255, 0, 255)  # Default color
        if className == "book":
            color = (255, 0, 255)  # Purple for books
            found_book = True
        elif className == "chair":
            color = (0, 255, 255)  # Yellow for chairs
            found_chair = True

        cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            im,
            f"{className.upper()} {int(confs[i]*100)}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)
    sucess, img = cap.read()
    blob = cv2.dnn.blobFromImage(im, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layernames = net.getLayerNames()
    # print(layernames)
    outputNames = [layernames[int(i) - 1] for i in net.getUnconnectedOutLayers()]

    # print(net.getUnconnectedOutLayers())
    outputs = net.forward(outputNames)
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])
    findObject(outputs, im)
    cv2.imshow("IMage", im)
    cv2.waitKey(1)
