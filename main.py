import cv2
import matplotlib.pylab as plt
import numpy as np

thres = 0.65  # Threshold to detect object
webcam = False



def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def mask(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255

    height = img.shape[0]
    width = img.shape[1]

    region_of_interest_vertices = [
        (0, height * 0.75),
        (width * 0.55, height * 0.6),
        (width, height * 0.75)
    ]

    cv2.fillPoly(mask, vertices, match_mask_color)


def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (0, height * 0.75),
        (width * 0.55, height * 0.6),
        (width, height * 0.75)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 200, 300)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi / 60,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)

    image_with_line = drow_the_lines(image, lines)
    return image_with_line


def turn_the_model(image):
    return image


cap = cv2.VideoCapture(0)
cap.set(10, 70)

classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread('jelzolampa2.png')
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    # print(classIds, bbox)

    img2 = img

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classNames[classId - 1] != '.':
                cv2.rectangle(img, box, color=(128, 0, 128), thickness=4)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 128), 2)
            if classNames[classId - 1] == "stop sign":
                objWidth = box[2]
                objHeight = box[3]
                objArea = objWidth * objHeight
                objArea1 = objArea
                objArea2 = objArea1
                objArea3 = objArea2
                objArea4 = objArea3
                objArea5 = objArea4
                objArea6 = objArea5
                objArea7 = objArea6
                objArea8 = objArea7
                objArea9 = objArea8
                objArea10 = objArea9

                objAreaAverage = (objArea1 + objArea2 + objArea3 + objArea4 + objArea5 +
                                  objArea6 + objArea7 + objArea8 + objArea9 + objArea10) / 10

                Distance = (objArea - 213414) / -3560
                Distance2 = (500 / objArea) + 12
                print(Distance, Distance2)
                cv2.putText(img, str(round(Distance)), (box[0] + 10, box[1] + 60),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    #img = process(img2)
    img = grayscale(img)
    imS = cv2.resize(img, (800, 1000))

    cv2.imshow("Output", imS)
    cv2.waitKey(1)
