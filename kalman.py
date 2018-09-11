import cv2
from vector import distance
import numpy as np
import time
from scipy import ndimage
from keras.models import load_model

cc=-1
def nextId():
    global cc
    cc += 1
    return cc

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if (mdist < r):
            retVal.append(obj)
    return retVal



def processVideo(path):
    #otvara video(frejm videa)
    cap = cv2.VideoCapture(path)
    ret, firstFrame = cap.read()
    blueLine = findBlueLine(firstFrame)
    greenLine = findGreenLine(firstFrame)
    #Kolor filter
    kernel = np.ones((2,2),np.uint8)
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('images/output-rezB.avi',fourcc, 20.0, (640,480))
    elements = []
    t =0
    counter = 0
    times = []
    while 1:
        start_time = time.time()
        ret, img = cap.read()
        if ret == False:
            break
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)
        img0 = 1.0 * mask
        img0 = cv2.dilate(img0, kernel)  # cv2.erode(img0,kernel)
        img0 = cv2.dilate(img0, kernel)
        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)

                cv2.putText(img, str(val) + ','+str(id) ,
                            (el['center'][0] + 10, el['center'][1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 255)


        elapsed_time = time.time() - start_time
        times.append(elapsed_time * 1000)
        cv2.putText(img, 'Counter: ' + str(counter), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)
        t += 1
        # cv2.circle(img, blueLine[1], 3, (0, 0, 255), -1)
        # cv2.circle(img, blueLine[0], 3, (0, 0, 255), -1)
        # cv2.circle(img, greenLine[1], 3, (0, 0, 255), -1)
        # cv2.circle(img, greenLine[0], 3, (0, 0, 255), -1)
        #showPicture(img, out)
        #time.sleep(0.6)



def findGreenLine(paramFrame):
    initialFrame = paramFrame.copy()
    #ignorisanje plave boje, da ostane samo ova druga linija(zelena linija)
    initialFrame[:, :, 0] = 0
    gray = cv2.cvtColor(initialFrame, cv2.COLOR_BGR2GRAY)
    return detectLine(gray)

def findBlueLine(paramFrame):
    #ignorisanje zelene boje
    initialFrame = paramFrame.copy()
    initialFrame[:, :, 1] = 0
    gray = cv2.cvtColor(initialFrame, cv2.COLOR_BGR2GRAY)
    return detectLine(gray)


def detectLine(gray):
    ret, t = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(t, 1, np.pi / 90, 100, 100, 10)
    xSouth = min(lines[:, 0, 0])
    # y donja(pocetna kordinata)
    ySouth = max(lines[:, 0, 1])
    # x gornja kordinata
    xNorth = max(lines[:, 0, 2])
    # y gornja kordinata
    yNorth = min(lines[:, 0, 3])
    return [(xSouth, ySouth), (xNorth, yNorth)]

model = load_model('my_model.h5');
def executeNeuralNetwork(region):
    return model.predict_classes(region.reshape(1, 1, 28, 28).astype('float32'))

def predict(img):
    val = executeNeuralNetwork(preprocessingImage(img))
    return val

def preprocessingImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if(not contours):
        return -1
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    rect = thresh1[y:y + h, x:x + w]
    return cv2.resize(rect, (28, 28), interpolation=cv2.INTER_LINEAR)


def showPicture(img,out):
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    out.write(img)
