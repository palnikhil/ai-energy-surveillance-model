import time
import cv2
from skimage import measure
import imutils
from imutils import contours
import numpy as np
from persondetection import DetectorAPI
from communicate import Send_Warning

video_capture = cv2.VideoCapture(0)
pdapi = DetectorAPI()
threshold = 0.5
x3 = 0
county3 = []
framex3 = []
max_acc3 = 0


def Detect_person(image):
  boxes, scores, classes, num = pdapi.processFrame(image)
  person = 0
  acc = 0
  for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > threshold:
                        box = boxes[i]
                        person += 1
  return person

def Detect_light():
  t=10
  prev = time.time()
  while t>=0:
    ret,frame = video_capture.read()
    frame = cv2.resize(frame,(800,600))
    p=Detect_person(frame)
    if p>0:
      break
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayFrame,(11,11),0)
    thresh = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh , None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    try:
     labels = measure.label(thresh, background=0)
     mask = np.zeros(thresh.shape, dtype="uint8")
     for label in np.unique(labels):
        if label == 0:
          continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > 300:
          mask = cv2.add(mask, labelMask)
     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     cnts = imutils.grab_contours(cnts)
     cnts = contours.sort_contours(cnts)[0]
     mins, secs = divmod(t, 60)
     timer = '{:02d}:{:02d}'.format(mins, secs)
     font = cv2.FONT_HERSHEY_SIMPLEX
     cv2.putText(frame, str(timer),
                        (400, 50), font,
                        2, (0, 255, 255),
                        1, cv2.LINE_AA)
     cur = time.time()
     if cur-prev >= 1:
      prev = cur
      t-=1
     for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)

        cv2.circle(frame, (int(cX), int(cY)), int(radius),
         (0, 0, 255), 3)
        cv2.putText(frame, "#{}".format(i + 1), (x, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        cv2.rectangle(frame, (0,0), (225,73), (245,117,16), -1)
        cv2.putText(frame, "Light Source Detected", 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(timer),
                        (400, 50), font,
                        2, (0, 255, 255),
                        1, cv2.LINE_AA)
        cv2.imshow("Energy Surveillance", frame)
    except:
     t = 10
     cv2.rectangle(frame, (0,0), (225,73), (245,117,16), -1)
     cv2.putText(frame, "No Light Source Detected", 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
     cv2.imshow("Energy Surveillance", frame)
    finally:
      if t == 0:
        Send_Warning()
        print("Restarting")
        t=10
    if cv2.waitKey(1) == 27:
        break


while True:
    _,frame = video_capture.read()
    image = cv2.resize(frame,(800,600))
    boxes, scores, classes, num = pdapi.processFrame(image)
    person = 0
    acc = 0
    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > threshold:
                        box = boxes[i]
                        person += 1
                        cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                        cv2.putText(image, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)  # (75,0,130),
                        acc += scores[i]
                        if (scores[i] > max_acc3):
                            max_acc3 = scores[i]
    if person > 0:
     cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
     cv2.putText(image, 'Persons', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
     cv2.putText(image, str(person), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
     cv2.imshow("Energy Surveillance", image)
    else:
      Detect_light()
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    county3.append(person)
    x3 += 1
    framex3.append(x3)
video_capture.release()
cv2.destroyAllWindows()