import cv2
import numpy as np
from model import FacialExpression

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpression("model.json", "model_expr_weight.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

vid = cv2.VideoCapture('facial_expression.mp4')
pred = 'None'
vid.set(3, 1280)
vid.set(4, 720)
  
while(vid.isOpened()):
    ret, frame = vid.read()
    if np.shape(frame) == ():
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facec.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        fc = gray[y-60:y+h+20, x:x+w]
        try:
            det_img = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(det_img[np.newaxis, :, :, np.newaxis])
        except Exception as e:
            pass
        
        cv2.putText(frame, pred, (x, y-10), font, 0.55, (255, 255, 0), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty('frame', 4) < 1:
        break

vid.release()
cv2.destroyAllWindows()