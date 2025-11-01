import cv2 as cv
import numpy as np

lowerLim=np.array([15,150,20])
upperLim=np.array([35,255,255])

webcam=cv.VideoCapture(0)

while True:
    ret,frame=webcam.read()

    img=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    mask=cv.inRange(img,lowerLim,upperLim)

    contours,hierarchy=cv.findContours(mask, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    if len(contours) !=0:
       for contour in contours:
          if cv.contourArea(contour)>300:
             x,y,w,h=cv.boundingRect(contour)
             cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
       

    cv.imshow('frame',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):  # Indica que con la tecla ingresada se detiene la captura
     break

webcam.release()
cv.destroyAllWindows()