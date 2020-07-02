import cv2

faceCasCade = cv2.CascadeClassifier("C:/Users/Nour/PycharmProjects/OpenCVPython/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
path = "C:/Users/Nour/Desktop/1.jpg"
img = cv2.imread(path)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCasCade.detectMultiScale(gray,1.1,4)

for (x,y,w,h)in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow("Result",img)
cv2.waitKey(0)