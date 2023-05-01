import cv2
import json

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('train_data/trainner.yml')
cascadePath = "haar/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = cam.read()
    if ret:
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf<50):
                with open('train_data/log.txt') as json_file:  
                    data = json.load(json_file)
                    for p in data['people']:
                        if(int(p['id']) == Id):                
                            name = p['name']
            else:
                name="Not matched"
                break
            cv2.putText(im,str(name), (x,y+h),font, 0.55, (0,255,0),1)
        cv2.imshow('im',im) 
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
cam.release()
cv2.destroyAllWindows()
