import cv2
import random
import json

data = {}
data['people'] = []
with open('train_data/log.txt', 'r') as json_file:  
    data = json.load(json_file)
detector=cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
name=str(input('Your name Sir: \n'))
ID=str(random.randint(0,999))
i=0
camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            i=i+1
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imwrite("train_data/dataset/"+name+"."+ID+'.'+ str(i) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('Camera',img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        if i > 100:
            data['people'].append({  
                'name': name,
                'id': ID
            })
            with open('train_data/log.txt', 'w') as outfile:  
                json.dump(data, outfile)
            print ("Thank You Mr. "+name)
            break
camera.release()
cv2.destroyAllWindows()
