import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face_LBPHFaceRecognizer.create()
detector= cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faceSamples=[]
    Ids=[]
    Names=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        Name=os.path.split(imagePath)[-1].split(".")[0]
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
            Names.append(Name)
    return faceSamples,Ids,Names

faces,Ids,Names = getImagesAndLabels('train_data/dataset')
recognizer.train(faces, np.array(Ids))
recognizer.save('train_data/trainner.yml')
print ('training complete')
