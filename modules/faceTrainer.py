import cv2
import numpy as np
import os
from PIL import Image

path = '../dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("../haarcascade/haarcascade_frontalface_default.xml")
i = 0
i = 0
ids = []
faceSamples = []
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
shortenedPath = []

def imageName():
    for i in range(0, len(imagePaths)):
        baseName = os.path.basename(imagePaths[i])
        baseName = (os.path.split(baseName)[-1].split(".")[0])
        shortenedPath.append(baseName)
    return shortenedPath

def imagesAndLabels(path):
    for i in range(0, len(imagePaths)):
        pilImg = Image.open(imagePaths[i]).convert('L')
        imgNumpy = np.array(pilImg, 'uint8')
        faces = detector.detectMultiScale(imgNumpy)

        for (x,y,w,h) in faces:
            faceSamples.append(imgNumpy[y:y+h, x:x+w])
            ids.append(int(newPathName[i]))
    return faceSamples, ids

print ("\n performing face training...")

newPathName = imageName()
faces, ids = imagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write('../trainer/trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))












