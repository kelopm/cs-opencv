import tkinter
import re
import cv2
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import os
import time
import sqlite3

con = sqlite3.connect("peopleDatabase.db") #This connects the sqlite3 library to the database.
cur = con.cursor() #Allows me to execute SQL-based commands.
cur.execute("UPDATE tblPeople SET 'Present' = 'No'") #Resets everyone's present status
con.commit() #Updates the table.

name = 0
presentStatus = 0

path = 'dataset'

i = 0
ids = []
faceSamples = []

imagePaths = 0

shortenedPath = []

recognizer = cv2.face.LBPHFaceRecognizer_create() #loads Local Binary Pattern Histogram used to recognise a face
recognizer.read('trainer/trainer.yml') #reads from trainer file

faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml') #loads cascade
detector = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0) #setting up cam feed
cap.set(3,320)
cap.set(4,240)

minW = 0.01*cap.get(3) #min height and width of squares
minH = 0.01*cap.get(4)

id = 0

win = Tk()
win.geometry("1024x768")

hasTakenPhoto = False

def imageName():
    global hasTakenPhoto
    global imagePaths
    if hasTakenPhoto == True:
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        for i in range(0, len(imagePaths)):
            baseName = os.path.basename(imagePaths[i])
            baseName = (os.path.split(baseName)[-1].split(".")[0])
            shortenedPath.append(baseName)
        return shortenedPath

def imagesAndLabels(path):
    global hasTakenPhoto
    global imagePaths
    if hasTakenPhoto == True:
        for i in range(0, len(imagePaths)):
            pilImg = Image.open(imagePaths[i]).convert('L')
            imgNumpy = np.array(pilImg, 'uint8')
            faces = detector.detectMultiScale(imgNumpy)

            for (x,y,w,h) in faces:
                faceSamples.append(imgNumpy[y:y+h, x:x+w])
                ids.append(int(newPathName[i]))
        return faceSamples, ids

def showFrames():
    global id, name, presentStatus, con, cur, nameLabel

    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)

    label = Label(win)
    label.grid(row=0, column=1)

    imgtk = ImageTk.PhotoImage(image = img)
    label.imgtk = imgtk
    label.configure (image = imgtk)

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,

        minNeighbors=4,
        scaleFactor=1.2,
        minSize=(int(minW), int(minH))
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if (confidence > 100):
            id = "unknown"

        if id != "unknown":
            #write(str(id))
            cur.execute("UPDATE tblPeople SET 'Present' = 'Yes' WHERE RollNumber = " + str(id))
            con.commit()
            name = cur.execute("SELECT Name FROM tblPeople WHERE RollNumber = " + str(id)).fetchall()
            print(name)
            name = str(name[0])
            name = re.findall("[a-zA-Z]+", name)
            name = (' '.join(name))

            presentStatus = cur.execute("SELECT Present FROM tblPeople WHERE RollNumber = " + str(id)).fetchall()
            presentStatus = str(presentStatus[0])
            presentStatus = re.findall("[a-zA-Z]+", presentStatus)
            presentStatus = (' '.join(presentStatus))




            nameLabel.configure(text="Name: " + str(name))
            presentStatusLabel.configure(text="Present: " + str(presentStatus))
    label.after(20, showFrames)


def studentNumber():
    global hasTakenPhoto
    hasTakenPhoto = False
    dataEntry.config(state="normal")

    faceID = dataEntry.get()

    if faceID != "":
        count = 1
        write("[INFO] Starting photo taking, please look into the camera!")
        while count != 0:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                minNeighbors=6,
                scaleFactor=1.2,
                minSize=(20, 20)
            )
            for (x, y, w, h) in faces:
                write("Stay still, taking photo...")
                time.sleep(3)
                cv2.imwrite("dataset/" + str(faceID) + ".jpg", gray[y:y + h, x:x + w])  # cuts out the rectangle,
                # saves as a greyscale jpg in the directory /dataset
                count = 0
                faceID = ""
                dataEntry.delete(0, END)
                dataEntry.config(state="disabled")

                hasTakenPhoto = True



def faceTrainer():
    global newPathName
    global hasTakenPhoto
    if hasTakenPhoto == True:
        newPathName = imageName()
        faces, ids = imagesAndLabels(path)
        recognizer.train(faces, np.array(ids))
        recognizer.write('trainer/trainer.yml')
        write("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
        hasTakenPhoto = False

def write(*message, end = "\n", sep = " "):
    global Console
    text = ""
    for item in message:
        text += "{}".format(item)
        text += sep
    text += end
    Console.insert(INSERT, text)

Console = Text(win)
Console.grid(row = 6, column = 1, pady = 4, padx = 4)




nameLabel = tkinter.Label(win, text="Name: " + str(name))
nameLabel.grid(row=4, column=2, pady=4, padx=4)

presentStatusLabel = tkinter.Label(win, text="Present: " + str(presentStatus))
presentStatusLabel.grid(row=5, column=2, pady=4, padx=4)

tkinter.Label(win, text="Enter roll number").grid(row=5, column=0, pady=4, padx=4)
dataEntry = tkinter.Entry(win, state="disabled")
dataEntry.grid(row=4, column=0, pady=4, padx=4)

showFrames()
tkinter.Button(win, text="1. Take photo", command=studentNumber).grid(row=3,column=0,pady=4, padx=4)

tkinter.Button(win, text="2. Train", command=faceTrainer).grid(row=3,column=1,pady=4, padx=4)


win.mainloop()
