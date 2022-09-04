import cv2 #facial recognition library
import time

count = 0
faceID = 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceCascade = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_alt2.xml') #loads haar cascade used to detect faces



count = int(input("How many students?\n"))
print("\nInstructions: Enter roll number, press enter, take the photo and repeat.\n")
print("CAPTURING... Look into the camera\nType in 'stop' when you are done\n")
faceID = str(input("Enter roll number \n"))


while count != 0:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        minNeighbors=6,
        scaleFactor=1.2,
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) #draws a rectangle on the camera stream
        print("Taking photo...")
        time.sleep(3)
        cv2.imwrite("dataset/" + str(faceID) + ".jpg", gray[y:y+h,x:x+w]) #cuts out the rectangle,
                                                                          # saves as a greyscale jpg in the directory /dataset

        count = count - 1
        if count != 0:
            faceID = str(input("Enter roll number \n"))


    k = cv2.waitKey(100) & 0xff
    if k == 27: #if ESC is held down, the program ends
        count = 0

print("Exiting...")
cap.release()
cv2.destroyAllWindows()