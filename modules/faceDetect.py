import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create() #loads Local Binary Pattern Histogram used to recognise a face
recognizer.read('trainer/trainer.yml') #reads from trainer file

faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml') #loads cascade

cap = cv2.VideoCapture(0) #setting up cam feed
cap.set(3,640)
cap.set(4,480)
font = cv2.FONT_HERSHEY_SIMPLEX #font used when displaying ID and confidence level

minW = 0.1*cap.get(3) #min height and width of squares
minH = 0.1*cap.get(4)

id = 0

path = 'dataset'

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        
        minNeighbors=4,
        scaleFactor=1.2
        ,     
        minSize=(int(minW), int(minH))
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if (confidence < 100):
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    print(k)
    if k == 27: #requires pressing 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
