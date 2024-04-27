import cv2
import os
# créer le dossier
contents = os.listdir('image')
input = input("Enter le prenom de la personne : ")
input = input.lower()
if input in contents:
    path = input
else:
    os.mkdir('image/'+input)
    path = input


# on charge le model
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
# on allume la camera
capture = cv2.VideoCapture(0)

# taille de la photo
c=50
# id de la photo
id=0
# on cboucle de manière infinite
while True:
    # on charge l'image de la camera
    ret, frame = capture.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # on detecte l'image
    face = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(c, c))
    # on boucle pour prendre des photo et on skock
    for x, y, w, h in face:
        cv2.imwrite("image/"+path+"/p-{:d}.png".format(id), frame[y:y+h, x:x+w])
        id+=1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # on affiche l'image
    cv2.imshow('video', gray)
    # cv2.imshow('video', frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break

capture.release()
cv2.destroyAllWindows()