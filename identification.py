#!/usr/bin/env python
import cv2
import pickle
import numpy as np

#  on change cv2 avec le model 
face_cascade= cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
# on charge le detector de model
recognizer=cv2.face.LBPHFaceRecognizer_create()
# on charge notre model
recognizer.read("trainner.yml")
# on initialise les variables
id_image=0
# on met les infos en noir
color_info=(255, 255, 255)
# en les erreurs en Rouge
color_ko=(0, 0, 255)
#  on met les valide en vert
color_ok=(0, 255, 0)

#  on ouvres le pickles
with open("labels.pickle", "rb") as f:
    #  on chage les datas 
    og_labels=pickle.load(f)
    labels={v:k for k, v in og_labels.items()}
    
def main():
    # on charge la camera
    cap=cv2.VideoCapture(0)
    while True:
        # on charge l'image
        ret, frame=cap.read()
        # les perfs 
        tickmark=cv2.getTickCount()
        # on relgle en noir et blanc
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #  on detecte les visage
        faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=4, minSize=(50, 50))
        # on boucle pour récuperer les infos, y,x width et height
        for (x, y, w, h) in faces:
            #  on dimension les images
            roi_gray=cv2.resize(gray[y:y+h, x:x+w], (50, 50))
            # prédits
            id_, conf=recognizer.predict(roi_gray)
            
            print(conf)
            
            if conf>95:
                color=color_ok
                name=labels[id_]
            else:
                color=color_ko
                name="Inconnu"
            label=name+" "+'{:5.2f}'.format(conf)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
        cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color_info, 2)
        cv2.imshow('Un humain', frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'):
            break



        
if __name__ == "__main__":
    main()

        

