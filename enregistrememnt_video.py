import cv2
import os

video_path = "Les secrets et coulisses du cultissime Morning Live.mp4"

# le dossier 
input_name = input("Entrez le nom de la personne : ").lower()
output_folder = os.path.join('image', input_name)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Charge le modèle
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Charge la vidéo 
capture = cv2.VideoCapture(video_path)

# Taille minimale du visage à détecter
min_face_size = 50

# ID pour nommer les visages enregistrés
id = 0

# Boucle sur chaque frame de la vidéo
while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    # Conversion en niveaux de gris pour la détection de visage
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détection des visages dans la frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_face_size, min_face_size))
    
    # Enregistrement de chaque visage détecté dans le dossier spécifié
    for x, y, w, h in faces:
        cv2.imwrite(os.path.join(output_folder, f"p-{id}.png"), frame[y:y+h, x:x+w])
        id += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Affichage de la vidéo avec les visages encadrés
    cv2.imshow('video', frame)
    
    # Attendre la touche 'q' pour quitter
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Libérer la capture et fermer les fenêtres
capture.release()
cv2.destroyAllWindows()
