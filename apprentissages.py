# diviser tous part 2 - 100 | 80 testeurs | 20 service base
import cv2 # image package
import os # nous executer des commande basique 
import numpy as np # mathématique poussé 
import pickle  # pickle package

# py -3 -m pip install numpy pickle 

image_dir = "./clean/" # repo 
current_id = 0 # id a nos images
label_ids = {} # labels { allan_0: image_1}
x_train = [] # { entrainenment }
y_labels = [] # label tester 

for root, dirs, files in os.walk(image_dir):
    # if files 
    if len(files):
        # label = name dossier
        label = root.split("/")[-1]
        for file in files:
            # files = array 
            # file = path name fichier
            if file.endswith("png"):
                # if png
                path = os.path.join(root, file)
                # on créer le path de l'image 
                # si le dossier n'a pas de label
                if not label in label_ids:
                    # on crer un labels
                    label_ids[label] = current_id
                    # on incremente le id du label
                    current_id += 1
                # on créer une deuxieme varibale id
                id_ = label_ids[label]
                # on lit l'image et on lui noir / blanc
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                # on chaneg la taille de l'image
                image = cv2.resize(image, (200, 200))
                # on array entrainnement 
                x_train.append(image)
                y_labels.append(id_)

# if pickle crate dossier sinon lit le 
with open("labels.pickle", "wb") as f:
    # ajoute les images a l'interieur
    a = pickle.dump(label_ids, f)

# change la taille des image et tu les met dans opencv
x_train = [cv2.resize(img, (200, 200)) for img in x_train]

# tu met dans un tableau np x_train
x_train = np.array(x_train)
# tu met dans un tableau np les labels
y_labels = np.array(y_labels)

#  tu actuve le model de reconnaissance faciale 
recognizer = cv2.face.LBPHFaceRecognizer_create()
#  tu entraine
recognizer.train(x_train, y_labels)
#  tu stocke dans le yml
recognizer.save("trainner.yml")





