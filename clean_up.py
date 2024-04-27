import cv2
import numpy as np
from utils import ReadFolder, ReadFolderIndex
import os


def CleanUp(basename):
    listName = ReadFolder(basename)
    for indice, element in enumerate(listName):
        list = ReadFolder(basename+"/"+element)
    return list, listName

def ImgTri(name, image, imgd, deg, level):
    img = cv2.imread(image)  
    while True:
        height, width = img.shape[:2]
        (cX, cY) = (width // 2, height // 2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(height):
            for j in range(width):
                if np.random.randint(2) == 0:
                    gray[i, j] = min(gray[i, j] + np.random.randint(0,1), level) # adding noise to image and setting values > 255 to 255. 
                else:
                    gray[i, j] = max(gray[i, j] - np.random.randint(0,1), 0) # subtracting noise to image and setting values < 0 to 0.
                    
        M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
        rotated = cv2.warpAffine(gray, M, (width, height))
        cv2.imshow('rotate', rotated)
        contents = os.listdir('clean')
        if not os.path.exists("clean/" + name):
            os.makedirs("clean/" + name)
                        
        cv2.imwrite("clean/"+name+"/p-{:d}-{:d}-{:d}.png".format(imgd, deg, level),rotated)
        cv2.destroyAllWindows()
        break
        
a = CleanUp('image')

for i, v in enumerate(a[1]):
    print(i)
    for j, k in enumerate(a[0]):
        for deg in range(361):
            for level in range(256):
                if deg % 60 == 0 and level % 128 == 0:
                    ImgTri(a[1][i], 'image/'+a[1][i]+"/"+a[0][j],j, deg, level)
