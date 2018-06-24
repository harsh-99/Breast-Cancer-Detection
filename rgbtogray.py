import cv2 
import os

path  = '/home/harsh/project/DDSM'
for root, dirs, files in os.walk(path):
        for filename in files:
            print(os.path.join(root,os.path.basename(filename)))
            image = cv2.imread(os.path.join(root,os.path.basename(filename)))
            print(image.shape)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(gray_image.shape)
            cv2.imwrite(os.path.join(root,os.path.basename(filename)),gray_image)

