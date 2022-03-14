import shutil
import os
import cv2

def delete_create_dir(path):
    if (os.path.exists(path) == True):
        shutil.rmtree(path)

    os.mkdir(path)
    return

def show_image(image):
    cv2.imshow('Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()