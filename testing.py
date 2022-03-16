from super_resolution import get_high_resolution
import utils
import os
import cv2

def main():
    # images_path = 'image_test/'
    images_path = 'oriented_pan/'
    # save_path = 'saved_output/'
    save_path = 'hq_pans/'
    
    utils.delete_create_dir(save_path)
    for index, file in enumerate(os.listdir(images_path)):
        try:

            print(f'Processing file {file}\n')
            image_path = os.path.join(images_path, file)
            hq = get_high_resolution(image_path)

            cv2.imwrite(os.path.join(save_path, file), hq)
        except:
            print(f'Error in file {file}')
        
    return



if __name__ == "__main__":
    main()
    # get_high_resolution('test_image.png')