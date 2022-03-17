import cv2
import utils
from real_esr_wrapper import ISR
import numpy as np

hq_image_creator = ISR(tile=128)


def get_high_resolution(img_path):
    image = cv2.imread(img_path)
    image = np.array(image)

    prediction= hq_image_creator.get_super_resolution(image,upscale_factor=4)
    utils.show_image(prediction)
    return prediction


if __name__ == "__main__":
    # main()
    get_high_resolution('test_image.png')





# def main():
#     model = get_model()
#     images_path = 'image_test/'
#     save_path = 'saved_output/'
    
#     delete_create_dir(save_path)
#     for index, file in enumerate(os.listdir(images_path)):
#         try:
#             image = cv2.imread(os.path.join(images_path, file))
#             # try:
#             output, _ = model.enhance(image, outscale=4)
#             # except:
#             #     output,_ = model.enhance(image,)
#             # print(output.shape)
#             # break

#             cv2.imwrite(os.path.join(save_path, file), output)
#         except:
#             print(f'Error in file {file}')
#     return