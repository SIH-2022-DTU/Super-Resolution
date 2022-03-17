import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from dotenv import load_dotenv
from realesrgan import RealESRGANer
import numpy as np
from gfpgan import GFPGANer

load_dotenv()
class ISR():
    def __init__(self,tile = 128):
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.model_path = os.getenv('MODEL_PATH')
        self.netscale = 4
        self.upsampler = RealESRGANer(scale=self.netscale, model_path=self.model_path, model=self.model)
        self.upsampler_tiled = RealESRGANer(scale=self.netscale, model_path=self.model_path, model=self.model,tile =tile )

    
    def define_enhancer(self,image_shape = (64,64,3)):
        """
        Args:
            shape (tuple): shape of image -> (h,w,c)
        """
        if(image_shape[0]*image_shape[1] > 128*128):
            self.enhancer = self.upsampler_tiled
        else :
            self.enhancer = self.upsampler

    def get_super_resolution(self,image,enhance_face:bool = False,upscale_factor:int = 4):
        """
        Args:
            image (numpy.array) : image of shape (h,w,c)
           
            upscale_factor (int): default -> 4 -> image wil be upscale to sh x sw x c from h x w x c where s is the upscale factor
        """
        output = None
        if enhance_face:
            face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=upscale_factor,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler)
            _, _, output = face_enhancer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
            return output
        
        if(image.shape[0]*image.shape[1] > 128*128):
            output = self.upsampler_tiled.enhance(image, outscale=upscale_factor)
        else:
            output = self.upsampler.enhance(image, outscale=upscale_factor)
        
        return output[0]