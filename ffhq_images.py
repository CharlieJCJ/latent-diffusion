import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light

path = "/home/zhren/Charlie/latent-diffusion/src/taming-transformers/data"

isExist = os.path.exists("data/ffhq")
print("data/ffhq", isExist)
isExist = os.path.exists("data/ffhqtrain.txt")
print("data/ffhqtrain.txt", isExist)
isExist = os.path.exists("./src")
print("./src", isExist)
isExist = os.path.exists("./src/taming-transformers")
print("./src/taming-transformers", isExist)
isExist = os.path.exists("/home/zhren/Charlie/latent-diffusion/")
print("/home/zhren/Charlie/latent-diffusion/", isExist)
isExist = os.path.exists("/home/zhren/Charlie/latent-diffusion/src")
print("/home/zhren/Charlie/latent-diffusion/src", isExist)
isExist = os.path.exists("/home/zhren/Charlie/latent-diffusion/src/taming-transformers")
print("/home/zhren/Charlie/latent-diffusion/src/taming-transformers", isExist)
isExist = os.path.exists("/home/zhren/Charlie/latent-diffusion/src/taming-transformers/data")
print("/home/zhren/Charlie/latent-diffusion/src/taming-transformers/data", isExist)
isExist = os.path.exists("/home/zhren/Charlie/latent-diffusion/src/taming-transformers/data/ffhqtrain.txt")
print("/home/zhren/Charlie/latent-diffusion/src/taming-transformers/data/ffhqtrain.txt", isExist)



class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex

# Used as a base, will resize in SR train and validation
class FFHQTrain(FacesBase):
    def __init__(self, size=1024, keys=None):
        super().__init__()
        root = os.path.join(path,"ffhq", "images1024x1024")
        with open(os.path.join(path, "ffhqtrain.txt"), "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys
class FFHQValidation(FacesBase):
    def __init__(self, size=1024, keys=None):
        super().__init__()
        root = os.path.join(path,"ffhq", "images1024x1024")
        with open(os.path.join(path, "ffhqvalidation.txt"), "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys

class FFHQSR(Dataset):
    def __init__(self, size=None,
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True, blur = False):
        """
        FFHQ Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.base = self.get_base()
        print("My base is ", self.base)
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop
        self.blur = blur
        


        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fn = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": PIL.Image.NEAREST,
            "pil_bilinear": PIL.Image.BILINEAR,
            "pil_bicubic": PIL.Image.BICUBIC,
            "pil_box": PIL.Image.BOX,
            "pil_hamming": PIL.Image.HAMMING,
            "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = self.base[i]
        image = Image.open(example["file_path_"])
        
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)
        print("total images = ", len(self.base))
        
        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)

        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]

        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        else:
            LR_image = self.degradation_process(image=image)["image"] # currently used

        if blur:
            sigma = np.random.uniform(low = 0.4, high = 0.6) # credit to Cascaded Diffusion Models for High Fidelity Image Generation
            LR_image = cv2.GaussianBlur(LR_image, (3, 3), sigmaX = sigma) # sigma Y = sigmaX if not specified

        example["image"] = (image/127.5 - 1.0).astype(np.float32);print("current image = ", image.shape);print("LR image = ",LR_image.shape)
        example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)
        return example


class FFHQSRTrain(FFHQSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = FFHQTrain()
        print("dset is", dset)
        return dset


class FFHQSRValidation(FFHQSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = FFHQValidation()
        return dset
