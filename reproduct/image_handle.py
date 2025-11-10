from typing import List
from addict import Dict
from PIL import Image, ImageOps
from abc import ABC
from typing import Optional, Tuple
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np


def pil_to_np(image: Image.Image):
    return np.array(image)

def load_image_pil_normal(image_path):
    """使用PIL加载图像(普通模式)"""
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"图片加载异常: {e}")
        return None

def load_image_pil_exif(image_path):
    """使用PIL加载图像(矫正模式)

    使用 ImageOps.exif_transpose() 自动根据图像的 EXIF 元数据（如旋转、翻转信息）对图像进行矫正。
    比如：一张手机竖拍的照片，在不矫正的情况下显示为横的，这个函数会自动将其旋转为正确的方向。
    """
    # 1. 尝试使用 Pillow 库的 Image.open() 打开指定路径的图像文件
    # 此时图像已加载到内存，但可能方向不正确（例如手机拍摄的照片带有 EXIF 旋转信息）。
    image = load_image_pil_normal(image_path)
    if image is None:
        return None
    
    try:
        # 2. 使用 ImageOps.exif_transpose() 自动根据图像的 EXIF 元数据（如旋转、翻转信息）对图像进行矫正。
        # 比如：一张手机竖拍的照片，在不矫正的情况下显示为横的，这个函数会自动将其旋转为正确的方向。
        return ImageOps.exif_transpose(image)
    except Exception as e:
        print(f"图片矫正异常: {e}")
        return image
        
def load_image_pil_rgb(image_path):
    """使用PIL加载图像(RGB模式)"""
    pil_img = load_image_pil_exif(image_path)
    return pil_img.convert("RGB")

def load_images_pil_rgb(conversations: List[Dict[str, str]]) -> List[Image.Image]:
    """使用PIL加载图像

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            # print('----------------')
            # print(image_path)
            # print('----------------')
            # exit()
            
            # pil_img = Image.open(image_path)
            pil_images.append(load_image_pil_rgb(image_path))

    return pil_images

class BaseTransform(ABC):

    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError

def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform


class BasicImageTransform(BaseTransform):
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        self.mean = mean
        self.std = std
    
        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        x = self.transform(x)
        return x