from . import transforms as T
from torchvision.transforms import Compose


def build_transforms(is_train=True):
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    multi_crop_test = False #  NOTE not sure this param will be used
    if is_train:
        transform = [T.GroupResize((224, 224)),
                           T.ToTensor(),
                           T.GroupNormalize(img_mean, img_std)]
    else:
        if multi_crop_test:
            transform = [
                T.GroupResize((256, 256)),
                T.GroupRandomCrop((256, 256))
            ]
        else:
            transform = [
                T.GroupResize((224, 224))
            ]
        transform +=[
            T.ToTensor(),
            T.GroupNormalize(img_mean, img_std)
        ]

    transform = Compose(transform)

    return transform

        
        
                           
