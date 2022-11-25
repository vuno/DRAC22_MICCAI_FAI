import cv2
import numpy as np

from albumentations import ImageOnlyTransform


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


def circle_crop(img, sigmaX=10):
    """
    Create circular crop around image centre    
    """    
    
    # img = cv2.imread(img)
    img = crop_image_from_gray(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, _ = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img

def apply_ben_color(img, img_size, sigmaX=10):
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
        
    return img


class CropFundus(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 1):
        super().__init__(always_apply, p)
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return crop_image_from_gray(img) # circle_crop(img)
    

class BensPreprocessing(ImageOnlyTransform):
    def __init__(self, sigmaX, always_apply: bool = False, p: float = 1):
        super().__init__(always_apply, p)
        self.sigmaX = sigmaX
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return cv2.addWeighted( img, 4, cv2.GaussianBlur( img , (0, 0) , self.sigmaX) , -4 ,128)


class CropResizeBens(ImageOnlyTransform):
    def __init__(self, circle_crop: bool = False, img_size: int = 0, sigmaX: int = 0, always_apply: bool = False, p: float = 1):
        super().__init__(always_apply, p)
        
        self.img_size = img_size
        self.circle_crop = circle_crop
        self.sigmaX = sigmaX
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if self.circle_crop:
            img = circle_crop(img)
        else:
            img = crop_image_from_gray(img)
        
        if self.img_size > 0:
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        if self.sigmaX > 0:
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), self.sigmaX), -4 ,128)
        return img