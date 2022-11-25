# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
from random import randrange
import cv2
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor


def Crop(img, v):
    assert -0.125 <= v <= 0.125
    h = img.height
    w = img.width
    x1 = randrange(0, w - int(v * w))
    y1 = randrange(0, h - int(v * h))
    img.crop((x1, y1, w - x1, h - y1))
    return img.resize((w, h))


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -45 <= v <= 45
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.01 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.01 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.01 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.01 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    n_cutout = random.randint(1,4)
    for _ in range(n_cutout):
        v = random.randint(20, 50)
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        # color = (0, 0, 0)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


# Real-world setting augmentations
FONTS = [cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC]

def RandomText(img, _):
    image = np.array(img)
    random_int = np.random.randint(0, 10, size=8)
    text = ""

    if random.random():
        text += "Date: "

    # day/month/year
    for i, r in enumerate(random_int):
        if i in [2, 4]:
            text += "/"
        text += str(r)

    if random.random():
        random_int = np.random.randint(0, 10, size=6)
        for i, r in enumerate(random_int):
            if i in [2, 4]:
                text += ":"
            text += str(r)

    r_font = np.random.choice(FONTS)
    r_thick = np.random.choice([1, 2, 3])
    r_color = random.choice([(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
    r_scale = np.random.choice([0.5, 0.8, 1.0])
    r_x = np.random.choice(np.arange(image.shape[1]))
    r_y = np.random.choice([10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60])
    image = cv2.putText(image, text, (r_x, r_y), r_font, color=r_color, thickness=r_thick, fontScale=r_scale)
    return Image.fromarray(image)


def RandomErase(img, v):
    image = np.array(img)
    h, w, c = image.shape
    if random.random():
        random_h = np.random.randint(10, h-10)
        mag = np.random.choice([2, 3, 4, 5, 6])
        image[random_h - mag:random_h + mag, :, :] = 0.
    else:
        random_w = np.random.randint(10, w - 10)
        mag = np.random.choice([2, 3, 4, 5, 6])
        image[:, random_w - mag:random_w + mag, :] = 0.
    return Image.fromarray(image)


def random_text_aug(images, p=0.1):
    def _random_text_aug(image, p=0.1):
        if random.random() < p:
            font = [cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN,
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC]

            random_int = np.random.randint(0, 10, size=8)
            text = ""

            if random.random():
                text += "Date: "

            # day/month/year
            for i, r in enumerate(random_int):
                if i in [2, 4]:
                    text += "/"
                text += str(r)

            if random.random():
                random_int = np.random.randint(0, 10, size=6)
                for i, r in enumerate(random_int):
                    if i in [2, 4]:
                        text += ":"
                    text += str(r)

            r_font = np.random.choice(font)
            r_thick = np.random.choice([1, 2, 3])
            r_color = random.choice([(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
            r_scale = np.random.choice([0.5, 0.8, 1.0])
            r_x = np.random.choice(np.arange(image.shape[1]))
            r_y = np.random.choice([10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60])
            image = cv2.putText(image, text, (r_x, r_y), r_font, color=r_color, thickness=r_thick, fontScale=r_scale)
        return image
    return [_random_text_aug(x, p) for x in images]


def random_erase_aug(images, p=0.1):
    def _random_erase_aug(image, p=0.1):
        if random.random() < p:
            # horizontal
            h, w, c = image.shape
            if random.random():
                random_h = np.random.randint(10, h-10)
                mag = np.random.choice([2, 3, 4, 5, 6])
                image[random_h - mag:random_h + mag, :, :] = 0.
            else:
                random_w = np.random.randint(10, w - 10)
                mag = np.random.choice([2, 3, 4, 5, 6])
                image[:, random_w - mag:random_w + mag, :] = 0.
        return image
    return [_random_erase_aug(x, p) for x in images]


def baseline_augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    l = [
        (Identity, 0., 1.),
        #(Crop, 0., 0.125),
        (Rotate, 0, 45),
        (Contrast, 0.5, 1.5),
        (Brightness, 0.5, 1.5),
        (Sharpness, 0.5, 1.5),
        (Flip, 0., 1.),
        #(CutoutAbs, 20, 50),
        #(Equalize, 0., 1.),
        #(ShearX, -0.3, 0.3),
        #(ShearY, -0.3, 0.3),
        #(TranslateX, -0.3, 0.3),
        #(TranslateY, -0.3, 0.3),
    ]
    return l

# https://arxiv.org/pdf/2107.04795.pdf
def strong_augment_list():
    l = [
        (AutoContrast, 0., 1.),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0., 1.),
        (Identity, 0., 1.),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0., 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
        (CutoutAbs, 20, 50),
    ]
    return l


def train_augment_list():
    l = [
        (AutoContrast, 0., 1.),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0., 1.),
        (Identity, 0., 1.),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0., 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
        (CutoutAbs, 20, 50),
        (Flip, 0., 1.),
        (Crop, 0., 0.125)
    ]
    return l

# https://arxiv.org/pdf/2107.04795.pdf
def weak_augment_list():
    l = [
        (Flip, 0., 1.),
        (Crop, 0., 0.125)
    ]
    return l


# https://arxiv.org/pdf/2107.04795.pdf
def augmix_augment_list():
    l = [
        (AutoContrast, 0., 1.),
        (Brightness, 0.1, 0.95),
        (Color, 0.1, 0.95),
        (Contrast, 0.1, 0.95),
        (Equalize, 0., 1.),
        (Identity, 0., 1.),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.1, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0., 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
        (CutoutAbs, 20, 50),
        (Flip, 0., 1.),
        (RandomText, 0., 1.),
        (RandomErase, 0., 1.)
    ]
    return l

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m=None):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = baseline_augment_list()

    def __call__(self, img):
        ops = random.sample(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            m = random.random()
            val = m * float(maxval - minval) + minval
            img = op(img, val)
        return img


class AugMix:
    def __init__(self, width=3, depth=-1):
        self.mixture_width = width
        self.mixture_depth = depth
        self.augmentation_list = augmix_augment_list()
    def __call__(self, img):
        img_arr = np.array(img).astype(np.float32)
        ws = np.float32(np.random.dirichlet([1] * self.mixture_width))
        m = np.float32(np.random.beta(1, 1))

        mix = np.zeros_like(img_arr)
        for i in range(self.mixture_width):
            image_aug = np.copy(img_arr)
            depth = args.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            ops = random.sample(self.augmentation_list, k=depth)
            for op, minval, maxval in ops:
                m = random.random()
                val = m * float(maxval - minval) + minval
                image_aug = op(Image.fromarray(image_aug.astype(np.uint8)), val)
                image_aug = np.array(image_aug).astype(np.float32)
            mix += ws[i] * (image_aug.astype(np.float32) / 255.)
        mixed = (1 - m) * (img_arr / 255.) + m * mix
        result = np.uint8(mixed * 255.)
        return Image.fromarray(result)