import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together. 把几个扩增方式压缩在一起
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms: #依次遍历各个扩增操作
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object): #对坐标归一化
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object): #饱和度变化
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper) #由于之前已经转换成HSV空间

        return image, boxes, labels


class RandomHue(object): #色度变化
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object): #交换通道Channels
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0)) #通道候选顺序

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))] #随机选择一个通道顺序
            shuffle = SwapChannels(swap)  # shuffle channels #交换通道
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object): #颜色空间转换
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object): #对比度变化，对像素乘以一个数，增加或减弱两个像素之间的差值
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object): #随机亮度，亮度由像素值大小控制
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options) #选择一种采样模式，我的理解是选择IOU阈值
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode #最大，最小IOU
            if min_iou is None:
                min_iou = float('-inf') #无穷小
            if max_iou is None:
                max_iou = float('inf') #无穷大

            # max trails 路径 (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width) #随机选择crop后的w,h
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                # crop的图片的纵横比必须在0.5~2之间
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w) #左上角随机选一点
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                # 选择的crop的框的坐标, 这时的坐标值是相对于输入图片来说的，因此起始左上角并不是0
                # 下面的boxes也是相对于输入图片来说的，因此他们的坐标比较才有意义
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect) #计算“所有”标注框和Crop框的IOU，注意是所有的标注框 ground truth

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max(): # overlap 必须要全部介于min_iou,max_iou，程序才能向下进行，有50次机会
                    continue

                # cut the crop from the image 从原始图片上剪裁出crop图片
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                # centers ground truth的中心点坐标
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0 #boxes = [xmin, ymin, xmax, ymax] 计算中心
                
                # 选取中心点在rect矩形内的boxes
                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any(): #如果没有有效的boxes，重新来过
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy() #此时选择的boxes，只是其中心点在我们选择的裁剪框内，下面还有调整boxes的坐标

                # take only matching gt labels
                current_labels = labels[mask]

                # 下面要调整boxes的坐标，开始并没有想通，为何要调整
                # 因为我们送到网络中的图片不再是输入图片，而是裁剪后的图片，因此在原始图片上标注的框，必须调整到和裁剪后的图片相对应起来，作为target
                # 也就是在Crop后的图片上画标注框
                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2]) #maximum 
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2] #裁剪框的左上角为原点，计算相对坐标

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2] ##裁剪框的左上角为原点，计算相对坐标

                return current_image, current_boxes, current_labels


class Expand(object): #扩展
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2): #随机选择做与不做
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4) #扩展图片边缘的比例
        left = random.uniform(0, width*ratio - width) #x_left
        top = random.uniform(0, height*ratio - height) #y_left

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean #用均值填充一张图片
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy() #计算框的偏移
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1] #图片width翻转
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2] #翻转框的坐标，y轴不变，x轴翻转（xmin, xmax）
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(), #对比对变化
            ConvertColor(transform='HSV'), #颜色空间转换
            RandomSaturation(), #配合上面的转换，进行饱和度变换
            RandomHue(), #配合HSV空间进行色度变化
            ConvertColor(current='HSV', transform='BGR'), #HSV->BGR
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness() #随机亮度变化
        self.rand_light_noise = RandomLightingNoise() #交换Channels

    def __call__(self, image, boxes, labels):
        im = image.copy() #复制原始图像
        im, boxes, labels = self.rand_brightness(im, boxes, labels) #随机亮度变化
        if random.randint(2):
            distort = Compose(self.pd[:-1]) #对比度变化放在前面
        else:
            distort = Compose(self.pd[1:]) #对比度变化放在后面
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels) #交换Channels


class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(), #把输入图像又int8转成float32
            ToAbsoluteCoords(), #左边之前进行了归一化，现在将其再转换回绝对坐标
            PhotometricDistort(), #光度失真，加干扰扩增
            Expand(self.mean), #将输入图片扩大，扩大的部分用mean填充
            RandomSampleCrop(), #随机裁剪图片，调整框的坐标
            RandomMirror(), #镜像翻转
            ToPercentCoords(), #对坐标归一化
            Resize(self.size), #resize到300*300
            SubtractMeans(self.mean) #减均值
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
