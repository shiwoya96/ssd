# i_C.MODEL.PRIORS = CN()
# _C.MODEL.PRIORS.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
# _C.MODEL.PRIORS.STRIDES = [8, 16, 32, 64, 100, 300]
# _C.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
# _C.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]
# _C.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# # When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
# # #boxes = 2 + #ratio * 2
# _C.MODEL.PRIORS.BOXES_PER_LOCATION = [
#     4, 6, 6, 6, 4, 4
# ]  # number of boxes per feature map location
# _C.MODEL.PRIORS.CLIP = True

from itertools import product
import torch
from math import sqrt


class PriorBox:

    def __init__(self):
        self.image_size = 300
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.strides = [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = True
        #PriorBox NUMBERS: 38*38*(1+1+2*1) + 19*19*(1+1+2*2) + 10*10*(1+1+2*2) + 5*5*(1+1+2*2) + 3*3*(1+1+2*1) + 1*1*(1+1+2*1) = 8732
        #F * F *(MIN_BOX + BIG_BOX + RATIO_BOX(2) * RATIO_NUM)
    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k]
            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                # small sized square box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors


if __name__ == '__main__':
    pb = PriorBox()
    print(pb().shape)
