import os
import cv2
from modules.xfeat import XFeat

ACC_FEATURES = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ACC_FEATURES)
xfeat = XFeat()

image1 = cv2.imread(os.path.join(ROOT_DIR, "bag/bag/images", "calib_0001.png"))
image2 = cv2.imread(os.path.join(ROOT_DIR, "bag/bag/images", "calib_0002.png"))

matches_list = xfeat.match_xfeat_star(image1, image2)
print(matches_list[0].shape)