import cv2
import numpy as np
from .capture import capture_game, grab_features

def observe(wndow):
    frame = capture_game(wndow)
    edges = grab_features(frame)

    h,w = edges.shape
    cropped_img = edges[int(h*0.3):, :]

    zones = w // 3
    left_zone = cropped_img[:, :zones]
    right_zone = cropped_img[:, 2*zones:]
    center_zone = cropped_img[:, zones:2*zones]

    left_density = np.sum(left_zone > 0) / left_zone.size
    right_density = np.sum(right_zone > 0) / right_zone.size
    center_density = np.sum(center_zone > 0) / center_zone.size

    return {
        "left_o": left_density,
        "right_o": right_density,
        "center_o": center_density
    }