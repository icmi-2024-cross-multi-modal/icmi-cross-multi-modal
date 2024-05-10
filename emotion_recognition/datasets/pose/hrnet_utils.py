from typing import Union, Tuple, List
import sys

import numpy as np


def apply_bbox_to_frame(frame, bbox)->np.ndarray:
    return frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def cut_frame_to_pose(extractor, frame:np.ndarray, return_bbox:bool=False)->Union[Tuple[np.ndarray, List[int]],
                                                                                  np.ndarray,
                                                                                  None]:
    adding_factor = 25
    height, width, _ = frame.shape
    prediction = extractor.predict(frame)
    if prediction is None or len(prediction[0]) == 0:
        return None
    bbox = prediction[0][0]
    # expand bbox so that it will cover all human with some space
    # height
    bbox[1] -= adding_factor
    bbox[3] +=adding_factor
    # width
    bbox[0] -= adding_factor
    bbox[2] += adding_factor
    # check if we are still in the frame
    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[3] > height:
        bbox[3] = height
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[2] > width:
        bbox[2] = width
    # cut frame
    bbox = [int(x) for x in bbox]
    cut_frame = apply_bbox_to_frame(frame, bbox)
    if return_bbox:
        return cut_frame, bbox
    return cut_frame