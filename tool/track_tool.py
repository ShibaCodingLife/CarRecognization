import os.path

import numpy as np
import torch

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tool.generate_cpu_detections import create_ov_box_encoder
from tool.generate_gpu_detections import create_box_encoder


def xxyy2xywh(boxes: np.ndarray):
    new_boxes = []
    for box in boxes:
        x1, x2, y1, y2 = box
        x, y = x1, y1
        w = x2 - x1
        h = y2 - y1
        new_boxes.append((x, y, w, h))
    return np.array(new_boxes)


def xyxy2xxyy(boxes):
    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        new_boxes.append([x1, x2, y1, y2])
    return new_boxes


def track_objects(tracker, frame, boxes, confidences, file):
    """
    Track objects in a single frame using DeepSORT.

    Parameters:
    tracker (Tracker): The DeepSORT tracker.
    frame (np.array): The frame image.
    boxes (list of list): The bounding boxes in the format [[x1, y1, x2, y2], ...].

    Returns:
    list of list: The tracked bounding boxes and their IDs in the format [[x1, y1, x2, y2, id], ...].
    """
    if type(file) == tuple:
        mars_xml, mars_bin = file
        encoder = create_ov_box_encoder(mars_xml, mars_bin)
    else:
        encoder = create_box_encoder(file, input_name='images', output_name='features', batch_size=len(boxes))
    # Extract features from the boxes
    features = encoder(frame, boxes)

    # Create detections
    boxes = xxyy2xywh(boxes)
    detections = [Detection(bbox, confidence, np.array(6), feature) for bbox, feature, confidence in
                  zip(boxes, features, confidences)]

    # Update tracker
    tracker.predict()
    tracker.update(detections)

    # Output the tracked bounding boxes
    tracked_boxes = []
    ids = []
    for track in tracker.tracks:
        bbox = track.to_tlbr()  # Get the predicted bounding box

        tracked_boxes.append(list(bbox))  # Add the track ID to the output
        ids.append(track.track_id)
    tracked_boxes = xyxy2xxyy(tracked_boxes)
    return tracked_boxes, ids
