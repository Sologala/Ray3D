# COCO关键点索引
import numpy as np

COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# H36M关键点索引（部分示例）
H36M_KEYPOINTS = [
    "Hip",
    "RHip",
    "RKnee",
    "RFoot",
    "LHip",
    "LKnee",
    "LFoot",
    "Spine",
    "Thorax",
    "Neck/Nose",
    "Head",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RShoulder",
    "RElbow",
    "RWrist",
]

# 关键点映射
KEYPOINT_MAPPING = {
    "Neck/Nose": "nose",
    "LShoulder": "left_shoulder",
    "RShoulder": "right_shoulder",
    "LElbow": "left_elbow",
    "RElbow": "right_elbow",
    "LWrist": "left_wrist",
    "RWrist": "right_wrist",
    "LHip": "left_hip",
    "RHip": "right_hip",
    "LKnee": "left_knee",
    "RKnee": "right_knee",
    "LFoot": "left_ankle",
    "RFoot": "right_ankle",
}

TRAIN_KEYPOINT_DEFINITIONS = [
    "nose",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def h36_to_coco(h36_keypoints):
    """
    convert points of H36M to COCO format
    :param h36_keypoints: keypoint Array H36M with shape $(n, 3)$, where $n$ is the number of key points.
    :return: key points arrays in COCO format have a shape of (17, 3).
    """
    coco_keypoints = np.zeros((len(COCO_KEYPOINTS), 3))
    for h36_index, h36_name in enumerate(H36M_KEYPOINTS):
        if h36_name in KEYPOINT_MAPPING:
            coco_name = KEYPOINT_MAPPING[h36_name]
            coco_index = COCO_KEYPOINTS.index(coco_name)
            coco_keypoints[coco_index] = h36_keypoints[h36_index]
    return coco_keypoints
