import os
import copy
import numpy as np
from lib.utils.utils import deterministic_random
from lib.camera.camera import normalize_screen_coordinates
from lib.utils.keypoint_format_convert import H36M_KEYPOINTS


class Data:

    def __init__(self, data_config):
        """

        :param data_config:
        """
        self.data_config = data_config

        # if True, we load 3D pose
        self.gt_eval = self.data_config["WORLD_3D_GT_EVAL"]
        # if True, remove irrelevant 2D pose
        self.rm_irrlvnt_kpt = self.data_config["REMOVE_IRRELEVANT_KPTS"] or self.data_config["KEYPOINTS"] == "universal"

        # load 3D
        dataset_path_3d = self.data_config["GT_3D"]
        self.load_world_3d_pose(dataset_path_3d)
        # if self.data_config["RAY_ENCODING"]:
        #     self.calculate_ray_3d_pose()

        # 默认计算相机系下的3d点
        self.calculate_camera_3d_pose()

        # load 2D
        dataset_path_2d = self.data_config["GT_2D"]
        self.load_pixel_2d_pose(dataset_path_2d)
        # self.file_names = self.load_view(dataset_path_2d, self.data_config['FRAME_PATH'])

        if self.data_config["INTRINSIC_ENCODING"]:
            self.calculate_intrinsic_2d_pose()

        elif self.data_config["RAY_ENCODING"]:
            self.calculate_ray_2d_pose(self.data_config["ADD_HEIGHT"])

        else:
            self.normalize_pixel_2d_pose()

        # calculate ori
        print("-------------cal ori-------")
        self.cal_ori()
        print("-------------cal ori done-------")
        # sanity check
        self.sanity_check()

    # -------------------------------- #

    def load_world_3d_pose(self, dataset_path):
        """

        :param dataset_path:
        :return:
        """
        if self.data_config["DATASET"] == "h36m":
            from .h36m_dataset import Human36mDataset

            self.dataset = Human36mDataset(dataset_path, universal=self.data_config["KEYPOINTS"] == "universal")

        elif self.data_config["DATASET"] == "humaneva":
            from .humaneva_dataset import HumanEvaDataset

            self.dataset = HumanEvaDataset(dataset_path, universal=self.data_config["KEYPOINTS"] == "universal")

        elif self.data_config["DATASET"] == "3dhp":
            from .mpii_3dhp_dataset import Mpii3dhpDataset

            self.dataset = Mpii3dhpDataset(dataset_path, universal=self.data_config["KEYPOINTS"] == "universal")

        else:
            raise ValueError("Invalid dataset: {}".format(self.data_config["DATASET"]))

    def calculate_camera_3d_pose(self):
        """
        convert 3D pose from world to camera and save in relative format with respect to pelvis
        :return:
        """
        if self.gt_eval:
            for subject in self.dataset.subjects():
                for action in self.dataset[subject].keys():
                    anim = self.dataset[subject][action]
                    if "positions" in anim:
                        positions_3d = []

                        # Method of ours
                        for cam_idx, camera in enumerate(self.dataset.camera_info[subject]):
                            positions_3d.append(camera.world2camera(anim["positions"]))
                        anim["positions_3d"] = positions_3d

    def calculate_ray_3d_pose(self):
        """
        convert 3D pose from world to intermediate space
        :return:
        """
        if self.gt_eval:
            for subject in self.dataset.subjects():
                for action in self.dataset[subject].keys():
                    anim = self.dataset[subject][action]
                    if "positions" in anim:
                        positions_3d = []
                        for cam_idx, camera in enumerate(self.dataset.camera_info[subject]):
                            camera = self.dataset.camera_info[subject][cam_idx]
                            positions_3d.append(camera.world2normalized(anim["positions"]))
                        anim["positions_3d"] = positions_3d

    def calculate_target_ori(self):
        """
        convert 3D pose from world to intermediate space
        :return:
        """
        if self.gt_eval:
            for subject in self.dataset.subjects():
                for action in self.dataset[subject].keys():
                    anim = self.dataset[subject][action]
                    if "positions" in anim:
                        positions_3d = []
                        ori_cam = []
                        for cam_idx, camera in enumerate(self.dataset.camera_info[subject]):
                            camera = self.dataset.camera_info[subject][cam_idx]
                            positions_3d.append(camera.world2camera(anim["positions"]))
                        anim["positions_3d"] = positions_3d

    def load_pixel_2d_pose(self, dataset_path):
        """
        load external 2D detections
        :return:
        """
        keypoints = np.load(dataset_path, allow_pickle=True)

        if self.rm_irrlvnt_kpt:
            self.keypoints, self.keypoints_metadata = self.remove_irrelevant_kpts(keypoints)
        else:
            self.keypoints, self.keypoints_metadata = keypoints["positions_2d"].item(), keypoints["metadata"].item()

    def load_view(self, dataset_path, frame_path):
        """
        load filename of detection bounding box
        :param dataset_path:
        :return:
        """
        keypoints = np.load(dataset_path, allow_pickle=True)
        keypoints = keypoints["positions_2d"].item()

        file_names = dict()
        for subject in keypoints.keys():
            file_names.setdefault(subject, dict())
            for action in keypoints[subject].keys():
                file_names[subject].setdefault(action, list())
                for cam_idx in range(len(keypoints[subject][action])):
                    if isinstance(keypoints[subject][action][cam_idx], dict):
                        names = keypoints[subject][action][cam_idx]["file_name"]
                        frames = list()
                        for name in names:
                            if self.data_config["DATASET"] == "3dhp":
                                if subject.startswith("S"):
                                    sbj, seq, cid = subject.split("_")
                                    frame_name = os.path.join(
                                        frame_path, sbj, seq, "imageSequence", "video_{}".format(cid), name
                                    )
                                if subject.startswith("T"):
                                    frame_name = os.path.join(frame_path, subject, "imageSequence", name)

                            try:
                                assert os.path.exists(frame_name)
                            except:
                                import ipdb

                                ipdb.set_trace()
                            frames.append(frame_name)
                        file_names[subject][action].append(frames)
                    else:
                        return None

        return file_names

    def remove_irrelevant_kpts(self, keypoints):
        return self.dataset.remove_irrelevant_kpts(keypoints, self.data_config["KEYPOINTS"] == "universal")

    def normalize_pixel_2d_pose(self):
        """
        normalize external 2D detections
        :return:
        """
        for subject in self.dataset.subjects():
            for action in self.keypoints[subject]:
                for cam_idx, kps in enumerate(self.keypoints[subject][action]):
                    # Normalize camera frame
                    cam = self.dataset.camera_info[subject][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam.res_w, h=cam.res_h)
                    self.keypoints[subject][action][cam_idx] = kps

    def calculate_intrinsic_2d_pose(self):
        """
        normalize external 2D detections
        :return:
        """
        for subject in self.dataset.subjects():
            for action in self.keypoints[subject]:
                for cam_idx, kps in enumerate(self.keypoints[subject][action]):
                    camera = self.dataset.camera_info[subject][cam_idx]
                    self.keypoints[subject][action][cam_idx] = camera.encode_uv_with_intrinsic(kps)

    def calculate_ray_2d_pose(self, add_height=False):
        """
        normalize external 2D detections
        :return:
        """
        for subject in self.dataset.subjects():
            for action in self.keypoints[subject]:
                for cam_idx, kps in enumerate(self.keypoints[subject][action]):
                    camera = self.dataset.camera_info[subject][cam_idx]
                    frame_dim, kpt_dim, vec_dim = kps.shape

                    if add_height:
                        kps_ray = np.zeros((frame_dim, kpt_dim + 1, vec_dim + 1))
                    else:
                        kps_ray = np.zeros((frame_dim, kpt_dim, vec_dim + 1))
                    kps_ray[:, :kpt_dim] = camera.get_cam_ray_given_uv(kps)
                    if add_height:
                        kps_ray[:, -1] = (-camera.Rn2c.T @ camera.Tn2c).reshape(3)
                    else:
                        pass
                    self.keypoints[subject][action][cam_idx] = kps_ray

    def sanity_check(self):
        """
        make sure that both number of 2D detections and of 3D ground-truths are aligned.
        :return:
        """
        if self.gt_eval:
            for subject in self.dataset.subjects():
                assert subject in self.keypoints, "Subject {} is missing from the 2D detections dataset".format(subject)
                for action in self.dataset[subject].keys():
                    assert (
                        action in self.keypoints[subject]
                    ), "Action {} of subject {} is missing from the 2D detections dataset".format(action, subject)
                    if "positions_3d" not in self.dataset[subject][action]:
                        continue

                    for cam_idx in range(len(self.keypoints[subject][action])):

                        # We check for >= instead of == because some videos in H3.6M contain extra frames
                        mocap_length = self.dataset[subject][action]["positions_3d"][cam_idx].shape[0]
                        assert self.keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                        if self.keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                            # Shorten sequence
                            self.keypoints[subject][action][cam_idx] = self.keypoints[subject][action][cam_idx][
                                :mocap_length
                            ]

                    assert len(self.keypoints[subject][action]) == len(self.dataset[subject][action]["positions_3d"])

    # -------------------------------- #

    def get_dataset(self):
        """
        retrieve dataset which contains world 3D pose, camera 3D pose and camera parameters
        :return:
        """
        return self.dataset

    def get_keypoints(self):
        """
        retrieve pixel 2D pose
        :return:
        """
        return self.keypoints

    def get_2d_kpts(self):
        """

        :return:
        """
        keypoints_symmetry = self.keypoints_metadata["keypoints_symmetry"]
        kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        return kps_left, kps_right

    def get_3d_joints(self):
        """

        :return:
        """
        joints_left, joints_right = list(self.dataset.skeleton().joints_left()), list(
            self.dataset.skeleton().joints_right()
        )
        return joints_left, joints_right

    # -------------------------------- #

    def fetch_via_subject(self, subjects, action_filter=None, subset=1, parse_3d_poses=True):
        """

        :param subjects:
        :param action_filter:
        :param subset:
        :param parse_3d_poses:
        :return:
        """
        out_poses_3d = []
        out_poses_2d = []
        out_poses_ori = []
        out_camera_params = []

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                poses_2d = self.keypoints[subject][action]
                poses_3d = self.dataset[subject][action]["positions_3d"]
                poses_ori = self.dataset[subject][action]["cam_ori"]
                assert len(poses_3d) == len(poses_2d) == len(poses_ori), "Camera count mismatch"
                for i in range(len(poses_2d)):  # Iterate across cameras
                    out_poses_2d.append(copy.deepcopy(poses_2d[i]))
                    out_poses_3d.append(copy.deepcopy(poses_3d[i]))
                    out_poses_ori.append(copy.deepcopy(poses_ori[i]))

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.data_config["DOWNSAMPLE"]
        if subset < 1:
            for i in range(len(out_poses_2d)):
                n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start : start + n_frames : stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][start : start + n_frames : stride]
        elif stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d, out_poses_ori

    def fetch_via_action(self, actions, camera_idx=None):
        """

        :param actions:
        :return:
        """
        out_poses_3d = []
        out_poses_2d = []
        out_poses_ori = []
        out_camera_params = []

        for subject, action in actions:
            poses_2d = self.keypoints[subject][action]
            poses_3d = self.dataset[subject][action]["positions_3d"]
            poses_ori = self.dataset[subject][action]["cam_ori"]
            assert len(poses_3d) == len(poses_2d) == len(poses_ori), "Camera count mismatch"
            for i in range(len(poses_2d)):  # Iterate across cameras
                if camera_idx is not None:
                    if i != camera_idx:
                        continue
                out_poses_2d.append(copy.deepcopy(poses_2d[i]))
                out_poses_3d.append(copy.deepcopy(poses_3d[i]))
                out_poses_ori.append(copy.deepcopy(poses_ori[i]))
                camera = self.dataset.camera_info[subject][i]
                out_camera_params.append(camera)

        if len(out_poses_3d) == 0:
            out_poses_3d = None
        if len(out_camera_params) == 0:
            out_camera_params = None

        stride = self.data_config["DOWNSAMPLE"]
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d, out_poses_ori

    def cal_ori(self):
        """
        Calculate target's orientation for each camera.
        The orientation is defined as the chest facing direction, which is the cross product of the vector
        from left shoulder to right shoulder and the vector of torso (from neck to hip center).
        :return:
        """
        if self.gt_eval:
            for subject in self.dataset.subjects():
                for action in self.dataset[subject].keys():
                    anim = self.dataset[subject][action]
                    if "positions" in anim:
                        positions_3d = []
                        ori_each_cam = []
                        for cam_idx, camera in enumerate(self.dataset.camera_info[subject]):
                            camera = self.dataset.camera_info[subject][cam_idx]
                            p3d_cam = camera.world2camera(anim["positions"])
                            positions_3d.append(p3d_cam)

                            # 提取关键点索引
                            l_shoulder_idx = H36M_KEYPOINTS.index("LShoulder")
                            r_shoulder_idx = H36M_KEYPOINTS.index("RShoulder")
                            neck_idx = H36M_KEYPOINTS.index("Neck/Nose")
                            l_hip_idx = H36M_KEYPOINTS.index("LHip")
                            r_hip_idx = H36M_KEYPOINTS.index("RHip")

                            # 批量提取关键点坐标
                            # 假设p3d_cam形状为 [帧数, 关节数, 3]
                            l_shoulder = p3d_cam[:, l_shoulder_idx]  # [帧数, 3]
                            r_shoulder = p3d_cam[:, r_shoulder_idx]
                            neck = p3d_cam[:, neck_idx]
                            l_hip = p3d_cam[:, l_hip_idx]
                            r_hip = p3d_cam[:, r_hip_idx]

                            # 计算髋关节中心 (向量化)
                            hip_center = (l_hip + r_hip) / 2  # [帧数, 3]

                            # 计算肩膀向量和躯干向量 (向量化)
                            shoulder_vector = l_shoulder - r_shoulder  # [帧数, 3]
                            torso_vector = hip_center - neck  # [帧数, 3]

                            # 计算叉乘 (向量化)
                            chest_direction = np.cross(torso_vector, shoulder_vector)  # [帧数, 3]

                            # 归一化向量 (向量化)
                            norms = np.linalg.norm(chest_direction, axis=1, keepdims=True)  # [帧数, 1]
                            valid_mask = norms > 1e-6  # 避免除以零

                            # 修复：使用广播处理归一化
                            chest_direction[valid_mask[:, 0]] /= norms[valid_mask[:, 0]]

                            # 计算yaw角 (向量化)
                            yaw = np.arctan2(chest_direction[:, 2], chest_direction[:, 0])  # [帧数]
                            yaw[yaw < 0] += np.pi * 2
                            yaw = np.rad2deg(yaw)
                            # 存储朝向信息
                            ori_each_cam.append(yaw)

                        anim["positions_3d"] = positions_3d
                        anim["cam_ori"] = ori_each_cam

    def cal_ori1(self):
        """
        Calculate target's orientation for each camera.
        The orientation is defined as the chest facing direction, which is the cross product of the vector
        from left shoulder to right shoulder and the vector of torso (from neck to hip center).
        :return:
        """
        if self.gt_eval:
            for subject in self.dataset.subjects():
                for action in self.dataset[subject].keys():
                    anim = self.dataset[subject][action]
                    if "positions" in anim:
                        positions_3d = []
                        ori_each_cam = []
                        for cam_idx, camera in enumerate(self.dataset.camera_info[subject]):
                            camera = self.dataset.camera_info[subject][cam_idx]
                            p3d_cam = camera.world2camera(anim["positions"])
                            positions_3d.append(p3d_cam)

                            # 计算每帧姿态在相机系下的朝向
                            oris = []
                            for pos_3d in p3d_cam:
                                # 提取关键点坐标
                                l_shoulder = pos_3d[H36M_KEYPOINTS.index("LShoulder")]  # 左肩膀
                                r_shoulder = pos_3d[H36M_KEYPOINTS.index("RShoulder")]  # 右肩膀
                                neck = pos_3d[H36M_KEYPOINTS.index("Neck/Nose")]  # 颈部
                                hip_center = (
                                    pos_3d[H36M_KEYPOINTS.index("LHip")] + pos_3d[H36M_KEYPOINTS.index("RHip")]
                                ) / 2  # 髋关节中心

                                # 计算左右肩膀向量
                                shoulder_vector = r_shoulder - l_shoulder

                                # 计算躯干向量 (颈部到髋关节中心)
                                torso_vector = hip_center - neck

                                # 计算叉乘 (得到胸部朝向向量)
                                chest_direction = np.cross(shoulder_vector, torso_vector)

                                # 归一化向量
                                if np.linalg.norm(chest_direction) > 0:
                                    chest_direction = chest_direction / np.linalg.norm(chest_direction)

                                # 计算yaw角 (仅考虑水平方向旋转)
                                # Yaw角定义为向量在XZ平面投影与Z轴的夹角
                                yaw = np.arctan2(chest_direction[0], chest_direction[2])

                                # 存储朝向信息
                                oris.append(yaw)
                            ori_each_cam.append(oris)

                        anim["positions_3d"] = positions_3d
                        anim["cam_ori"] = ori_each_cam
