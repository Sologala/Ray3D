from lib.dataset import Data
import os
import importlib
from cfg.arguments import parse_args
from lib.visualization.viewer import Human36MPoseVisualizer
from matplotlib import pyplot as plt


def main():
    args = parse_args()
    cfg = importlib.import_module("cfg." + args.cfg)
    data_config, model_config, train_config, plot_config = (
        cfg.data_config,
        cfg.model_config,
        cfg.train_config,
        cfg.plot_config,
    )
    pose_data = Data(data_config)
    print(pose_data.dataset)

    for sub in pose_data.dataset.subjects():
        for action in pose_data.dataset._data[sub].keys():
            positions = pose_data.dataset._data[sub][action]["positions_3d"]
            keypoints = pose_data.keypoints[sub][action]
            yaws = pose_data.dataset._data[sub][action]["cam_ori"]
            n_cam = len(pose_data.dataset.camera_info[sub])
            viewer = Human36MPoseVisualizer(n_cam=n_cam, title=f"{sub}_{action}".replace(" ", "_"), attributes=["2d", "ori"])
            n_frame = len(positions[0])
            for i in range(n_frame):
                for j in range(n_cam):
                    # viewer.update_plot(j, positions[j][i])
                    viewer.update_plot_2d(j, keypoints[j][i])
                    viewer.update_ori(j, yaws[j][i])
                viewer.show(0.003)


if __name__ == "__main__":
    main()
