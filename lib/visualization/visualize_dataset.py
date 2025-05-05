from lib.dataset.h36m_dataset import Human36mDataset, h36m_skeleton
from lib.utils.keypoint_format_convert import COCO_KEYPOINTS, h36_to_coco, H36M_KEYPOINTS
from matplotlib import pyplot as plt
from lib.visualization.viewer import Human36MPoseVisualizer


def main():
    # load dataset
    dataset = Human36mDataset("data/h36m/data_3d_h36m.npz")
    viewer = Human36MPoseVisualizer()
    print(dataset.subjects())
    for sub in dataset.subjects():
        for action in dataset._data[sub].keys():
            positions = dataset._data[sub][action]["positions"]
            for ts, pts in enumerate(positions):
                # print(ts, pts[0])
                viewer.update_plot(pts)
                plt.pause(0.1)
    pass


if __name__ == "__main__":
    main()
