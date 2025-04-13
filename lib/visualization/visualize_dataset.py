from lib.dataset.h36m_dataset import Human36mDataset, h36m_skeleton
from lib.utils.keypoint_format_convert import COCO_KEYPOINTS, h36_to_coco, H36M_KEYPOINTS
from matplotlib import pyplot as plt


class Human36MPoseVisualizer:
    def __init__(self) -> None:
        self.init_plot()
        pass

    def init_plot(self):
        """
        初始化 3D 绘图窗口
        """
        # 创建一个 3D 图形
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        # 设置坐标轴标签
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        # 开启交互模式
        plt.ion()
        # 初始化散点图和线条为空
        self.sc = None
        self.lines = []
        # 定义 COCO 关键点的连接顺序，用于绘制骨架
        self.INDICES = [
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 4),
            (4, 5),
            (5, 6),
            (0, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (8, 11),
            (11, 12),
            (12, 13),
            (8, 14),
            (14, 15),
            (15, 16),
        ]
        self.COLORS = [
            (0 / 255.0, 215 / 255.0, 255 / 255.0),
            (0 / 255.0, 255 / 255.0, 204 / 255.0),
            (0 / 255.0, 134 / 255.0, 255 / 255.0),
            (0 / 255.0, 255 / 255.0, 50 / 255.0),
            (77 / 255.0, 255 / 255.0, 222 / 255.0),
            (77 / 255.0, 196 / 255.0, 255 / 255.0),
            (77 / 255.0, 135 / 255.0, 255 / 255.0),
            (191 / 255.0, 255 / 255.0, 77 / 255.0),
            (77 / 255.0, 255 / 255.0, 77 / 255.0),
            (0 / 255.0, 127 / 255.0, 255 / 255.0),
            (255 / 255.0, 127 / 255.0, 77 / 255.0),
            (0 / 255.0, 77 / 255.0, 255 / 255.0),
            (77 / 255.0, 255 / 255.0, 77 / 255.0),
            (0 / 255.0, 127 / 255.0, 255 / 255.0),
            (255 / 255.0, 127 / 255.0, 77 / 255.0),
            (0 / 255.0, 77 / 255.0, 255 / 255.0),
        ]

    def update_plot(self, points_3d):
        """
        Update key points and skeletal structure in 3D drawing window.

        Parameter:
        points_3d (numpy.ndarray): shape (17, 3) numpy array containing 17 key points with 3D coordinates.
        """
        # 清空之前的散点图和连线
        if self.sc is not None:
            self.sc.remove()
            self.sc = None
        for line in self.lines:
            line.remove()
        self.lines = []

        # 提取 x, y, z 坐标
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]

        # 更新散点图
        self.sc = self.ax.scatter(x, y, z, c="r", marker="o")

        body_keypoint_names = [
            "LHip",
            "RHip",
            "LKnee",
            "RKnee",
            "LShoulder",
            "RShoulder",
            "LFoot",
            "RFoot",
            "Neck/Nose",
            "Head",
            "LElbow",
            "RElbow",
            "LWrist",
            "RWrist",
        ]

        for ii, idx in enumerate(self.INDICES):
            start, end = idx
            start_point_name, end_point_name = H36M_KEYPOINTS[start], H36M_KEYPOINTS[end]
            if not (start_point_name in body_keypoint_names and end_point_name in body_keypoint_names):
                continue

            (line,) = self.ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], c=self.COLORS[ii])
            self.lines.append(line)

        # 重绘图形
        self.fig.canvas.draw()
        # 处理事件
        self.fig.canvas.flush_events()


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
