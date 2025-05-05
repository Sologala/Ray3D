from matplotlib import pyplot as plt
import numpy as np


class Human36MPoseVisualizer:
    def __init__(self, n_cam=4, figsize=(15, 10), title="", attributes=["3d", "2d", "ori"]) -> None:
        self.n_cam = n_cam
        self.figsize = figsize
        self.title = title
        self.attributes = attributes
        self.init_plot()

        # 预分配箭头对象
        self.ori_arrows = [[None for _ in attributes] for __ in range(n_cam)]

    def init_plot(self):
        self.fig = plt.figure(figsize=self.figsize)
        self.axes = []
        rows = int(np.ceil(np.sqrt(self.n_cam)))
        cols = int(np.ceil(self.n_cam / rows)) * len(self.attributes)

        for i in range(self.n_cam):
            sub_axs = {}
            for j, attr_name in enumerate(self.attributes):
                if attr_name == "3d":
                    ax = self.fig.add_subplot(rows, cols, len(self.attributes) * i + j + 1, projection="3d")
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_zlabel("Z")
                    self.set_camera_view(ax)
                elif attr_name == "2d":
                    ax = self.fig.add_subplot(rows, cols, len(self.attributes) * i + j + 1)
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_xlim([-0.25, 0.25])
                    ax.set_ylim([-0.5, 0.5])
                    ax.invert_yaxis()
                elif attr_name == "ori":
                    ax = self.fig.add_subplot(rows, cols, len(self.attributes) * i + j + 1, projection="polar")
                    # 初始化极坐标图
                    ax.set_rticks([])
                    ax.set_thetagrids(np.arange(0, 360, 45))
                    ax.set_theta_zero_location("E")
                    ax.set_theta_direction(1)
                    ax.grid(True)
                ax.set_title(f"{self.title}_{attr_name}_{i+1}")
                sub_axs[attr_name] = ax
            self.axes.append(sub_axs)

        # plt.ion()
        self.scats = [None] * self.n_cam
        self.lines = [[] for _ in range(self.n_cam)]

        self.scats2d = [None] * self.n_cam
        self.lines2d = [[] for _ in range(self.n_cam)]

        # 关节连接关系
        self.INDICES = [
            (0, 1),
            (1, 2),
            (0, 4),
            (2, 3),
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

        # 颜色定义 (RGB归一化)
        RED = (1.0, 0.0, 0.0)
        BLUE = (0.0, 0.0, 1.0)
        GREEN = (0.0, 1.0, 0.0)

        self.COLORS = [BLUE, BLUE, RED, BLUE, RED, RED, GREEN, GREEN, GREEN, GREEN, RED, RED, RED, BLUE, BLUE, BLUE]

    def update_plot(self, cam_idx, points_3d, title=None):
        if cam_idx >= self.n_cam:
            return

        ax = self.axes[cam_idx].get("3d")
        if not ax:
            return

        # 优化: 使用set_data_3d而非每次创建新对象
        if self.scats[cam_idx] is None:
            self.scats[cam_idx] = ax.scatter([], [], [], c="r", marker="o", s=30)
        self.scats[cam_idx]._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])

        if title:
            ax.set_title(title)

        # 优化: 重用线条对象
        if len(self.lines[cam_idx]) == 0:
            # 创建线条对象
            for ii, idx in enumerate(self.INDICES):
                (line,) = ax.plot([], [], [], c=self.COLORS[ii], linewidth=2.0)
                self.lines[cam_idx].append(line)

        # 更新线条数据
        for ii, idx in enumerate(self.INDICES):
            start, end = idx
            self.lines[cam_idx][ii].set_data_3d(
                [points_3d[start, 0], points_3d[end, 0]],
                [points_3d[start, 1], points_3d[end, 1]],
                [points_3d[start, 2], points_3d[end, 2]],
            )

    def update_plot_2d(self, cam_idx, points_2d, title=None):
        """
        更新指定子图的关键点和骨架结构

        参数:
            cam_idx (int): 要更新的摄像头/子图索引 (从0开始)
            points_2d (numpy.ndarray): 形状为 (17, 2) 的numpy数组，表示17个关键点的2D坐标
            title (str): 可选，更新子图标题
        """
        if cam_idx >= self.n_cam:
            raise ValueError(f"Camera index {cam_idx} out of range (0-{self.n_cam-1})")

        # 获取对应子图
        if "2d" not in self.axes[cam_idx].keys():
            print(f"no 2d ax created for camera {cam_idx}")
            return

        ax = self.axes[cam_idx]["2d"]

        # 清空之前的散点图和连线
        if self.scats2d[cam_idx] is not None:
            self.scats2d[cam_idx].remove()
            self.scats2d[cam_idx] = None
        for line in self.lines2d[cam_idx]:
            line.remove()
        self.lines2d[cam_idx] = []

        # 提取 x, y 坐标
        x = points_2d[:, 0]
        y = points_2d[:, 1]

        # 更新散点图
        if self.scats2d[cam_idx] is None:
            self.scats2d[cam_idx] = ax.scatter([], [], c="r", marker="o", s=30)
        self.scats2d[cam_idx].set_offsets(np.c_[x, y])

        # 更新子图标题
        if title:
            ax.set_title(title)

        # 绘制骨架连接
        if len(self.lines2d[cam_idx]) == 0:
            # 创建线条对象
            for ii, idx in enumerate(self.INDICES):
                (line,) = ax.plot([], [], c=self.COLORS[ii], linewidth=2.0)
                self.lines2d[cam_idx].append(line)

        # 更新线条数据
        for ii, idx in enumerate(self.INDICES):
            start, end = idx
            self.lines2d[cam_idx][ii].set_data(
                [x[start], x[end]],
                [y[start], y[end]],
            )

    def update_ori(self, cam_idx, yaw_ang_rad, title=None):
        if cam_idx >= self.n_cam:
            return

        ax = self.axes[cam_idx].get("ori")
        if not ax:
            return

        # 清除上一次的箭头
        if self.ori_arrows[cam_idx]:
            for arrow in self.ori_arrows[cam_idx]:
                if arrow:
                    arrow.remove()
            self.ori_arrows[cam_idx] = []

        yaw_degrees = np.degrees(yaw_ang_rad) % 360

        # 使用annotate绘制带箭头的线段
        arrow = ax.annotate(
            "",
            xy=(yaw_ang_rad, 1),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red", linewidth=3, mutation_scale=20),
        )
        self.ori_arrows[cam_idx].append(arrow)

        # 添加角度文本
        text = ax.text(
            yaw_ang_rad,
            0.7,
            f"{yaw_degrees:.1f}°",
            ha="center",
            va="center",
            fontsize=14,
            bbox=dict(facecolor="white", alpha=0.8),
        )
        self.ori_arrows[cam_idx].append(text)

        if title:
            ax.set_title(title)

    def show(self, wait_time=0.01):
        """
        刷新整个窗口显示

        参数:
            wait_time (float): 暂停时间，控制更新频率
        """
        # 重绘图形
        self.fig.canvas.draw()
        # 处理事件
        self.fig.canvas.flush_events()
        if wait_time > 0:
            plt.pause(wait_time)

    def set_camera_view(self, ax, elev=-90, azim=-90, fov=90):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev, azim)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-3.0, 0.0])
        ax.set_zlim([0.0, 3.0])

    def save_fig(self, filename="pose_visualization.png"):
        self.fig.savefig(filename, dpi=300, bbox_inches="tight")

    def close(self):
        plt.close(self.fig)
