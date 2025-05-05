import matplotlib.pyplot as plt
import math

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 18

import numpy as np


def visualize_yaw(yaw_angle, title="Target Orientation"):
    """
    将yaw角度绘制在一个带有刻度的圆环内

    参数:
    yaw_angle: float - 偏航角（弧度）
    title: str - 图表标题
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 将弧度转换为度数
    yaw_degrees = np.degrees(yaw_angle)

    # 确保角度在0-360度范围内
    yaw_degrees = yaw_degrees % 360

    # 绘制方向标记
    ax.plot([0, yaw_angle], [0, 1], "r-", linewidth=3)
    ax.annotate(
        "",  # 不显示文本
        xy=(yaw_angle, 1),  # 箭头终点
        xytext=(0, 0),  # 箭头起点
        arrowprops=dict(arrowstyle="->", color="red", linewidth=3, mutation_scale=20),  # 控制箭头大小
    )

    # # 绘制圆形边界
    # theta = np.linspace(0, 2 * np.pi, 36)
    # ax.plot(theta, np.ones_like(theta), "k-", linewidth=2)

    # 设置刻度和标签
    ax.set_rticks([])  # 不显示径向刻度
    ax.set_thetagrids(np.arange(0, 360, 45))  # 每45度设置一个刻度
    ax.set_theta_zero_location("E")  # 0度指向北方
    ax.set_theta_direction(1)  # 逆时针方向为正

    # 添加方向标签
    # directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    # for i, dir_label in enumerate(directions):
    #     angle_rad = np.radians(i * 45)
    #     ax.text(angle_rad, 1.1, dir_label, ha="center", va="center", fontsize=12)

    # 添加角度数值
    ax.text(
        yaw_angle,
        0.7,
        f"{yaw_degrees:.1f}°",
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # 设置标题
    ax.set_title(title, fontsize=16, pad=20)

    # 显示网格线
    ax.grid(True)

    return fig, ax


# 使用示例
if __name__ == "__main__":
    # 示例：面向东北方向
    yaw = np.radians(45)
    fig, ax = visualize_yaw(yaw, "example")
    plt.show()
