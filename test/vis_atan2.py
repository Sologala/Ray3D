import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np


def visualize_atan2(x, y, title="atan2(y, x) Visualization"):
    """
    可视化atan2(y, x)函数计算的角度

    参数:
    x: float - x坐标值
    y: float - y坐标值
    title: str - 图表标题
    """
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 计算atan2角度（弧度）
    angle_rad = np.arctan2(y, x)
    angle_deg = np.degrees(angle_rad)

    # 第一部分：向量图
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax1.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

    # 绘制向量
    ax1.quiver(0, 0, x, y, angles="xy", scale_units="xy", scale=1, color="blue", width=0.015, label=f"向量 ({x}, {y})")

    # 绘制参考线
    if x != 0:
        ax1.plot([0, x], [0, 0], "r--", alpha=0.5, label="x轴投影")
    if y != 0:
        ax1.plot([0, 0], [0, y], "g--", alpha=0.5, label="y轴投影")

    # 绘制角度弧线
    radius = 0.3
    if x != 0 or y != 0:
        theta = np.linspace(0, angle_rad, 100)
        ax1.plot(radius * np.cos(theta), radius * np.sin(theta), "k-", alpha=0.5)
        # 角度标记
        mid_angle = angle_rad / 2
        label_radius = radius * 1.2
        ax1.text(
            label_radius * np.cos(mid_angle),
            label_radius * np.sin(mid_angle),
            f"{angle_deg:.1f}°",
            ha="center",
            va="center",
        )

    ax1.legend()
    ax1.set_title("向量与角度")
    ax1.set_xlabel("X轴 (向右为正)")
    ax1.set_ylabel("Y轴 (向上为正)")

    # 第二部分：单位圆角度图
    ax2 = fig.add_subplot(122, polar=True)

    # 绘制单位圆
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax2.plot(theta_circle, np.ones_like(theta_circle), "k-", linewidth=1)

    # 绘制角度射线
    ax2.plot([0, angle_rad], [0, 1], "blue", linewidth=2, label=f"角度: {angle_deg:.1f}°")

    # 设置极坐标图属性
    ax2.set_rticks([])  # 不显示径向刻度
    ax2.set_thetagrids(np.arange(0, 360, 45))  # 每45度设置一个刻度
    ax2.set_theta_zero_location("E")  # 0度指向东方(符合atan2的约定)
    ax2.set_theta_direction(1)  # 逆时针方向为正

    # 添加方向标签
    directions = ["0° (东)", "90° (北)", "180° (西)", "270° (南)"]
    for i, dir_label in enumerate(directions):
        angle_rad = np.radians(i * 90)
        ax2.text(angle_rad, 1.1, dir_label, ha="center", va="center", fontsize=10)

    ax2.legend(loc="lower right")
    ax2.set_title("单位圆上的角度")

    # 设置整体标题
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    return fig, (ax1, ax2)


# 使用示例
if __name__ == "__main__":
    # 示例1: 第一象限
    x1, y1 = 1, 1
    fig1, _ = visualize_atan2(x1, y1, "第一象限示例 (x=1, y=1)")

    # 示例2: 第二象限
    x2, y2 = -1, 1
    fig2, _ = visualize_atan2(x2, y2, "第二象限示例 (x=-1, y=1)")

    # 示例3: 第三象限
    x3, y3 = -1, -1
    fig3, _ = visualize_atan2(x3, y3, "第三象限示例 (x=-1, y=-1)")

    # 示例4: 第四象限
    x4, y4 = 1, -1
    fig4, _ = visualize_atan2(x4, y4, "第四象限示例 (x=1, y=-1)")

    plt.show()
