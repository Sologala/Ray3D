import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus' ] = False
# 设置中文字体

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_aspect('equal')  # 确保坐标轴比例相等，使圆看起来是圆的
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_title('圆周运动')
ax.grid(True)

# 绘制静态的圆
circle = plt.Circle((0, 0), 1, fill=False, color='blue', linestyle='-', linewidth=2)
ax.add_patch(circle)

# 初始化运动点
point, = ax.plot([], [], 'ro', markersize=10)  # 红色圆点

# 初始化文本显示角度
angle_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

# 动画更新函数
def update(frame):
    # 计算当前角度（弧度）
    angle = np.radians(frame)
    # 计算圆上点的坐标
    x = np.cos(angle)
    y = np.sin(angle)
    # 更新点的位置
    point.set_data(x, y)
    # 更新角度显示
    angle_text.set_text(f'角度: {frame:.1f}°')
    return point, angle_text

# 创建动画
from matplotlib.animation import FuncAnimation

ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 180), 
                    interval=10, blit=True, repeat=True)

plt.tight_layout()
plt.show()

# 如果需要保存动画，可以取消下面的注释
# ani.save('circle_animation.gif', writer='pillow', fps=30)    
