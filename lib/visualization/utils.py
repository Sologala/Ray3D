import numpy as np
import cv2
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt

def draw_kps_to_image(normalized_kps: np.ndarray, width = 416, height = 416):

    # 关节连接关系
    INDICES = [
        (0, 1), (1, 2), (0, 4), (2, 3), (4, 5), (5, 6), 
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), 
        (12, 13), (8, 14), (14, 15), (15, 16)
    ]
    
    # 颜色定义 (RGB归一化)
    RED = (1.0, 0.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    GREEN = (0.0, 1.0, 0.0)
    
    COLORS = [BLUE, BLUE, RED, BLUE, RED, RED, GREEN, GREEN, 
              GREEN, GREEN, RED, RED, RED, BLUE, BLUE, BLUE]
    
    # 缩放归一化点以适应图像大小
    border = 10
    # 找到最大范围并缩放，使所有点都在图像内
    f = np.max(np.abs(normalized_kps)) / (0.5 * (min(width, height) - border))
    # print("f", f)
    pts = normalized_kps / f  # 注意这里应该是除法而不是乘法
    # print(pts) 
    # 将坐标原点移到图像中心，并调整Y轴方向（图像Y轴向下为正）
    pts[:, 0] = pts[:, 0] + width / 2
    pts[:, 1] = pts[:, 1] + height / 2  # 负号是因为图像Y轴方向与标准坐标系相反
    
    # 转换为整数坐标
    pts = np.round(pts).astype(int)
    
    # 创建空白图像
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 绘制骨架连接
    for i, (a, b) in enumerate(INDICES):
        color = tuple(int(c * 255) for c in COLORS[i])  # 转换为OpenCV的BGR格式
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    
    # 绘制关键点
    for pt in pts:
        cv2.circle(img, tuple(pt), 5, (255, 255, 255), -1, cv2.LINE_AA)  # 白色圆点
    return img 

def create_kps_animation(kps_sequence: np.ndarray, fps=10, width=416, height=416, show=True):
    """
    将关键点序列生成动画并显示
    
    参数:
        kps_sequence: 关键点序列，形状为 (batch=1, frames, points=17, dim=2) 或 (frames, points=17, dim=2)
        fps: 动画帧率
        width, height: 图像尺寸
        show: 是否显示动画
    
    返回:
        matplotlib.animation.FuncAnimation 对象
    """
    # 去除batch维度
    if kps_sequence.ndim == 4 and kps_sequence.shape[0] == 1:
        kps_sequence = kps_sequence[0]  # 形状变为 (frames, 17, 2)
    
    num_frames = kps_sequence.shape[0]
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    
    # 初始化图像
    img = draw_kps_to_image(kps_sequence[0], width, height)
    im = ax.imshow(img)
    
    # 更新函数
    def update(frame):
        img = draw_kps_to_image(kps_sequence[frame], width, height)
        cv2.putText(img, f'Frame {frame+1}/{num_frames}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        im.set_array(img)
        return [im]
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=True)
    
    if show:
        plt.close()  # 避免显示静态图像
        return HTML(ani.to_jshtml())  # 用于Jupyter环境
    
    return ani

def draw_orientation(image, yaw_ang_rad, cam_idx=0, title=None, 
                     canvas_size=(416, 416), arrow_color=(0, 0, 255), 
                     text_color=(0, 0, 0)):
    """
    使用OpenCV在图像上绘制朝向箭头和角度
    
    参数:
        image: 输入图像（如果为None，则创建空白图像）
        yaw_ang_rad: 偏航角（弧度）
        cam_idx: 相机索引（用于标识不同相机视图）
        title: 标题文本
        canvas_size: 画布尺寸
        arrow_color: 箭头颜色 (BGR格式)
        text_color: 文本颜色 (BGR格式)
    
    返回:
        绘制后的图像
    """
    # 如果没有提供图像，创建一个空白画布
    if image is None:
        image = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255  # 白色背景
    
    # 获取图像尺寸
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) * 0.4  # 箭头长度
    
    # 计算箭头终点坐标
    yaw_degrees = np.degrees(yaw_ang_rad) % 360
    end_x = int(center_x + radius * np.cos(yaw_ang_rad))
    end_y = int(center_y + radius * np.sin(yaw_ang_rad))
    
    # 绘制箭头
    image = cv2.arrowedLine(
        image, 
        (center_x, center_y), 
        (end_x, end_y), 
        color=arrow_color, 
        thickness=3, 
        line_type=cv2.LINE_AA,
        tipLength=0.15  # 箭头头部长度比例
    )
    
    # 绘制角度文本
    text = f"{yaw_degrees:.1f}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = int(center_x + radius * 0.7 * np.cos(yaw_ang_rad))
    text_y = int(center_y + radius * 0.7 * np.sin(yaw_ang_rad))
    
    # 添加文本背景
    text_bg_x1, text_bg_y1 = text_x - text_size[0] // 2 - 10, text_y - text_size[1] // 2 - 10
    text_bg_x2, text_bg_y2 = text_x + text_size[0] // 2 + 10, text_y + text_size[1] // 2 + 10
    cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), 
                 color=(255, 255, 255), thickness=-1)  # 白色背景
    
    # 添加文本
    cv2.putText(image, text, (text_x - text_size[0] // 2, text_y + text_size[1] // 2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    
    # 添加标题
    if title:
        cv2.putText(image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    # 添加相机索引
    cv2.putText(image, f"Camera {cam_idx}", (width - 120, height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    return image