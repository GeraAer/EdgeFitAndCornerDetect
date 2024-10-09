import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
from tqdm import tqdm
from skimage.draw import line
# 失败的用不了的定义带交叉验证的岭回归类
# 我们于是采用了别的办法
# class RidgeRegressionCV:
#     def __init__(self, alphas, n_folds=5):
#         self.alphas = alphas
#         self.n_folds = n_folds
#
#     def fit(self, X, y):
#         best_alpha = None
#         best_score = float('-inf')
#
#         for alpha in self.alphas:
#             fold_size = len(X) // self.n_folds
#             scores = []
#
#             for fold in range(self.n_folds):
#                 val_start = fold * fold_size
#                 val_end = val_start + fold_size
#
#                 X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
#                 y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)
#                 X_val = X[val_start:val_end]
#                 y_val = y[val_start:val_end]
#
#                 # 训练岭回归模型
#                 model = RidgeRegression(alpha)
#                 model.fit(X_train, y_train)
#
#                 # 评估模型性能
#                 score = model.score(X_val, y_val)
#                 scores.append(score)
#
#             avg_score = np.mean(scores)
#
#             if avg_score > best_score:
#                 best_score = avg_score
#                 best_alpha = alpha
#
#         # 使用最佳的 alpha 参数重新训练模型
#         self.best_alpha = best_alpha
#         self.model = RidgeRegression(best_alpha)
#         self.model.fit(X, y)
#
#     def predict(self, X):
#         return self.model.predict(X)

# 预处理图像，将其转换为灰度并归一化
def preprocess_image(image):
    # 使用 PIL 库的 ImageOps 对象将图像转换为灰度图，并进行直方图均衡化
    image = ImageOps.equalize(ImageOps.grayscale(image))
    # 将 PIL 图像对象转换为 NumPy 数组，并进行归一化处理
    return np.array(image) / 255.0

# 检测图像中的边缘点
def detect_edges(image, sigma=0.1):
    # 计算图像的梯度，并将水平和垂直方向上的梯度取绝对值后相加得到边缘图像
    edges = np.abs(np.gradient(image, axis=0)) + np.abs(np.gradient(image, axis=1))
    # 根据设定的阈值 sigma，找出大于阈值的边缘点的位置
    edge_points = np.argwhere(edges > sigma)
    return edge_points

# 绘制结果，包括边缘和角点
def plot_results(processed_image, lines, corners, output_path, filename):
    # 创建一个全零矩阵作为边缘图像
    edge_image = np.zeros(processed_image.shape, dtype=np.uint8)
    # 在边缘图像上根据给定的线段信息画线
    for start, end in lines:
        rr, cc = line(start[0], start[1], end[0], end[1])  # 使用 skimage.draw.line 函数画线
        edge_image[rr, cc] = 255  # 在边缘图像上标记边缘
    # 创建一个图像窗口，显示绘制好的边缘图像
    plt.figure(figsize=(8, 8))
    plt.imshow(edge_image, cmap='gray')  # 绘制灰度图像
    plt.axis('off')  # 关闭坐标轴显示
    # 在图像中标记角点并添加编号
    for i, corner in enumerate(corners):
        plt.scatter(corner[1], corner[0], color='yellow', s=50)  # 在角点位置绘制黄色点
        plt.text(corner[1] + 5, corner[0], str(i), color='red', fontsize=12)  # 添加角点编号
    # 将绘制好的图像保存到指定路径
    save_path = os.path.join(output_path, 'edge_' + filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 保存图像，去除空白边缘
    plt.close()  # 关闭图像窗口

# 处理并保存单张图像的函数
def process_and_save_image(img_path, output_path):
    img = Image.open(img_path)  # 使用 PIL 库打开图像文件
    processed_image = preprocess_image(img)  # 预处理图像，转换为灰度并归一化
    edge_points = detect_edges(processed_image)  # 检测图像边缘点
    # 根据边缘点确定图像的四个角点
    top_left = min(edge_points, key=lambda point: point[0] + point[1])
    top_right = min(edge_points, key=lambda point: point[0] - point[1])
    bottom_left = max(edge_points, key=lambda point: point[0] - point[1])
    bottom_right = max(edge_points, key=lambda point: point[0] + point[1])
    lines = [(top_left, top_right), (top_right, bottom_right),  # 构建图像四边形的边线段
             (bottom_right, bottom_left), (bottom_left, top_left)]
    corners = [top_left, top_right, bottom_right, bottom_left]  # 图像的四个角点
    plot_results(processed_image, lines, corners, output_path, os.path.basename(img_path))  # 绘制并保存结果图像

# 主函数
def main(data_folder):
    output_path = os.path.join(data_folder, 'output')  # 指定结果图像保存的文件夹路径
    if not os.path.exists(output_path):  # 如果结果图像保存的文件夹不存在，则创建
        os.makedirs(output_path)
    image_files = [f for f in os.listdir(data_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]  # 获取数据文件夹中的图像文件列表
    for filename in tqdm(image_files, desc="处理图像"):  # 遍历图像文件列表，并显示处理进度
        img_path = os.path.join(data_folder, filename)  # 拼接图像文件路径
        process_and_save_image(img_path, output_path)  # 处理并保存图像

if __name__ == "__main__":
    main('./data')  # 调用主函数，处理指定文件夹中的图像数据
