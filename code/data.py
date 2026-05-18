import kagglehub
import shutil #shutil 是 Python 自带的标准库，无需额外安装，主要用来操作文件 / 文件夹。
import os

#===============step1:数据下载======================

# Download latest version
path=kagglehub.dataset_download("zoltanszekely/mini-traffic-detection-dataset")
print("Path to dataset files:", path)
#现在是.jpg格式。不属于纯张量、Image 或 PIL 图像中的任何一种。

#我不想它把数据下到C盘。但是kagglehub默认不支持直接指定路径。遂用移动文件夹的方式实现。

# 自定义目标保存路径
target_path = r"/\data"
#加个"r",使反斜杠不会进行转义

# 遍历下载的文件，移动到指定路径
for item in os.listdir(path):
    src = os.path.join(path, item)    # 源文件路径
    dst = os.path.join(target_path, item) # 目标路径
    shutil.move(src, dst)

print("已移动到指定路径:", target_path)