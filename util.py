from PIL import Image
import os
import numpy as np

# 先找到最长边，构建背景正方形，再将图片复制上去，再等比例压缩成256*256
def keep_size(path, size=(256,256)):
    img = Image.open(path)
    # # 将PIL图片转换为NumPy数组
    # img_np = np.array(img)
    # print(img_np.shape)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


if __name__ == '__main__':
    path = "D:/Python Program/UNet/dataset/carvana_dataset/mask_image/"
    img_list = os.listdir(path)
    path = (os.path.join(path, img_list[5]))
    mask_test = keep_size(path).show()
