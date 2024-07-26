import os
from util import *
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

from torch.utils.data import Dataset

path = "D:/Python Program/UNet/dataset/carvana_dataset/"
# images = sorted([path + 'image/' + i for i in os.listdir(os.path.join(path, 'image'))][:10])
# images_mask = sorted([path + 'mask_image/' + i for i in os.listdir(os.path.join(path, 'mask_image'))][:10])
# images_mask = sorted(os.listdir(os.path.join(path, 'mask_image')))[:10]
# # print(images)
# print(images_mask)
# img = Image.open(images[8]).convert("RGB")
# path1 = os.path.join(path, 'mask_image')
# img_m = Image.open(os.path.join(path1, images_mask[8])).convert("L")
# # # 展示图片
# img.show()
# img_m.show()

class Mydata(Dataset):
    def __init__(self, path, limit=10):
        self.path = path
        self.limit = limit

    def __len__(self):
        return min(len(os.listdir(os.path.join(self.path, 'image'))), self.limit)

    def __getitem__(self, index):
        # 得到每张图片的地址
        image_name = os.listdir(os.path.join(self.path, 'image'))[index]
        segment_name = os.listdir(os.path.join(self.path, 'mask_image'))[index]
        image_path = os.path.join(self.path, 'image', image_name)
        segment_path = os.path.join(self.path, 'mask_image', segment_name)
        # 对图片进行转换，统一大小
        img_new = keep_size(image_path)
        segment_new = keep_size(segment_path)
        # return img_new, segment_new
        return transform(img_new), transform(segment_new)

if __name__ == '__main__':
    data = Mydata("D:/Python Program/UNet/dataset/carvana_dataset/")
    data[5][0].show()
    data[5][1].show()








