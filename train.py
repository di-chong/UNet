import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image
from dataset.data_prepare import *
from unet import *
from torch import nn, optim
from torchvision.utils import save_image


# 1.设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# 2.准备数据
para_path = "D:/Python Program/UNet/paras/my_checkpoint.pth"
data_path = "D:/Python Program/UNet/dataset/carvana_dataset/"
save_path = "save_image"
generator = torch.Generator().manual_seed(32)    # 创建一个生成器实例并设置种子
data = Mydata(data_path, 200)
train_dataset, test_dataset = random_split(data, [0.6, 0.4], generator=generator)
test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)
train_data_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=5, shuffle=True)

# 3.模型训练
net = Unet(3, 3).to(device)
opt = optim.Adam(net.parameters(), lr=3e-4)
loss_fn = nn.BCEWithLogitsLoss()

if os.path.exists(para_path):
    net.load_state_dict(torch.load(para_path))
    print("successfully load para")
else:
    print("not successfully load para")

epoch = 500
train_losses = []
val_losses = []
for i in tqdm(range(epoch)):
    net.train()
    train_running_loss = 0
    for j, (img, img_mask) in enumerate(tqdm(train_data_loader)):
        img, img_mask = img.to(device), img_mask.to(device)
        out = net(img)
        loss = loss_fn(out, img_mask)
        train_running_loss += loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()

        # 将输入图片，标签，以及输出合并在一张图片上看看效果
        _img = img[0]
        _img_mask = img_mask[0]
        _out = out[0]
        save_img = torch.stack([_img, _img_mask, _out], dim=0)
        save_image(save_img, f"{save_path}/{i}_{j}.png")

    train_loss = train_running_loss / (j + 1)
    train_losses.append(train_loss)
    if i % 10 == 0:
        with torch.no_grad():
            net.eval()
            val_running_loss = 0
            val_running_dc = 0
            for idx, img_mask in enumerate(tqdm(val_data_loader, position=0, leave=True)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = net(img)
                loss = loss_fn(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        val_losses.append(val_loss)
    # 根据自己的需要设置参数存储轮次
    if i % 1 == 0:
        torch.save(net.state_dict(), para_path)
    print("-" * 30)
    print(f"Training Loss EPOCH {i + 1}: {train_loss:.4f}")
    print("\n")
    print(f"Validation Loss EPOCH {i + 1}: {val_loss:.4f}")
    print("-" * 30)




