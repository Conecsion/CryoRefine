import torch
from typing import Tuple
import torch.nn as nn
from torchvision.models import swin_t
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import cv2
from tifffile import imread

device = 'cuda'


class ProjectionDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 img_dir,
                 transform=None,
                 target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # print(img_path)
        # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = imread(img_path)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        # print(image.size())
        ang1 = float(self.img_labels.iloc[idx, 1])
        ang2 = float(self.img_labels.iloc[idx, 2])
        ang3 = float(self.img_labels.iloc[idx, 3])
        label = torch.tensor([ang1, ang2, ang3])
        return image, label

training_data = ProjectionDataset('../../../data/train/labels.csv',
                                  '../../../data/train/')
train_dataloader = DataLoader(training_data, batch_size=12, shuffle=True)
# a, b = next(iter(train_dataloader))
# print(a.size())
# print(b.size())


class Encoder(torch.nn.Module):

    def __init__(self, image_size: Tuple[int, int, int] = (1, 338, 338)):
        super(Encoder, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(in_channels=self.image_size[0],
                               out_channels=3,
                               kernel_size=1)
        self.swin = swin_t(weights='DEFAULT')

        # Optional
        self.head = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.swin(x)
        # Optional
        x = self.head(x)
        return x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class OrientEmb(nn.Module):
    """
    Embeddings of protein particle orientations and shifts
    """

    def __init__(self, angles, dim_head=1000):
        super().__init__()
        scale = dim_head**-0.5
        # self.angles = None


model = Encoder().to(device)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if i % 1000 == 999:
        if i:
            last_loss = running_loss / 10
            print('  batch {} loss: {}'.format(i, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            running_loss = 0.

    return last_loss


# timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
epoch_number = 0

EPOCHS = 500
for epoch in range(EPOCHS):
    print('Epoch {}'.format(epoch_number + 1))
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, None)
    # running_vloss = 0.

    # model.eval()
    # with torch.no_grad():
    #     for i, vdata in enumerate(validation_loader)
    model_path = 'model_{}'.format(epoch_number)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join('./checkpoints/', model_path))
    epoch_number += 1
