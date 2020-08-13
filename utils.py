from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from scipy.io import loadmat
import os

class NucleiDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None):
        self.img_ls = [image_path+i for i in sorted(os.listdir(image_path))]
        self.mask_ls = [ mask_path+i for i in sorted(os.listdir(mask_path))]
        self.transform = transform

    def __len__(self):
        return len(self.img_ls)

    def __getitem__(self, idx):
        img_name = self.img_ls[idx]
        img = Image.open(img_name).convert('RGB')
        img.load()
        mask_name = self.mask_ls[idx]
        x = loadmat(mask_name)
        mask = np.pad((x['type_map']==1).astype(int) + (x['type_map']==2).astype(int),12)
        mask = mask[:, :, None]
        temp = np.pad((x['type_map']==3).astype(int) + (x['type_map']==4).astype(int),12)[:, :, None]
        mask = np.concatenate((mask,temp),axis=2)
        temp = np.pad((x['type_map']==5).astype(int) + (x['type_map']==6).astype(int) + (x['type_map']==7).astype(int),12)[:, :, None]
        mask = np.concatenate((mask,temp),axis=2)
        mask = mask.astype(float)
        mask[mask>=1] = 1
        
        if self.transform:
            img = self.transform(img)

        return img, mask.transpose((2, 0, 1))

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp