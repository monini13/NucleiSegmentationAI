from utils import NucleiDataset, reverse_transform
import numpy as np
from model import ResNetUNet
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def dice(pred,targ,smooth=1/1e32):
    intersection = (pred * targ).sum(dim=2).sum(dim=2)
    score = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + targ.sum(dim=2).sum(dim=2) + smooth)
    return score

def jaccard(pred,targ,smooth=1/1e32):
    intersection = (pred * targ).sum(dim=2).sum(dim=2)
    score = (intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + targ.sum(dim=2).sum(dim=2) - intersection + smooth)
    return score

def eval(model, test_loader, thresholds = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)):
    model.eval()

    for i, (inputs, masks) in enumerate(test_loader):
        with torch.no_grad():
            outputs = torch.sigmoid(model(inputs.to(device)))

        for j,threshold in enumerate(thresholds):
            pred = outputs.to('cpu')
            pred[pred >= threshold] = 1
            pred[pred < threshold] = 0

            if j == 0:
                dice_score = dice(pred,masks)
                jaccard_score = jaccard(pred,masks)
            else:
                dice_score += dice(pred,masks)
                jaccard_score += jaccard(pred,masks)
        
        if i == 0:
            avg_dice_score = dice_score/len(thresholds)
            avg_jaccard_score = jaccard_score/len(thresholds)
        else:
            avg_dice_score += dice_score/len(thresholds)
            avg_jaccard_score += jaccard_score/len(thresholds)

    avg_dice_score /= len(test_loader)
    avg_jaccard_score /= len(test_loader)

    return avg_dice_score, avg_jaccard_score



if __name__ == "__main__":

    # Set up model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNetUNet(3)
    model.load_state_dict(torch.load("./weights_3channel_dropout_1"))
    model.to(device)

    # Test Set Loader
    label_path = "./Test/Labels/"
    img_path = "./Test/Images/"
    trans = transforms.Compose([
            transforms.Pad(12),    # given image is 1000x1000, pad it to make it 1024x1024
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet normalization
        ])
    test_set = NucleiDataset(img_path,label_path, transform = trans)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)

    dice_score,jaccard_score = eval(model,test_loader)
    

    print("dice:\t",dice_score.numpy()[0], " \tmean:",dice_score.mean().item())            # RGB
    print("jaccard:",jaccard_score.numpy()[0], " \tmean:",jaccard_score.mean().item())      # RGB






