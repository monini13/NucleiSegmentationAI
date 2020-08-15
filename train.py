import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from model import ResNetUNet
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from evaluate import eval
from torch.utils.data import DataLoader
from utils import NucleiDataset, reverse_transform

def weighted_loss(pred,targ,bce_weight=0.5, smooth=1.):
    
    bce = F.binary_cross_entropy_with_logits(pred, targ)
    pred = torch.sigmoid(pred)
    
    pred = pred.contiguous()
    targ = targ.contiguous()  

    intersection = (pred * targ).sum(dim=2).sum(dim=2)
    dice = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + targ.sum(dim=2).sum(dim=2) + smooth)))
    loss = bce * bce_weight + dice.mean() * (1 - bce_weight)
    
    return loss

def train_model(model, optimizer, scheduler, device, num_epochs=25):
    best_loss = 1e10
    
    loss_ls = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()
        
        loss = 0
        
        for inputs,masks in train_loader:
            model.train()
            inputs = inputs.to(device)
            masks = masks.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            epoch_loss = weighted_loss(outputs,masks,bce_weight=0.5)
            epoch_loss.backward()
            optimizer.step()
            
            n_samples = len(inputs)
            loss+=(epoch_loss/n_samples).item()
        
        loss /= len(train_loader)
        print("epoch loss:",loss)
        loss_ls.append(loss)
        
        # if (epoch+1)%5==0:
        #     print("\n","Input Image")
        #     plt.imshow(reverse_transform(inputs.to('cpu').detach()[0]))
        #     plt.show()
        #     print("Predicted Mask Sigmoid")
        #     plt.imshow(torch.sigmoid(outputs).to('cpu').detach().numpy()[0].transpose((1,2,0)))
        #     plt.show()
        #     print("Actual Mask")
        #     plt.imshow(masks.to('cpu').detach().numpy()[0].transpose((1,2,0)))
        #     plt.show()
            
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60),"\n")
    
    return loss_ls

if __name__ == "__main__":

    # Set up model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_class = 3
    model = ResNetUNet(num_class).to(device)
    model.train()

    # trainloader
    label_path = "./Train/Labels/"
    img_path = "./Train/Images/"
    trans = transforms.Compose([
            transforms.Pad(12),    # given image is 1000x1000, pad it to make it 1024x1024
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet normalization
        ])
    train_set = NucleiDataset(img_path,label_path, transform = trans)
    batch_size = 1  #my gpu is 8gb, using batchsize of 2 already insufficient memory, so i use batch size 1
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)

    # train
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)
    loss_ls = train_model(model, optimizer_ft, exp_lr_scheduler, device,num_epochs=30)

    # testloader
    label_path = "./Test/Labels/"
    img_path = "./Test/Images/"
    test_set = NucleiDataset(img_path,label_path, transform = trans)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)
    
    # evaluate
    dice_score,jaccard_score = eval(model,test_loader,device)
    

    print("dice:\t",dice_score.numpy()[0], " \tmean:",dice_score.mean().item())            # RGB, mean
    print("jaccard:",jaccard_score.numpy()[0], " \tmean:",jaccard_score.mean().item())     # RGB, mean