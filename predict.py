from PIL import Image
from torchvision import transforms
from model import ResNetUNet
from matplotlib import cm
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def predict(weights_path,img):
    """
    Args:
    weights_path (str): path to where the weights are stored, which will be loaded unto the model
    img (PIL): Nuclei PIL image. Image size = 1000x1000

    Returns:
    Mask (PIL): PIL image detailing the predicted segmentation of the nuclei
    """

    # Prepare image
    transform = transforms.Compose([
            transforms.Pad(12),    # given image is 1000x1000, pad it to make it 1024x1024
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet normalization
            ])

    img = transform(img)
    img = torch.unsqueeze(img,dim=0)


    # Prepare model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_class = 3
    model = ResNetUNet(num_class).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()


    # Predict
    with torch.no_grad():
        outputs = torch.sigmoid(model(img.to(device)))

    pred = outputs.to('cpu').detach().numpy()[0].transpose((1,2,0))
    mask = Image.fromarray(np.uint8(pred*255)).convert('RGB')

    return mask


def get_actual_mask(label):
    """
    Args:
    label (mat): loaded mat files containing labels

    Returns:
    mask (np.array): shape 1024x1024x3 image array of ground truth mask
    """

    mask = np.pad((label['type_map']==1).astype(int),12)
    mask = mask[:, :, None]
    temp = np.pad((label['type_map']==3).astype(int) + (label['type_map']==4).astype(int),12)[:, :, None]
    mask = np.concatenate((mask,temp),axis=2)
    temp = np.pad((label['type_map']==5).astype(int) + (label['type_map']==6).astype(int) + (label['type_map']==7).astype(int),12)[:, :, None]
    mask = np.concatenate((mask,temp),axis=2)
    mask = mask.astype(float)
    mask[mask>=1] = 1

    return mask


if __name__=="__main__":
    img = Image.open('./Test/Images/test_7.png').convert('RGB')
    mask = predict("./weights_3channel_dropout_1",img)  # PIL Image
    mask.show()         # show predicted mask

    label = loadmat('./Test/Labels/test_7.mat')
    true_mask = get_actual_mask(label)

    plt.imshow(true_mask)
    plt.show()          # show actual mask

    ###############################################
    # if want to convert actual mask to PIL image #
    ############################################### 
    # true_mask = Image.fromarray(np.uint8(true_mask*255)).convert('RGB')
    # true_mask.show()