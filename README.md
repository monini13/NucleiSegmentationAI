# Nuclei Segmentation (CoNSeP Dataset)

Dataset is obtained from: https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet

**Data Preprocessing**

Default image size is 1000x1000 pixels. We pad the images by 12 pixels on each border, resulting in a new image size of 1024x1024.



**Architecture**

Pre-trained ResNet18 from torchvision model, modified to form a UNet.

Input shape: (N, C, 1024,1024), where N = batch size, C = number of channels (3 in our case, since its an RGB image).

Label shape: (N, 1024, 1024), again N=batch size, each pixel is either 0 or 1, where 1 denotes that pixel is part of a nucleus.



**Naive Approach**

Assume only a single class of nuclei. Therefore, the model treats every single nucleus as the same type.

Output shape: (N,1, 1024, 1024), where the *1* symbolizes *one* class



**Metrics**

- BCE across every single pixel
- Dice loss (1-DSC)

Loss will be the weighted average between combined loss of Binary Cross Entropy over every single pixel, and dice loss
$$
loss = \lambda(BCE)+(1-DSC)(1-\lambda)
$$


**Prediction**

Prediction output from model passes through sigmoid function, so each pixel ranges from 0-1. 

Setting the threshold to 0.95 seems to work well, i.e. if pixel >=0.95, pixel = 1, else 0. But best if we can make a slider on the front end that varies the threshold.



<u>**To dos:**</u>

- Class wise segmentation (There are 7 types of nuclei in this dataset)
- Front-end GUI to upload image and display predicted segmentations
- Augment training model. Tinker with loss function, learning rate, etc
- Do report



## References

S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019. [[doi\]](https://doi.org/10.1016/j.media.2019.101563)

UNet referenced from: https://github.com/usuyama/pytorch-unet