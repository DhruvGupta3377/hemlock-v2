import lpips
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import PIL
from PIL import Image
from skimage.metrics import structural_similarity
import cv2
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle


with open("/kaggle/input/tensors/og_data_4.pkl", "rb") as file:
    og_data = pickle.load(file)

with open("/kaggle/input/tensors/sty_data_4.pkl", "rb") as file:
    sty_data = pickle.load(file)

original_train = og_data[2:396]
original_test = og_data[0:2]
stylized_train = sty_data[2:396]
stylized_test = sty_data[0:2]

def compute_scores(emb_one, emb_two, dim=2):
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two, dim=dim)
    return scores

def cloaking(original, styled, similarity_scores):
    cloaked = original.clone()
    for i in range(1,similarity_scores.shape[0]):
        for j in range(1,similarity_scores.shape[1]):
            if similarity_scores[i][j] > 0.17 and similarity_scores[i][j] < 0.29:
                cloaked[0][i][j] = styled[0][i][j]
    return cloaked

class ConvBlock_encoder(nn.Module):
    def __init__(self, input_filters, output_filters):
        super(ConvBlock_encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_filters, output_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_filters)
        self.conv2 = nn.Conv2d(output_filters, output_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x
    
class ConvBlock_decoder(nn.Module):
    def __init__(self, input_filters, output_filters):
        super(ConvBlock_decoder, self).__init__()
        self.conv1 = nn.Conv2d(input_filters, output_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_filters)
        self.conv2 = nn.Conv2d(output_filters, output_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_filters)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, input_filters, output_filters):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock_encoder(input_filters, output_filters)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, input_filters, output_filters):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(input_filters, output_filters, 2, stride=2)
        self.conv_block = ConvBlock_decoder(output_filters+output_filters, output_filters)

    def forward(self, x, skip_features):
        x = self.conv_transpose(x)
        x = torch.cat([x, skip_features], axis=1)
        x = self.conv_block(x)
        return x

class CloakBlock(nn.Module):
    def __init__(self):
        super(CloakBlock, self).__init__()

    def forward(self, original, styled):
        similarity_scores = compute_scores(original[0], styled[0])
        global og
        global st
        og = original
        st = styled
        cloaked = cloaking(original, styled, similarity_scores)
        return cloaked

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.s1 = EncoderBlock(3,64)
        self.s2 = EncoderBlock(64,128)
        self.s3 = EncoderBlock(128,256)
        self.s4 = EncoderBlock(256,512)
        
        self.k1 = EncoderBlock(3,64)
        self.k2 = EncoderBlock(64,128)
        self.k3 = EncoderBlock(128,256)
        self.k4 = EncoderBlock(256,512)
        
        self.b1 = ConvBlock_encoder(512,1024)
        self.b2 = ConvBlock_encoder(512,1024)
        
        self.cloak_block = CloakBlock()
        self.d1 = DecoderBlock(1024,512)
        self.d2 = DecoderBlock(512,256)
        self.d3 = DecoderBlock(256,128)
        self.d4 = DecoderBlock(128,64)
        self.output = nn.Conv2d(64, 3, 1, padding=0)

    def forward(self, input1, input2):
        # Original image
        s1, p1 = self.s1(input1)
        s2, p2 = self.s2(p1)
        s3, p3 = self.s3(p2)
        s4, p4 = self.s4(p3)
        b1 = self.b1(p4)

        # Styled image
        k1, q1 = self.k1(input2)
        k2, q2 = self.k2(q1)
        k3, q3 = self.k3(q2)
        k4, q4 = self.k4(q3)
        b2 = self.b2(q4)

        # Cloaked features
        b3 = self.cloak_block(b1, b2)

        # Image reconstruction
        d1 = self.d1(b3, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        output = self.output(d4)
        return output

def psnr_loss(output, target, max_pixel_value=1.0):
    mse = F.mse_loss(output, target)
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return -psnr 

lpips_loss = lpips.LPIPS(net='vgg')

def loss_fun(cloaked, original):
    lpips_val = lpips_loss(cloaked, original)
    psnr_val = psnr_loss(cloaked, original)
    net_loss = 0.65*lpips_val + 0.35*psnr_val
    return net_loss

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = True
    
    model.eval()
    
    return model

model = load_checkpoint('/kaggle/input/hemlock/pytorch/v4/1/checkpoint_LPIPS_PSNR_4.pth')
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
lpips_loss = lpips_loss.to(device)
original_train = [tensor.to(device) for tensor in original_train]
original_test = [tensor.to(device) for tensor in original_test]
stylized_train = [tensor.to(device) for tensor in stylized_train]
stylized_test = [tensor.to(device) for tensor in stylized_test]

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i in range(len(original_train)):
        # Forward pass
        outputs = model(original_train[i], stylized_train[i]) 
        loss = loss_fun(outputs, original_train[i]) 
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        optimizer.step()
#         print("Done",i)
        running_loss += loss.item()

    epoch_loss = running_loss / len(original_train)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

checkpoint = {'model': UNet(),
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint_LPIPS_PSNR_5.pth')




# MODEL TILL HERE. LATER IS MODEL TESTING & EVALUATION



class Denormalize(transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / std
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        tensor = super().__call__(tensor.clone())
        return tensor

ImgTransform = transforms.Compose([
    Denormalize((0.5,), (0.5,)),
    transforms.ToPILImage()
])

output_img = model(original_test[0], stylized_test[0])
output_img = torch.squeeze(output_img)
output_img = ImgTransform(output_img)

sty_img = stylized_test[0]
sty_img = torch.squeeze(sty_img)
sty_img = ImgTransform(sty_img)

input_img = original_test[0]
input_img = torch.squeeze(input_img)
input_img = ImgTransform(input_img)

display(input_img)
display(sty_img)
display(output_img)

from matplotlib import pyplot as plt 
  
# create figure 
fig = plt.figure(figsize=(15, 10)) 
  
# setting values to rows and column variables 
rows = 1
columns = 3
  
fig.add_subplot(rows, columns, 1) 
  
# showing image 
plt.imshow(input_img) 
plt.axis('off') 
plt.title("Original") 
  
# Adds a subplot at the 2nd position 
fig.add_subplot(rows, columns, 2) 
  
# showing image 
plt.imshow(sty_img) 
plt.axis('off') 
plt.title("Stylized") 
  
# Adds a subplot at the 3rd position 
fig.add_subplot(rows, columns, 3) 
  
# showing image 
plt.imshow(output_img) 
plt.axis('off') 
plt.title("Cloaked")

input_img.save('original.png')
output_img.save('cloaked.png')

img1 = Image.open('/kaggle/working/original.png').convert('RGB')
img2 = Image.open('/kaggle/working/cloaked.png').convert('RGB')
img2 = img2.resize(img1.size)

img1_tensor = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1
img2_tensor = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1

loss_fn = lpips.LPIPS(net='alex')
lpips_score = loss_fn(img1_tensor, img2_tensor)
print("LPIPS: ",lpips_score)

psnr_score = -psnr_loss(img1_tensor, img2_tensor)
print("PSNR: ",psnr_score)

img1_gray = np.array(img1.convert('L'))
img2_gray = np.array(img2.convert('L'))
img1_np = np.array(img1)
img2_np = np.array(img2)

(ssim_score, ssim_diff) = structural_similarity(img1_gray, img2_gray, full=True)
print("SSIM", ssim_score)

ssim_diff = (ssim_diff * 255).astype("uint8")

thresh = cv2.threshold(ssim_diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(img1_np.shape, dtype='uint8')
filled_after = img2_np.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img1_np, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(img2_np, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

fig = plt.figure(figsize=(15, 10)) 
rows = 2
columns = 3
  
fig.add_subplot(rows, columns, 1) 
plt.title('Orignal')
plt.imshow(img1_np)
plt.axis('off')

fig.add_subplot(rows, columns, 2) 
plt.title('Cloaked')
plt.imshow(img2_np)
plt.axis('off')

fig.add_subplot(rows, columns, 3) 
plt.title('Diff')
plt.imshow(ssim_diff)
plt.axis('off')

fig.add_subplot(rows, columns, 4) 
plt.title('Mask')
plt.imshow(mask)
plt.axis('off')

fig.add_subplot(rows, columns, 5) 
plt.title('Filled after')
plt.imshow(filled_after)
plt.axis('off')
