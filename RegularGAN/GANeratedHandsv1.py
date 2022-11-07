#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

print("torch.__version__", torch.__version__)

# Allow for GPU use
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

config={
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 16,
    "NUM_EPOCHS": 1000,
    "PIN_MEMORY": True,
    "TRAIN_SPLIT": 0.8,
    "CPU_COUNT": int(os.cpu_count()),
    "TARGET_IMG_SIZE": (256, 256),
    "PIN_MEMORY": True if device == 'cuda' else False
}

class CenterCropImageDataset(Dataset):
    def __init__(self, dataset_dir):
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if '.png' in x]

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index])
        image_as_tensor = T.ToTensor()(image)
        resized_tensor = T.Resize((286, 286), interpolation=T.InterpolationMode.BICUBIC, antialias=True)(image_as_tensor)
        resized_and_cropped = T.CenterCrop((256, 256))(resized_tensor)
        return resized_and_cropped
    
    def __len__(self):
        return len(self.image_filenames)


synth_hands_dataset = CenterCropImageDataset("./synth_all_male_noobject_with_white_bg")
real_hands_dataset = CenterCropImageDataset("./RealHandsForKagglev2")

real_train_dataset, real_val_dataset = torch.utils.data.random_split(real_hands_dataset, [round(len(real_hands_dataset)*config['TRAIN_SPLIT']), round(len(real_hands_dataset)*(1-config['TRAIN_SPLIT']))])
synth_train_dataset, synth_val_dataset = torch.utils.data.random_split(synth_hands_dataset, [round(len(synth_hands_dataset)*config['TRAIN_SPLIT']), round(len(synth_hands_dataset)*(1-config['TRAIN_SPLIT']))])

real_train_dataloader = DataLoader(real_train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["CPU_COUNT"], pin_memory=config["PIN_MEMORY"], drop_last=True)
real_val_dataloader = DataLoader(real_val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["CPU_COUNT"], pin_memory=config["PIN_MEMORY"], drop_last=True)
synth_train_dataloader = DataLoader(synth_train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["CPU_COUNT"], pin_memory=config["PIN_MEMORY"], drop_last=True)
synth_val_dataloader = DataLoader(synth_val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["CPU_COUNT"], pin_memory=config["PIN_MEMORY"], drop_last=True)



# In[8]:


def discriminatorLayer(in_feat, out_feat, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=4, stride=stride),
        nn.BatchNorm2d(out_feat),
        nn.LeakyReLU(0.2)
    )

def generateLayerList(n):
    convList = []
    for i in range(n):
        convList.append(discriminatorLayer(64*2**i,64*2**(i+1), 2))
    return nn.ModuleList(convList)

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2)
        self.leaky = nn.LeakyReLU(0.2)
        self.convList = generateLayerList(3)
        self.last_conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.leaky(self.conv1(x))
        for conv in self.convList:
            x = conv(x)
        x = self.last_conv(x)
        x = self.sigmoid(x)
        return torch.mean(x)


# In[9]:


gen = torchvision.models.resnet50(pretrained=True)
print(gen)


# In[10]:


class Conv2dTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


# In[11]:


class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.res_blocks = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:6])
        self.upsample_blocks = nn.Sequential(Conv2dTransposeBlock(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                                             Conv2dTransposeBlock(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                             Conv2dTransposeBlock(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.last = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.res_blocks(x)
        x = self.upsample_blocks(x)
        x = self.last(x)
        x = self.sigmoid(x)
        return x


gen = Generator().to(device)
print(gen)


# In[12]:


disc = Discriminator().to(device)
opt_disc = torch.optim.Adam(disc.parameters(), lr=config["LEARNING_RATE"])


opt_gen = torch.optim.Adam(gen.parameters(), lr=config["LEARNING_RATE"])

elapsed_epochs = 0
train_G_losses, train_real_D_losses, train_fake_D_losses, train_acc = [], [], [], []
val_G_losses, val_real_D_losses, val_fake_D_losses, val_acc = [], [], [], []


# In[24]:


# Training function 
bar_string = "-"*75
REAL_LABEL = 1
FAKE_LABEL = 0

l1_loss_function = nn.L1Loss()

def gen_loss(original, ganerated, discriminator):
    adverserial_loss = torch.mean(-torch.log10(discriminator(ganerated)))
    l1_loss = l1_loss_function(original, ganerated)
    loss = adverserial_loss + l1_loss * 0.2
    return loss


def discriminator_loss(preds, label):
    loss_fun = nn.BCELoss()
    size = preds.size(0)
    label_tensor = torch.full((size,), label, dtype=torch.float, device=device)
    loss = loss_fun(preds, label_tensor)
    return loss

def apply_threshold(y_pred, threshold=0.5):
    y_prediction = torch.zeros(y_pred.size(), device=device)
    y_prediction[y_pred >= threshold] = 1
    y_prediction = y_prediction.squeeze()
    return y_prediction

def accuracyCalc(preds, label):
    size = preds.size(0)
    labels = torch.full((size,), label, dtype=torch.float, device=device)
    acc = ((apply_threshold(preds) == labels).sum().item())/size
    return acc



# In[25]:


def printResults(G_losses, real_D_losses, fake_D_losses, acc_vals, mode):
    mean_G_loss = np.mean(np.array(G_losses))
    real_mean_D_loss = np.mean(np.array(real_D_losses))
    fake_mean_D_loss = np.mean(np.array(fake_D_losses))
    mean_acc = np.mean(np.array(acc_vals))
    
    if mode=="Training":
        train_G_losses.append(mean_G_loss)
        train_real_D_losses.append(real_mean_D_loss)
        train_fake_D_losses.append(fake_mean_D_loss)
        train_acc.append(mean_acc)
        
    elif mode=="Validation":
        val_G_losses.append(mean_G_loss)
        val_real_D_losses.append(real_mean_D_loss)
        val_fake_D_losses.append(fake_mean_D_loss)
        val_acc.append(mean_acc)
    
    print(mode+f" Generator loss: {mean_G_loss:>7f}\n"+
          mode+f" Discriminator fake loss: {fake_mean_D_loss:>7f} | real loss: {real_mean_D_loss:>7f} | Accuracy: {mean_acc:>7f}\n"
          +bar_string)


# In[26]:


# Training function
def train(real_dataloader, synth_dataloader, generator, discriminator, gen_optimizer, disc_optimizer):
    num_batches = min(len(real_dataloader), len(synth_dataloader))
    print("num_batches=", num_batches)
    
    G_losses, fake_D_losses, real_D_losses, acc_vals = [], [], [], []
    
    discriminator.train()
    generator.train()
    
    for i, (real_batch, synth_batch) in enumerate(zip(real_dataloader, synth_dataloader)): # zip() will use the minimum available size
        real_batch, synth_batch = real_batch.to(device, non_blocking=config["PIN_MEMORY"]), synth_batch.to(device, non_blocking=config["PIN_MEMORY"])
        real_batch = T.RandomRotation(180)(real_batch)
        
        disc_optimizer.zero_grad()
        gen_optimizer.zero_grad()
        
        # Train with all real data
        disc_preds = discriminator(real_batch).view(-1)
        real_D_loss = discriminator_loss(disc_preds, REAL_LABEL)
        
        real_D_loss.backward()
        real_D_losses.append(real_D_loss.detach().cpu())
        
        acc = accuracyCalc(disc_preds, REAL_LABEL)
        acc_vals.append(acc)
        
        ganerated_batch = generator(synth_batch)
        ganerated_batch = T.RandomRotation(180)(ganerated_batch)
        # Train with all fake data
        disc_preds = discriminator(ganerated_batch.detach()).view(-1)
        fake_D_loss = discriminator_loss(disc_preds, FAKE_LABEL)
        
        fake_D_loss.backward()
        fake_D_losses.append(fake_D_loss.detach().cpu())
        
        acc = accuracyCalc(disc_preds, FAKE_LABEL)
        acc_vals.append(acc)
        
        disc_optimizer.step()
        
        G_loss = gen_loss(synth_batch, ganerated_batch, discriminator)
        G_losses.append(G_loss.item())
        G_loss.backward()
        
        gen_optimizer.step()
    printResults(G_losses, real_D_losses, fake_D_losses, acc_vals, "Training")

# Evaluation function
def evaluate(real_dataloader, synth_dataloader, generator, discriminator):
    num_batches = min(len(real_dataloader), len(synth_dataloader))
    print("num_batches=", num_batches)
    
    G_losses, fake_D_losses, real_D_losses, acc_vals = [], [], [], []
    
    discriminator.eval()
    generator.eval()    
    
    with torch.no_grad():
        for i, (real_batch, synth_batch) in enumerate(zip(real_dataloader, synth_dataloader)): # zip() will use the minimum available size
            real_batch, synth_batch = real_batch.to(device, non_blocking=config["PIN_MEMORY"]), synth_batch.to(device, non_blocking=config["PIN_MEMORY"])
            real_batch = T.RandomRotation(180)(real_batch)
            # Calculate generator loss and PSNR
            ganerated_batch = generator(synth_batch)
            ganerated_batch = T.RandomRotation(180)(ganerated_batch)
            
            # Calculate discriminator loss and accuracy
            disc_preds = discriminator(real_batch).view(-1)
            real_D_loss = discriminator_loss(disc_preds, REAL_LABEL)
            real_D_losses.append(real_D_loss.cpu())
            acc = accuracyCalc(disc_preds, REAL_LABEL)
            acc_vals.append(acc)
        
            disc_preds = discriminator(ganerated_batch).view(-1)
            fake_D_loss = discriminator_loss(disc_preds, FAKE_LABEL)
            fake_D_losses.append(fake_D_loss.cpu())
            acc = accuracyCalc(disc_preds, FAKE_LABEL)
            acc_vals.append(acc)
            G_loss = gen_loss(synth_batch, ganerated_batch, discriminator)
            G_losses.append(G_loss.item())
    printResults(G_losses, real_D_losses, fake_D_losses, acc_vals, "Validation")


# In[ ]:


for epoch in range(config["NUM_EPOCHS"]):
    print(f"\nEpoch {epoch+1}\n"+bar_string)
    train(real_train_dataloader, synth_train_dataloader, gen, disc, opt_gen, opt_disc)
    evaluate(real_val_dataloader, synth_val_dataloader, gen, disc)


# In[ ]:cc
path = "./GANeratedHandsv1-gen.pth"
torch.save(gen.state_dict(), path)

path = "./GANeratedHandsv1-disc.pth"
torch.save(disc.state_dict(), path)

