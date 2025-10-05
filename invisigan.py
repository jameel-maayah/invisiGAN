import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import torchvision.datasets as datasets

import matplotlib.pyplot as plt

#################

# Training hyperparameters
DATASET = "CIFAR10" #["MNIST", "CIFAR10", "CIFAR100"]

EPOCHS = 100
BATCH_SIZE = 64
LATENT_DIM = 100
MESSAGE_WIDTH = 16

if DATASET == "MNIST":
    IMAGE_SHAPE = (1, 28, 28)
    SINGLE_CHANNEL = True
elif DATASET == "CIFAR10" or DATASET == "CIFAR100":
    IMAGE_SHAPE = (3, 32, 32)
    SINGLE_CHANNEL = False

# Model hyperparameters (later)
'''
GENERATOR_NUM_FILTERS
DISCRIMINATOR_NUM_FILTERS
DECODER_NUM_FILTERS
'''

# Save paths
ANIMATION_DIR = "invisigan_generated_images"
CHECKPOINT_DIR = "invisigan_model_checkpoints"

# MPS backend
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
print(f"Using device: {device}")

# Data preprocessing
if SINGLE_CHANNEL:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # mean, std
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) # mean, std for all 3 channels
    ])

# Load dataset
if DATASET == "MNIST":
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
elif DATASET == "CIFAR10":
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
elif DATASET == "CIFAR100":
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# Create sample image directory
os.makedirs(ANIMATION_DIR, exist_ok=True)

# Generator: conditioned on latent + message bits
class Generator(nn.Module):
    def __init__(self, latent_dim, msg_width, img_shape):
        super().__init__()

        self.init_size = img_shape[1] // 4 # Since we upscale twice by a scale factor of 2
        self.l1 = nn.Sequential(nn.Linear(latent_dim + msg_width, 128 * self.init_size ** 2)) # Concatenate latent vector and message vector
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), # Upscale 2x
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), # Upscale 2x
            nn.Conv2d(128, 64, 3, 1, 1), 
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, 1, 1),
            nn.Tanh() # Each pixel is [0, 1]
        )

    def forward(self, z, msg):
        x = torch.cat((z, msg), dim=1) # Concatenate latent dim and binary message
        out = self.l1(x)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        return self.conv_blocks(out)

# Discriminator: predicts real/fake
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        # Using standard convnet architecture
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        _ = torch.zeros(1, *img_shape)
        flat_dim = self.model(_).shape[1]
        self.fc = nn.Linear(flat_dim, 1) # -> Outputs real/fake prediction (no sigmoid, BCELoss expects raw logits)

    def forward(self, img):
        return self.fc(self.model(img))

# Decoder: reconstructs message bits from generated images
class Decoder(nn.Module):
    def __init__(self, img_shape, msg_width):
        super().__init__()

        # Using standard convnet architecture
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * (img_shape[1] // 4) * (img_shape[2] // 4), 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, msg_width),
            nn.Sigmoid()  # Outputs probabilities for each bit (either 0 or 1)
        )

    def forward(self, img):
        return self.model(img)

# Instantiate networks
generator =     Generator(LATENT_DIM, MESSAGE_WIDTH, IMAGE_SHAPE).to(device)
discriminator = Discriminator(IMAGE_SHAPE).to(device)
decoder =       Decoder(IMAGE_SHAPE, MESSAGE_WIDTH).to(device)

# Instantiate losses
adversarial_loss =  nn.BCEWithLogitsLoss()
decoder_loss =      nn.BCELoss()

# Instantiate optimizers
gen_optim =     optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)) # Generator
disc_optim =    optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)) # Discriminator
dec_opt =       optim.Adam(decoder.parameters(), lr=0.0002, betas=(0.5, 0.999)) # Decoder

# For consistent image sampling
fixed_noise = torch.randn(100, LATENT_DIM, device=device)
fixed_msg = torch.randint(0, 2, (100, MESSAGE_WIDTH), device=device, dtype=torch.float)

# Train loop
for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)

        # Uniform labels (these are used later to compute the discriminator accuracy for both fake and real detection)
        valid = torch.ones(imgs.size(0), 1, device=device)
        fake = torch.zeros(imgs.size(0), 1, device=device) 

        # Train discriminator
        disc_optim.zero_grad()

        z = torch.randn(imgs.size(0), LATENT_DIM, device=device) # Sample from latent space
        msgs = torch.randint(0, 2, (imgs.size(0), MESSAGE_WIDTH), device=device, dtype=torch.float) # Random messages for the generator to encode
        fake_imgs = generator(z, msgs).detach() # Sample real images from dataset

        real_loss = adversarial_loss(discriminator(imgs), valid) # Loss for detecting real images
        fake_loss = adversarial_loss(discriminator(fake_imgs), fake) # Loss for detecting fake images
        disc_loss = 0.5 * (real_loss + fake_loss) # Average them for now

        # Backpropogate discriminator loss
        disc_loss.backward() 
        disc_optim.step()

        # Train generator and decoder mutually
        gen_optim.zero_grad()
        dec_opt.zero_grad()

        z = torch.randn(imgs.size(0), LATENT_DIM, device=device) # Sample from latent space
        msgs = torch.randint(0, 2, (imgs.size(0), MESSAGE_WIDTH), device=device, dtype=torch.float) # Random messages for the generator to encode
        gen_imgs = generator(z, msgs) # Generated images
        validity = discriminator(gen_imgs) # Discriminator predictions of real/fake
        decoded = decoder(gen_imgs) # Decoder prediction of the original messages

        gen_adv_loss = adversarial_loss(validity, valid) # BCE between predicted and expected real/fake distribution (expected is all valid, so np.ones)
        gen_aux_loss = decoder_loss(decoded, msgs) # BCE between decoded and original message bits
        gen_loss = 0.5 * (gen_adv_loss + gen_aux_loss) # Average them (we can weigh these differently later, balance readability vs. realism)

        # Backpropogate generator loss
        gen_loss.backward() 
        gen_optim.step()

        # Backpropogate decoder loss
        dec_opt.step()

        pred_bits = (decoded > 0.5).float() # Discretize decoder bit predictions
        bitwise_acc = (pred_bits == msgs).float().mean().item() # Compute bitwize accuracy per message, then average them
        full_recovery = ((pred_bits == msgs).all(dim=1).float().mean().item()) # Compute full recovery per message, then average them

        # Print loss/ other metrics
        if (i + 1) % 100 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {disc_loss.item():.3f}] [G loss: {gen_loss.item():.3f}] "
                  f"(Adv: {gen_adv_loss.item():.3f}, Dec: {gen_aux_loss.item():.3f})"
                  f"[Bit acc: {bitwise_acc:.3f}] [Full rec: {full_recovery:.3f}]")

    # Save grid of generated images each epoch
    with torch.no_grad():
        sample_imgs = generator(fixed_noise, fixed_msg).cpu().numpy()
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))

        for j in range(100):
            if SINGLE_CHANNEL:
                axes[j // 10, j % 10].imshow(sample_imgs[j][0], cmap="gray")
                axes[j // 10, j % 10].axis("off")
            else:
                # Pytorch images have shape (channels, height, width), but matplotlib expects (height, width, channels)
                axes[j // 10, j % 10].imshow(np.transpose((sample_imgs[j] + 1) / 2, (1, 2, 0))) # sample_imgs has range [-1, 1], matplotlib expects range [0, 1]
                axes[j // 10, j % 10].axis("off")
            
        plt.tight_layout()
        plt.savefig(f"{ANIMATION_DIR}/{epoch:03d}.png")
        plt.close(fig)
