import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import pickle  


transform = transforms.Compose([
    transforms.Resize((299, 299)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def calculate_activation_statistics(dataloader):
    """Accumulates activations for mean and covariance calculation."""
    activations = []
    for images in tqdm(dataloader):
        images = images.to(device)
        with torch.no_grad():
            pred = model(images).cpu().numpy()
        activations.append(pred)
    
    activations = np.concatenate(activations, axis=0)
    mean = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mean, cov


def get_image_paths(root_dir, exts=['.tif', '.png', '.jpeg', '.jpg']):
    image_paths = []
    for root, _, files in tqdm(os.walk(root_dir)):
        for file in files:
            if any(file.lower().endswith(ext) for ext in exts):
                image_paths.append(os.path.join(root, file))
    return image_paths


for name in ["crop_delineation_final", "EuroSAT_final" , "deep_globe_final","SEN12MS"]:
    if name == "SEN12MS":
        path = "/shared/data/benchmark/SEN12MS/imgs_all"
    elif name == "deep_globe_final":
        path = "/shared/data/benchmark/deep_globe_final/imgs"
    elif name == "EuroSAT_final":
        path = "/shared/data/benchmark/EuroSAT_final"
    elif name == "bigearthnet_final":
        path = "/shared/data/benchmark/bigearthnet_final"
    elif name == "crop_delineation_final":
        path = "/shared/data/benchmark/crop_delineation_final/imgs"
    dataset_paths = get_image_paths(path)
    dataset = ImageDataset(dataset_paths, transform=transform)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.inception_v3(pretrained=True).to(device)
    model.eval()

    print(f"Computing statistics for dataset {name}...")
    mu, sigma = calculate_activation_statistics(dataloader)

    with open(f'dataset_{name}_statistics.pkl', 'wb') as f:
        pickle.dump({'mean': mu, 'cov': sigma}, f)



with open('imagenet_statistics.pkl', 'rb') as f1:
    imagenet_stats = pickle.load(f1)
    mu1, sigma1 = imagenet_stats['mean'], imagenet_stats['cov']


with open('geonet_statistics.pkl', 'rb') as f2:
    geonet_stats = pickle.load(f2)
    mu2, sigma2 = geonet_stats['mean'], geonet_stats['cov']

with open('fid_scores.txt', 'w') as file:
    for name in ["bigearthnet_final", "crop_delineation_final", "EuroSAT_final" , "deep_globe_final","SEN12MS"]:
        with open(f'dataset_{name}_statistics.pkl', 'rb') as f3:
            stats = pickle.load(f3)
            mu, sigma = stats['mean'], stats['cov']

            diff_in = mu - mu1 
            diff_gn = mu - mu2

            covmean_in = sqrtm(sigma.dot(sigma1))
            if np.iscomplexobj(covmean_in):
                covmean_in = covmean_in.real

            covmean_gn = sqrtm(sigma.dot(sigma2))
            if np.iscomplexobj(covmean_gn):
                covmean_gn = covmean_gn.real

            fid_score_in = diff_in.dot(diff_in) + np.trace(sigma + sigma1 - 2 * covmean_in)
            fid_score_gn = diff_gn.dot(diff_gn) + np.trace(sigma + sigma2 - 2 * covmean_gn)

            line = f"FID score between dataset {name} and ImageNet dataset: {fid_score_in}\n"
            line += f"FID score between dataset {name} and GeoNet dataset: {fid_score_gn}\n"
            print(line.strip())
            file.write(line)
