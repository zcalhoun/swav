import numpy as np
import os
from PIL import Image
import time
import random
from tqdm import tqdm

def calculate_mean_std(img_path):
    mean = np.array([0., 0., 0.])
    std = np.array([0., 0., 0.])
    std_temp = np.array([0., 0., 0.])
    #images = os.listdir(img_path) works for when images are under one folder only
    start_time = time.time()
    # subset = random.sample(images, sample_size)
    for root, _, images in os.walk(img_path):
        for image in tqdm(images):
            np_img = np.array(Image.open(os.path.join(root,image)))
            np_img = np_img / 255.
            for i in range(3):
                mean[i] += np.mean(np_img[:,:,i])
        nimages += len(images)
    mean = mean / nimages
    print(f"Mean is {mean}")
    for root, _, images in os.walk(img_path):        
        for image in tqdm(images):
            np_img = np.array(Image.open(os.path.join(root,image)))
            np_img = np_img / 255.
            for i in range(3):
                std_temp[i] += ((np_img[:,:,i] - mean[i])**2).sum() / (np_img.shape[0]*np_img.shape[1]) 
    print(std_temp)
    std = np.sqrt(std_temp/nimages) 
    print(f"Std is {std}")

    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60} min to calculate the mean & the std of {len(images)}")
    with open(f"mean_std.txt", "w") as output:
        output.write(f"Mean is {mean}" + "\n" + f"Std is {std}")
    return mean, std


#TODO: fill path
img_path = ""

calculate_mean_std(img_path)