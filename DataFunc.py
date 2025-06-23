#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.com
@file: DataFunc.py
@time: 2025/5/26 16:09
"""
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transf
import csv
import os
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat


def create_samples(root, portion=1.):
    f = pd.read_csv(root, na_values='')
    f = f.fillna(-99)
    Total_Num = len(f)
    num_data = int(Total_Num * portion)
    data_samples_rgb = []
    data_samples_radar = []
    pred_beam = []
    inp_beam = []
    for idx, row in f.head(num_data).iterrows():
        vision_data = row['camera1':'camera8'].tolist()
        data_samples_rgb.append(vision_data)
        radar_data = row['radar1':'radar8'].tolist()
        data_samples_radar.append(radar_data)

        future_beam = row['future_beam1':'future_beam3'].tolist()
        pred_beam.append(future_beam)
        # future_beam_id = np.asarray([np.argmax(np.loadtxt(pwr)) for pwr in future_beam])
        # pred_beam.append(future_beam_id)

        input_beam = row['beam1':'beam8'].tolist()
        # input_beam_id = np.asarray([np.argmax(np.loadtxt(pwr)) for pwr in input_beam]) # start with 0
        inp_beam.append(input_beam)

    print('list is ready')
    return data_samples_rgb, data_samples_radar, inp_beam, pred_beam


class DataFeed(Dataset):
    def __init__(self, data_root, root_csv, seq_len, transform=None,  portion=1.):

        self.data_root = data_root
        self.samples_rgb, self.samples_radar, self.inp_val, self.pred_val = create_samples(root_csv, portion=portion)
        self.seq_len = seq_len
        self.transform = transform


    def __len__(self):
        return len(self.samples_rgb)

    def __getitem__(self, idx):
        samples_rgb = self.samples_rgb[idx]
        samples_radar = self.samples_radar[idx]
        beam_val = self.pred_val[idx]
        input_beam = self.inp_val[idx]

        sample_rgb = samples_rgb[-self.seq_len:]
        sample_radar = samples_radar[-self.seq_len:]
        input_beam1 = input_beam[-self.seq_len:]

        # out_beam = torch.zeros((3,))
        image_val = np.zeros((self.seq_len, 224,224))
        image_dif = np.zeros((self.seq_len-1, 224, 224))
        image_motion_masks = np.zeros((self.seq_len - 1, 224, 224))

        beam_past = []
        for i, (smp_rgb_path,smp_radar_path) in enumerate(zip(sample_rgb,sample_radar)):
            beam_past_i = int(np.argmax(np.loadtxt(self.data_root + input_beam1[i][1:]))) 
            beam_past.append(beam_past_i)
            # Load the image
            img = self.transform(io.imread(self.data_root + smp_rgb_path[1:]))

            img = rgb2gray(img)  # Convert to grayscale

            # Apply Gaussian filtering
            img_smoothed = gaussian_filter(img, sigma=1)  # Adjust sigma for smoothing strength

            # Store the smoothed image
            image_val[i, ...] = img_smoothed

            # Compute the difference with the previous frame
            if i >= 1:
                diff = np.abs(image_val[i, ...] - image_val[i - 1, ...])
                image_dif[i - 1, ...] = diff

                # Calculate the dynamic threshold as 10% of the maximum pixel value in the difference image
                max_pixel_value = np.max(diff)
                threshold_value = 0.1 * max_pixel_value

                # Generate binary mask of significant changes
                motion_mask = (diff > threshold_value).astype(np.uint8)
                image_motion_masks[i - 1, ...] = motion_mask

        image_masks = torch.tensor(image_motion_masks,dtype=torch.float32)
        # radar_masks = torch.tensor(radar_motion_masks,dtype=torch.float32)

  

        beam_future = []
        for i in range(len(beam_val)):
            beam_future_i = int(np.argmax(np.loadtxt(self.data_root + beam_val[i][1:]))) 
            beam_future.append(beam_future_i)


        input_beam = torch.tensor(beam_past, dtype=torch.int64)
        out_beam = torch.tensor(beam_future, dtype=torch.int64)

        pass
        return image_masks, input_beam.long(), torch.squeeze(out_beam.long())


if __name__ == "__main__":
    num_classes = 64
    batch_size = 4
    val_batch_size = 64
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    data_root = parent_dir + '/dataset/scenario9'
    train_dir = data_root + '/train_seqs.csv'

    seq_len = 8
    img_resize = transf.Resize((224, 224))
    # img_norm = transf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    proc_pipe = transf.Compose(
        [transf.ToPILImage(),
         img_resize]
    )
    FFT_TUPLE = (64, 64, 16) #FFT_ANGLE, FFT_RANGE, FFT_VELOCITY
    DATASET_PCT = 0.1
    train_loader = DataLoader(DataFeed(data_root, train_dir, seq_len, proc_pipe, portion=DATASET_PCT, fft_tuple=FFT_TUPLE),
                              batch_size=batch_size, shuffle=True)
    data = next(iter(train_loader))
    print('done')

    # Path to the CSV file



    # Apply thresholding to both maps

    # filtered_range_angle_map = threshold_map(range_angle_map, percentile=60)
    # filtered_range_velocity_map = threshold_map(range_velocity_map, percentile=60)
    #
    # plt.figure(figsize=(10, 5))
    # plt.imshow(filtered_range_angle_map.T, aspect='auto', cmap='jet', origin='lower')
    # plt.xlabel("Angle Index")
    # plt.ylabel("Range Index")
    # plt.title("Range-Angle Map")
    # plt.colorbar(label="Power")
    # plt.savefig(target_path + sample_name+'_RA_refined.jpg')  # Save the figure to the target path
    # plt.show()
    #
    # # Plot Range-Velocity Map
    # plt.figure(figsize=(10, 5))
    # plt.imshow(filtered_range_velocity_map, aspect='auto', cmap='jet', origin='lower')
    # plt.xlabel("Velocity Index")
    # plt.ylabel("Range Index")
    # plt.title("Range-Velocity Map")
    # plt.colorbar(label="Power")
    # plt.savefig(target_path + sample_name + '_RV_refined.jpg')  # Save the figure to the target path
    # plt.show()