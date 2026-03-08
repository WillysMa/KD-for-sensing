#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact: mamengyuan410@gmail.com
@file: DataFeed.py
@time: 2025/12/22 12:30
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transf
from skimage import io
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from Radar_KPI import *
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

        # Dynamic approach: get all future_beam columns
        future_beam_cols = [col for col in f.columns if col.startswith('future_beam')]
        future_beam_cols.sort()  # Ensure consistent ordering (future_beam1, future_beam2, etc.)
        future_beam = row[future_beam_cols].tolist()
        pred_beam.append(future_beam)

        input_beam = row['beam1':'beam8'].tolist()

        inp_beam.append(input_beam)

    # print('list is ready')
    return data_samples_rgb, data_samples_radar, inp_beam, pred_beam


class DataFeed(Dataset):
    def __init__(self, data_root, root_csv, seq_len, transform=None,   
    fft_tuple=(64, 256,128), clipped_range=128, portion=1.):

        self.data_root = data_root
        self.samples_rgb, self.samples_radar, self.inp_val, self.pred_val = create_samples(root_csv, portion=portion)
        self.seq_len = seq_len
        self.transform = transform
        self.fft_tuple = fft_tuple
        self.clipped_range = clipped_range


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
        clipped_range = self.clipped_range

        radar_val_range_angle = np.zeros((self.seq_len, clipped_range, self.fft_tuple[0]))
        radar_val_doppler_angle = np.zeros((self.seq_len, self.fft_tuple[2], self.fft_tuple[0]))
        radar_dif_RA = np.zeros((self.seq_len - 1, clipped_range, self.fft_tuple[0]))
        radar_dif_DA = np.zeros((self.seq_len - 1, self.fft_tuple[2], self.fft_tuple[0]))

        def _p(rel_path):
            # CSV paths start with '/', so join safely without duplicating separators
            return os.path.join(self.data_root, rel_path.lstrip("/"))

        for i, (smp_rgb_path,smp_radar_path) in enumerate(zip(samples_rgb,samples_radar)):
            beam_past_i = int(np.argmax(np.loadtxt(_p(input_beam1[i])))) # start with 0
            beam_past.append(beam_past_i)
            # Load the image
            img = self.transform(io.imread(_p(smp_rgb_path)))
            # Load the radar
            range_angle_map = np.load(_p(smp_radar_path))

            range_angle_clipped = range_angle_map[:clipped_range, ...]
            smp_radar_path_DA = smp_radar_path.replace('_RA', '_DA')
            doppler_angle_map = np.load(_p(smp_radar_path_DA))
            # # Store the smoothed image
            radar_val_range_angle[i,...] = range_angle_clipped #/np.max(smp_radar[:clipped_range, ...]+ 1e-6) # normalize the radar data
            radar_val_doppler_angle[i,...] = doppler_angle_map #/np.max(smp_radar[:clipped_range, ...]+ 1e-6) # normalize the radar data


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
                #------------------------------------below is the radar part------------------------------------
                diff_radar_RA = np.abs(radar_val_range_angle[i,...] - radar_val_range_angle[i - 1,...])
                diff_radar_DA = np.abs(radar_val_doppler_angle[i,...] - radar_val_doppler_angle[i - 1,...])
                radar_dif_RA[i - 1, ...] = diff_radar_RA
                radar_dif_DA[i - 1, ...] = diff_radar_DA
        image_masks = torch.tensor(image_motion_masks,dtype=torch.float32)

        radar_RA = torch.tensor(radar_val_range_angle,dtype=torch.float32)
        radar_DA = torch.tensor(radar_val_doppler_angle,dtype=torch.float32)

        beam_future = []
        for i in range(len(beam_val)):
            beam_future_i = int(np.argmax(np.loadtxt(_p(beam_val[i])))) 
            beam_future.append(beam_future_i)

        input_beam = torch.tensor(beam_past,dtype=torch.int64)
        out_beam = torch.tensor(beam_future,dtype=torch.int64)
        pass
        return image_masks, radar_RA, radar_DA, input_beam.long(), torch.squeeze(out_beam.long())



if __name__ == "__main__":
    num_classes = 64
    batch_size = 4
    val_batch_size = 5
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
    FFT_TUPLE = (64, 256, 128) #FFT_ANGLE, FFT_RANGE, FFT_VELOCITY
    DATASET_PCT = 0.1
    FeederData = DataFeed( data_root,train_dir, seq_len, proc_pipe, portion=DATASET_PCT, fft_tuple=FFT_TUPLE, clipped_range=FFT_TUPLE[1]//2)
    train_loader = DataLoader(FeederData, batch_size=batch_size, shuffle=True)
    data = next(iter(train_loader))
    image_masks, radar_masks, input_beam, out_beam = data
    print('image_masks: ', image_masks.shape)
    print('radar_masks: ', radar_masks.shape)
    print('input_beam: ', input_beam.shape)
    print('out_beam: ', out_beam.shape)
    print('done')

    # Path to the CSV file
    Radar_set = Radar_KPI()
    Radar_set.print_KPI()
    # Access the specific value
    samples_rgb, samples_radar, inp_val, pred_val = create_samples(train_dir, portion=DATASET_PCT)
    sample_id = 33
    print('sample_id: ', sample_id)
    print('samples_rgb: ', samples_rgb[sample_id][0])
    print('samples_radar: ', samples_radar[sample_id][0])
    path_str = samples_radar[sample_id][0]

    # Extract the folder name (S31) and the number (4630)

    # Combine into the desired format
    sample =  loadmat(data_root + path_str[1:])['data']
    sample_name = samples_rgb[sample_id][0].split('/')[-1].split('.')[0]
    # image_val[i] = torch.tensor(img, requires_grad=False)

    radar_cube = Radar_Cube(sample, FFT_TUPLE, remove_mean=True)


    # Compute Range-Angle Map (Summing over velocity axis)
    range_angle_map = Range_Angle(radar_cube, mean=True, log_scale=True) # Shape (num_ranges,num_angles)

    # Compute Range-Velocity Map (Summing over angle axis)
    range_velocity_map = Range_Doppler(radar_cube, mean=True, log_scale=True) # Shape (num_ranges, num_velocities)

    # Compute Doppler-Angle Map (Summing over range axis)
    doppler_angle_map = Doppler_Angle(radar_cube, mean=True, log_scale=True)# Shape (num_velocities, num_angles)
    

    range_axis = np.arange(0, FFT_TUPLE[1]) * Radar_set.range_res # [0, 255]

    velocity_bins = (np.arange(FFT_TUPLE[2]) - FFT_TUPLE[2]/2) * Radar_set.velocity_res # [-128, 128]
   
    k_a = np.arange(FFT_TUPLE[0])
    k_a_shift = k_a - FFT_TUPLE[0]/2

    sin_theta = k_a_shift / (FFT_TUPLE[0]/2)
    sin_theta = np.clip(sin_theta, -1, 1)

    angle_axis = np.degrees(np.arcsin(sin_theta))


    # Plot Range-Angle Map
    target_path = './'
    plt.figure(figsize=(10, 5))
    plt.imshow(range_angle_map, 
    extent=[angle_axis[0], angle_axis[-1], range_axis[0], range_axis[-1]],
    aspect='auto', cmap='jet', origin='lower')
    plt.xlabel("Angle [deg]")
    plt.ylabel("Range [m]")
    plt.title("Range-Angle Map")
    plt.colorbar(label="Power")
    # plt.savefig(target_path + sample_name+'_RA.jpg')  # Save the figure to the target path
    plt.show()

    # Plot Range-Velocity Map
    plt.figure(figsize=(10, 5))
    plt.imshow(range_velocity_map, 
    extent=[velocity_bins[0], velocity_bins[-1],range_axis[0], range_axis[-1]],
    aspect='auto', cmap='jet', origin='lower')
    plt.xlabel("Velocity [km/h]")
    plt.ylabel("Range [m]")
    plt.title("Range-Velocity Map")
    plt.colorbar(label="Power")
    # plt.savefig(target_path + sample_name + '_RV.jpg')  # Save the figure to the target path
    plt.show()

    # Plot Doppler-Angle Map
    plt.figure(figsize=(10, 5))
    plt.imshow(doppler_angle_map, 
    extent=[angle_axis[0], angle_axis[-1], velocity_bins[0], velocity_bins[-1]],
    aspect='auto', cmap='jet', origin='lower')
    plt.xlabel("Angle [deg]")
    plt.ylabel("Velocity [km/h]")
    plt.title("Doppler-Angle Map")
    plt.colorbar(label="Power")
    # plt.savefig(target_path + sample_name + '_DA.jpg')  # Save the figure to the target path
    plt.show()

    ccc=1