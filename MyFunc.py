#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.com
@file: MyFunc.py
@time: 2025/5/26 16:09
"""
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transf
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter

import torch.nn as nn
import torch.nn.functional as F


class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999, warmup_steps=1000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # Store EMA parameters
        self.ema_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_params[name] = param.data.clone()
    
    def update(self, model):
        """Update EMA parameters"""
        self.step_count += 1
        
        # Calculate dynamic decay with warmup
        if self.step_count <= self.warmup_steps:
            decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        else:
            decay = self.decay
        
        # Update EMA parameters
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ema_params:
                self.ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def apply_to_model(self, model):
        """Apply EMA parameters to model"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ema_params:
                param.data.copy_(self.ema_params[name])
    
    def store_model_params(self, model):
        """Store current model parameters"""
        stored_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                stored_params[name] = param.data.clone()
        return stored_params
    
    def restore_model_params(self, model, stored_params):
        """Restore model parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in stored_params:
                param.data.copy_(stored_params[name])
    
    def state_dict(self):
        """Return EMA state dict"""
        return {
            'ema_params': self.ema_params,
            'decay': self.decay,
            'warmup_steps': self.warmup_steps,
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict"""
        self.ema_params = state_dict['ema_params']
        self.decay = state_dict['decay']
        self.warmup_steps = state_dict['warmup_steps']
        self.step_count = state_dict['step_count']

    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_topk_accuracy(outputs, labels, k_values=[1, 2, 3, 5, 10]):
    """Calculate top-k accuracy for given k values"""
    num_pred = labels.shape[1]
    topk_correct = {k: np.zeros((num_pred,)) for k in k_values}
    total = torch.sum(labels != -100, dim=0).cpu().numpy()
    
    _, idx = torch.topk(outputs, max(k_values), dim=-1)
    idx = idx.cpu().numpy()
    labels = labels.cpu().numpy()
    
    for i in range(labels.shape[1]):  # for each time step
        for j in range(labels.shape[0]):  # examine all samples
            for k in k_values:
                topk_correct[k][i] += np.isin(labels[j, i], idx[j, i, :k])
    
    # Calculate accuracy
    topk_acc = {}
    for k in k_values:
        topk_acc[k] = topk_correct[k] / (total + 1e-8)  # Add small epsilon to avoid division by zero
    
    return topk_acc, total

def calculate_dba_score(outputs, labels, delta=5):
    """Calculate DBA (Distance-Based Accuracy) score"""
    num_pred = labels.shape[1]
    dba_score = np.zeros((num_pred,))
    valid_count = np.zeros((num_pred,))
    
    _, idx = torch.topk(outputs, 3, dim=-1)  # top-3 predictions for DBA
    idx = idx.cpu().numpy()
    labels = labels.cpu().numpy()
    
    for t in range(labels.shape[1]):
        for b in range(labels.shape[0]):
            gt = labels[b, t]
            if gt == -100:
                continue  # skip invalid label
            
            preds = idx[b, t, :3]  # top-3 predictions
            norm_dists = np.minimum(np.abs(preds - gt) / delta, 1.0)
            min_norm_dist = np.min(norm_dists)
            
            dba_score[t] += min_norm_dist
            valid_count[t] += 1
    
    # Avoid division by zero
    valid_count[valid_count == 0] = 1
    dba_score = 1 - (dba_score / valid_count)
    
    return dba_score

def save_checkpoint(state, save_path, filename='checkpoint.pth'):
    """Save training checkpoint"""
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(save_path, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint_path = os.path.join(save_path, 'Final_model.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return start_epoch, checkpoint.get('test_loss', 0.0)
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, 0.0
    

    

def plot_training_curves(train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist, lrs, save_path, train_task_loss_hist=None, train_distill_loss_hist=None):
    """Plot and save training curves including knowledge distillation losses"""
    epochs = len(train_acc_hist)
    
    # Learning rate schedule
    plt.figure()
    plt.plot(np.arange(1, len(lrs) + 1), lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.grid(True)
    plt.title('Learning Rate Schedule')
    plt.savefig(os.path.join(save_path, 'LR_schedule.png'))
    plt.close()
    
    # Accuracy curves
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), train_acc_hist, '-o', label='Train')
    plt.plot(np.arange(1, epochs + 1), test_acc_hist, '-o', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train vs Test Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Accuracy_curves.png'))
    plt.close()
    
    # Loss curves
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), train_loss_hist, '-o', label='Train Total')
    plt.plot(np.arange(1, epochs + 1), test_loss_hist, '-o', label='Test')
    
    # Add knowledge distillation loss components if available
    if train_task_loss_hist is not None:
        plt.plot(np.arange(1, epochs + 1), train_task_loss_hist, '--', label='Train Task Loss')
    if train_distill_loss_hist is not None:
        plt.plot(np.arange(1, epochs + 1), train_distill_loss_hist, ':', label='Train Distillation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Components (Knowledge Distillation)')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Loss_curves_KD.png'))
    plt.close()



def create_samples(root, portion=1.):
    f = pd.read_csv(root, na_values='')
    f = f.fillna(-99)
    Total_Num = len(f)
    num_data = int(Total_Num * portion)
    data_samples_rgb = []
    # data_samples_radar = []
    pred_beam = []
    inp_beam = []
    for idx, row in f.head(num_data).iterrows():
        vision_data = row['camera1':'camera8'].tolist()
        data_samples_rgb.append(vision_data)
        # radar_data = row['radar1':'radar8'].tolist()
        # data_samples_radar.append(radar_data)

        # Dynamic approach: get all future_beam columns
        future_beam_cols = [col for col in f.columns if col.startswith('future_beam')]
        future_beam_cols.sort()  # Ensure consistent ordering (future_beam1, future_beam2, etc.)
        future_beam = row[future_beam_cols].tolist()
        pred_beam.append(future_beam)
        # future_beam_id = np.asarray([np.argmax(np.loadtxt(pwr)) for pwr in future_beam])
        # pred_beam.append(future_beam_id)

        input_beam = row['beam1':'beam8'].tolist()
        # input_beam_id = np.asarray([np.argmax(np.loadtxt(pwr)) for pwr in input_beam]) # start with 0
        inp_beam.append(input_beam)

    print('list is ready')
    return data_samples_rgb, inp_beam, pred_beam


class DataFeed(Dataset):
    def __init__(self, data_root, root_csv, seq_len, transform=None,  portion=1.):

        self.data_root = data_root
        self.samples_rgb, self.inp_val, self.pred_val = create_samples(root_csv, portion=portion)
        self.seq_len = seq_len
        self.transform = transform


    def __len__(self):
        return len(self.samples_rgb)

    def __getitem__(self, idx):
        samples_rgb = self.samples_rgb[idx]
        # samples_radar = self.samples_radar[idx]
        beam_val = self.pred_val[idx]
        input_beam = self.inp_val[idx]

        sample_rgb = samples_rgb[-self.seq_len:]
        # sample_radar = samples_radar[-self.seq_len:]
        input_beam1 = input_beam[-self.seq_len:]

        # out_beam = torch.zeros((3,))
        image_val = np.zeros((self.seq_len, 224,224))
        image_dif = np.zeros((self.seq_len-1, 224, 224))
        image_motion_masks = np.zeros((self.seq_len - 1, 224, 224))

        beam_past = []
        for i in range(self.seq_len):
            beam_past_i = int(np.argmax(np.loadtxt(self.data_root + input_beam1[i][1:]))) 
            beam_past.append(beam_past_i)
            # Load the image
            img = self.transform(io.imread(self.data_root + sample_rgb[i][1:]))

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