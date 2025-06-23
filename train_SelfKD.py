#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.com
@file: train_SelfKD.py
@time: 2025/5/27 13:00
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os
import shutil
import time 
import json
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
from tqdm import tqdm
import sys
import datetime
from DataFunc import DataFeed
from model import *
import torchvision.transforms as transf
import matplotlib.pyplot as plt

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Single Modal Training with Knowledge Distillation')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['ce', 'focal'], 
                        help='Loss function type')
    
    # Knowledge Distillation parameters
    parser.add_argument('--temperature', type=float, default=3.0, help='Temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for distillation loss (1-alpha for task loss)')
    parser.add_argument('--teacher_model_path', type=str, default='./Teacher_model_v1.pth', help='Path to teacher model')
    
    # Model parameters
    parser.add_argument('--feature_size', type=int, default=64, help='Feature size')
    parser.add_argument('--gru_hidden_size', type=int, default=64, help='GRU hidden size')
    parser.add_argument('--gru_num_layers', type=int, default=1, help='Number of GRU layers')
    parser.add_argument('--num_classes', type=int, default=64, help='Number of classes')
    parser.add_argument('--seq_length', type=int, default=8, help='Sequence length')
    parser.add_argument('--num_pred', type=int, default=3, help='Number of predictions')
    parser.add_argument('--downsample_ratio', type=int, default=1, help='Downsample ratio')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='../dataset/scenario9', help='Data root directory')
    parser.add_argument('--dataset_pct', type=float, default=1.0, help='Dataset percentage to use')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    
    # Training settings
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Use tensorboard logging')
    parser.add_argument('--save_dir', type=str, default='saved_folder_train', help='Save directory')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode (saves to saved_folder_debug)')
    
    # Checkpoint settings
    parser.add_argument('--resume', type=bool, default=False, help='Path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch (for resume)')
    
    # Scheduler parameters
    parser.add_argument('--milestones', type=int, nargs='+', default=[20, 50, 80, 100], help='Learning rate decay milestones')
    parser.add_argument('--gamma', type=float, default=0.5, help='Learning rate decay factor')
    
    return parser.parse_args()

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

class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining task loss and distillation loss
    """
    def __init__(self, task_criterion, temperature=4.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.task_criterion = task_criterion
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits: logits from student model [batch_size * seq_len, num_classes]
            teacher_logits: logits from teacher model [batch_size * seq_len, num_classes]
            targets: ground truth labels [batch_size * seq_len]
        """
        # Task loss (standard cross-entropy or focal loss)
        task_loss = self.task_criterion(student_logits, targets)
        
        # Distillation loss (KL divergence between teacher and student soft predictions)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        return total_loss, task_loss, distillation_loss

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

def train_model(teacher_model, student_model, dataloaders, args, optimizer, scheduler, device, save_path):
    """Main training function with knowledge distillation"""
    # Initialize loss function
    if args.loss_type == 'focal':
        task_criterion = FocalLoss(alpha=1, gamma=2)
        print(f"=====Using Focal Loss (alpha=1, gamma=2)=====")
    else:
        task_criterion = nn.CrossEntropyLoss()
        print(f"=====Using CrossEntropy Loss=====")
    
    # Initialize distillation loss
    distillation_criterion = DistillationLoss(
        task_criterion=task_criterion,
        temperature=args.temperature,
        alpha=args.alpha
    )
    print(f"=====Using Knowledge Distillation with temperature={args.temperature}, alpha={args.alpha}=====")
    
    # Set teacher model to evaluation mode (no gradient updates)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Initialize tensorboard writer
    if args.tensorboard:
        now = datetime.datetime.now().strftime("%H_%M_%S")
        date = datetime.date.today().strftime("%y_%m_%d")
        writer = SummaryWriter(comment=now + '_' + date + '_' + student_model.name + '_KD')
    
    # Load checkpoint if resuming
    start_epoch = args.start_epoch
    best_test_loss = 1e+3
    if args.resume:
        start_epoch, best_test_loss = load_checkpoint(save_path, student_model, optimizer, scheduler)
    
    start_time = time.time()
    print('Start training with knowledge distillation')

    # Initialize tracking arrays
    train_loss_all = []
    train_task_loss_all = []
    train_distill_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    lrs = []
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        student_model.train()
        running_loss = 0.
        running_task_loss = 0.
        running_distill_loss = 0.
        running_acc = 0.
        lrs.append(optimizer.param_groups[0]["lr"])
        
        with tqdm(dataloaders['train'], unit="batch", file=sys.stdout) as tepoch:
            for i, (img, beam, label) in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")

                # Prepare data
                beam_downsampled = torch.floor(beam.float() / args.downsample_ratio).to(torch.int64)
                label_downsampled = torch.floor(label.float() / args.downsample_ratio).to(torch.int64)

                img = img.unsqueeze(2)
                image_batch = torch.cat([img, torch.zeros_like(img[:,:args.num_pred, ...])], dim=1).to(device)
                label = torch.cat([beam_downsampled[..., -1:], label_downsampled[:,:args.num_pred]], dim=-1).to(device)
                beam_opt = beam_downsampled[:,0].type(torch.LongTensor).to(device)

                # Forward pass - Student model
                optimizer.zero_grad()
                student_outputs, _, _ = student_model(image_batch, beam_opt) # beam_opt is not used when using only image for training, left for extension
                
                # Forward pass - Teacher model (no gradients)
                with torch.no_grad():
                    teacher_outputs, _, _ = teacher_model(image_batch, beam_opt)
                
                # Adjust outputs based on prediction length
                if -3+args.num_pred == 0:
                    student_outputs = student_outputs[:, -4:, :]
                    teacher_outputs = teacher_outputs[:, -4:, :]
                else:
                    student_outputs = student_outputs[:,-4:-3+args.num_pred, :]
                    teacher_outputs = teacher_outputs[:,-4:-3+args.num_pred, :]
                
                # Reshape for loss calculation
                student_logits = student_outputs.reshape(-1, args.num_classes)
                teacher_logits = teacher_outputs.reshape(-1, args.num_classes)
                targets = label.flatten()
                
                # Calculate distillation loss
                total_loss, task_loss, distill_loss = distillation_criterion(
                    student_logits, teacher_logits, targets
                )
                
                total_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 10)
                optimizer.step()

                # Calculate accuracy
                prediction = torch.argmax(student_outputs, dim=-1)
                acc = (prediction == label).sum().item() / int(torch.sum(label != -100).cpu())
                
                # Update running statistics
                running_loss = (total_loss.item() + i * running_loss) / (i + 1)
                running_task_loss = (task_loss.item() + i * running_task_loss) / (i + 1)
                running_distill_loss = (distill_loss.item() + i * running_distill_loss) / (i + 1)
                running_acc = (acc + i * running_acc) / (i + 1)
                
                log = OrderedDict()
                log['total_loss'] = running_loss
                log['task_loss'] = running_task_loss
                log['distill_loss'] = running_distill_loss
                log['acc'] = running_acc
                tepoch.set_postfix(log)
        
        scheduler.step()

        # Validation
        val_loss, topk_acc, dba_score = validate_model(epoch, student_model, dataloaders['test'], args, device, save_path)
        
        # Logging
        if args.tensorboard:
            writer.add_scalar('Loss/train_total', running_loss, epoch)
            writer.add_scalar('Loss/train_task', running_task_loss, epoch)
            writer.add_scalar('Loss/train_distill', running_distill_loss, epoch)
            writer.add_scalar('Loss/test', val_loss, epoch)
            writer.add_scalar('acc/train', running_acc, epoch)
            writer.add_scalar('acc/test', topk_acc[1][0], epoch)

        # Store metrics
        train_acc_all.append(running_acc)
        train_loss_all.append(running_loss)
        train_task_loss_all.append(running_task_loss)
        train_distill_loss_all.append(running_distill_loss)
        val_loss_all.append(val_loss)
        val_acc_all.append(topk_acc[1][0])

        # Save best model
        current_test_loss = val_loss
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            torch.save(student_model.state_dict(), os.path.join(save_path, 'model_best.pth'))
            print(f"New best student model saved test loss: {best_test_loss:.4f}")
        
        if epoch > 20:
            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'test_loss': val_loss,
            }, save_path, f'epoch_{epoch}_KD_model.pth')

    if args.tensorboard:
        writer.close()

    time_elapsed = time.time() - start_time
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Finished Training with Knowledge Distillation')

    return train_acc_all, train_loss_all, val_acc_all, val_loss_all, lrs, train_task_loss_all, train_distill_loss_all


def validate_model(epoch, model, dataloader, args, device, save_path):
    """Test function with comprehensive evaluation"""
    model.eval()
    
    # Initialize loss function
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.CrossEntropyLoss()
    
    val_loss = 0
    all_outputs = []
    all_labels = []
    
    for i, (img, beam, label) in enumerate(dataloader, 0):

        # Prepare data
        beam_downsampled = torch.floor(beam.float() / args.downsample_ratio).to(torch.int64)
        label_downsampled = torch.floor(label.float() / args.downsample_ratio).to(torch.int64)

        img = img.unsqueeze(2)
        image_batch = torch.cat([img, torch.zeros_like(img[:, :args.num_pred, ...])], dim=1).to(device)
        label = torch.cat([beam_downsampled[..., -1:], label_downsampled[:, :args.num_pred]], dim=-1).to(device)
        beam_opt = beam_downsampled[:, 0].type(torch.LongTensor).to(device)
        
        with torch.no_grad():
        # Forward pass
            student_outputs, _, _ = model(image_batch, beam_opt)
        
        # Adjust outputs
        if -3 + args.num_pred == 0:
            student_outputs = student_outputs[:, -4:, :]
        else:
            student_outputs = student_outputs[:, -4:-3 + args.num_pred, :]

        val_loss += criterion(student_outputs.reshape(-1, args.num_classes), label.flatten()).item()
        
        all_outputs.append(student_outputs)
        all_labels.append(label)

    # Concatenate all outputs and labels
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    topk_acc, total = calculate_topk_accuracy(all_outputs, all_labels)
    dba_score = calculate_dba_score(all_outputs, all_labels)
    
    val_loss /= len(dataloader)
    
    print(f'Test Loss: {val_loss:.4f}', flush=True)
    print("DBA-Score (Top-3):", dba_score)
    print('Top-K Accuracy:', flush=True)
    for k, acc in topk_acc.items():
        print(f'Top-{k}: {acc}', flush=True)
    

     # Save results
    with open(os.path.join(save_path, 'test_results.txt'), "a") as f:
        f.write(f"Epoch {epoch} Results Summary\n\n")
        f.write(f"Test Loss: {val_loss:.4f}\n\n")
        
        # Write DBA-Score
        dba_str = ", ".join([f"{x:.4f}" for x in dba_score])
        f.write(f"DBA-Score (Top-3): [{dba_str}]\n\n")
        
        # Write Top-K Accuracy
        f.write("Top-K Accuracy Per Time Slot:\n")
        for k, acc in topk_acc.items():
            acc_str = ", ".join([f"{a:.4f}" for a in acc])
            f.write(f"Top-{k} Accuracy: [{acc_str}]\n")
        f.write("=" * 50 + "\n\n")

    return val_loss, topk_acc, dba_score


def test_model(model, dataloader, args, device, save_path):
    """Test function with comprehensive evaluation"""
    model.eval()
    
    # Initialize loss function
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.CrossEntropyLoss()
    
    val_loss = 0
    all_outputs = []
    all_labels = []
    
    with tqdm(dataloader, unit="batch", file=sys.stdout) as tepoch:
        for i, (img, beam, label) in enumerate(tepoch, 0):
            tepoch.set_description(f"Testing batch {i}")

            # Prepare data
            beam_downsampled = torch.floor(beam.float() / args.downsample_ratio).to(torch.int64)
            label_downsampled = torch.floor(label.float() / args.downsample_ratio).to(torch.int64)

            img = img.unsqueeze(2)
            image_batch = torch.cat([img, torch.zeros_like(img[:, :args.num_pred, ...])], dim=1).to(device)
            label = torch.cat([beam_downsampled[..., -1:], label_downsampled[:, :args.num_pred]], dim=-1).to(device)
            beam_opt = beam_downsampled[:, 0].type(torch.LongTensor).to(device)
            
            # Forward pass
            with torch.no_grad():
                student_outputs, _, _ = model(image_batch, beam_opt)
            
            # Adjust outputs
            if -3 + args.num_pred == 0:
                student_outputs = student_outputs[:, -4:, :]
            else:
                student_outputs = student_outputs[:, -4:-3 + args.num_pred, :]

            val_loss += criterion(student_outputs.reshape(-1, args.num_classes), label.flatten()).item()
            
            all_outputs.append(student_outputs)
            all_labels.append(label)
        
    # Concatenate all outputs and labels
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    topk_acc, total = calculate_topk_accuracy(all_outputs, all_labels)
    dba_score = calculate_dba_score(all_outputs, all_labels)
    
    val_loss /= len(dataloader)
    
    print(f'Test Loss: {val_loss:.4f}', flush=True)
    print("DBA-Score (Top-3):", dba_score)
    print('Top-K Accuracy:', flush=True)
    for k, acc in topk_acc.items():
        print(f'Top-{k}: {acc}', flush=True)
    

     # Save results
    with open(os.path.join(save_path, 'test_results.txt'), "a") as f:
        f.write("Test Results Summary\n\n")
        f.write(f"Test Loss: {val_loss:.4f}\n\n")
        
        # Write DBA-Score
        dba_str = ", ".join([f"{x:.4f}" for x in dba_score])
        f.write(f"DBA-Score (Top-3): [{dba_str}]\n\n")
        
        # Write Top-K Accuracy
        f.write("Top-K Accuracy Per Time Slot:\n")
        for k, acc in topk_acc.items():
            acc_str = ", ".join([f"{a:.4f}" for a in acc])
            f.write(f"Top-{k} Accuracy: [{acc_str}]\n")
        f.write("=" * 50 + "\n\n")

    return val_loss, topk_acc, dba_score

def plot_training_curves(train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist, lrs, save_path, train_task_loss_hist=None, train_distill_loss_hist=None):
    """Plot and save training curves including knowledge distillation losses"""
    epochs = len(train_acc_hist)
    
    # Learning rate schedule
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), lrs)
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

def main():
    """Main function"""
    args = parse_args()

    
    # Set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Create save directory
    dayTime = datetime.datetime.now().strftime('%m-%d-%Y')
    hourTime = datetime.datetime.now().strftime('%H_%M')

    current_dir = os.path.dirname(__file__) #get the current directory

    # Set base save directory based on debug mode
    base_save_dir = 'saved_folder_debug' if args.debug else args.save_dir

    if args.resume:
        save_directory = os.path.join(current_dir, base_save_dir, 'Continuous_train')
        print(f"=====Resuming training from {save_directory}=====")
    else:
        mode_suffix = "_DEBUG" if args.debug else ""
        save_directory = os.path.join(current_dir, base_save_dir, f'image_{dayTime}_{hourTime}{mode_suffix}')

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Log debug mode status
    if args.debug:
        print(f"=====DEBUG MODE ENABLED - Saving to debug folder=====")

        args.teacher_model_path =  './Single-modal/Teacher_model_v1.pth'
        # Override some settings for faster debugging
        if args.epochs > 5:
            args.epochs = 3
            print(f"=====DEBUG MODE: Reducing epochs to {args.epochs} for faster testing=====")
        if args.dataset_pct > 0.2:
            args.dataset_pct = 0.1
            print(f"=====DEBUG MODE: Reducing dataset to {args.dataset_pct*100}% for faster testing=====")
    
    with open(os.path.join(save_directory, 'params.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # Copy source files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for file in ['train_SelfKD.py', 'DataFunc.py', 'model.py']:
        if os.path.exists(os.path.join(script_dir, file)):
            shutil.copy(os.path.join(script_dir, file), save_directory)
    
    print(f"=====Save directory: {save_directory}=====")
    print(f"=====Training started at: {dayTime} {hourTime}=====")
    
    # Setup data
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    data_root = parent_dir + '/dataset/scenario9'
    
    train_dir = os.path.join(data_root, 'train_seqs.csv')
    test_dir = os.path.join(data_root, 'test_seqs.csv')
    
    # Data preprocessing
    img_resize = transf.Resize((224, 224))
    proc_pipe = transf.Compose([transf.ToPILImage(), img_resize])

    
    # Create data loaders
    train_loader = DataLoader(
        DataFeed(data_root, train_dir, args.seq_length, transform=proc_pipe,portion=args.dataset_pct),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        DataFeed(data_root, test_dir, args.seq_length, transform=proc_pipe, portion=args.dataset_pct),
        batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    print(f'TrainDataSize: {len(train_loader.dataset)}, TestDataSize: {len(test_loader.dataset)}')
    dataloaders = {'train': train_loader, 'test': test_loader}
    
    # Setup device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define GRU parameters
    GRU_PARAMS = (args.feature_size, args.gru_hidden_size, args.gru_num_layers)

    # Create teacher model
    teacher_model = ImageModalityNet(args.feature_size, args.num_classes, GRU_PARAMS)
    teacher_model.load_state_dict(torch.load(args.teacher_model_path, map_location=device))
    teacher_model.to(device)
    print(f"=====Teacher model loaded from: {args.teacher_model_path}=====")

    # Create student model (identical architecture)
    student_model = ImageModalityNet(args.feature_size, args.num_classes, GRU_PARAMS)
    student_model.to(device)
    print("=====Student model initialized with identical architecture to teacher=====")

    # Print model summary using student model
    image_input = torch.randn(args.batch_size, args.seq_length-1, 1, 224, 224).to(device)
    beam = torch.randint(low=0, high=args.num_classes, size=(args.batch_size,), dtype=torch.long).to(device)
    
    # Save model summary and parameters to file
    with open(os.path.join(save_directory, 'params.txt'), 'a') as f:
        f.write("\n\nModel Architecture Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Capture model summary
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        # try:
        print(summary(student_model, image_input, beam, show_input=True, show_hierarchical=True))
        # except:
        #     print("Model summary could not be generated")
        
        sys.stdout = old_stdout
        model_summary = buffer.getvalue()
        
        f.write(model_summary)
        f.write("\n" + "=" * 50 + "\n")
        
        # Calculate and write parameter information
        trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in student_model.parameters())
        
        f.write(f"\nStudent Model Parameters:\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
        f.write(f'TrainDataSize: {len(train_loader.dataset)}\n')
        f.write(f'TestDataSize: {len(test_loader.dataset)}\n')
        f.write(f"\nKnowledge Distillation Parameters:\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Alpha (distillation weight): {args.alpha}\n")
        f.write(f"Teacher model path: {args.teacher_model_path}\n")
        f.write(f"\nTraining Mode:\n")
        f.write(f"Debug mode: {args.debug}\n")
        f.write(f"Save directory: {save_directory}\n")

    print(f"Total trainable parameters in student model: {trainable_params:,}")
    
    if args.debug:
        print(f"=====DEBUG MODE: All outputs saved to {save_directory}=====")
    else:
        print(f"=====TRAINING MODE: All outputs saved to {save_directory}=====")
        
    # Setup optimizer and scheduler for student model
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    
    # Train student model with knowledge distillation
    train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist, lrs, train_task_loss_hist, train_distill_loss_hist = train_model(
        teacher_model, student_model, dataloaders, args, optimizer, scheduler, device, save_directory
    )
    
    # Save training outputs to numpy files
    training_outputs = {
        'train_acc_hist': np.array(train_acc_hist),
        'train_loss_hist': np.array(train_loss_hist),
        'test_acc_hist': np.array(test_acc_hist),
        'test_loss_hist': np.array(test_loss_hist),
        'learning_rates': np.array(lrs),
        'train_task_loss_hist': np.array(train_task_loss_hist),
        'train_distill_loss_hist': np.array(train_distill_loss_hist)
    }
    
    # Save all outputs in a single numpy file
    np.savez(os.path.join(save_directory, 'training_outputs.npz'), **training_outputs)
    
    print(f"=====Training outputs saved to numpy files in {save_directory}=====")

    
    # Test student model
    print('\nStart testing student model...\n', flush=True)
    student_model.load_state_dict(torch.load(os.path.join(save_directory, 'model_best.pth')))
    _, test_acc, DBA_score = test_model(student_model, test_loader, args, device, save_directory)
    
    # Plot training curves
    plot_training_curves(train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist, lrs, save_directory, train_task_loss_hist, train_distill_loss_hist)
    
    print(f"Training completed. Results saved to: {save_directory}")

if __name__ == "__main__":
    main()


