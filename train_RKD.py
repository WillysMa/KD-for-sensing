#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.com
@file: train_RKD.py
@time: 2025/6/12 15:00
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
from MyFunc import *
from model import *
import torchvision.transforms as transf
import matplotlib.pyplot as plt

# Automatically select least used GPU
os.environ["CUDA_VISIBLE_DEVICES"] = select_best_gpu()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Single Modality Training with Relational Knowledge Distillation')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['crossentropy', 'focal'], 
                        help='Loss function type')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='Gradient clipping max norm')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--use_early_stopping', action='store_true', default=True, help='Enable early stopping')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum change to qualify as improvement')
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['cosine', 'multistep'], help='Scheduler type')  # cosine or multistep
    
    # Knowledge Distillation parameters
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for distillation loss (1-alpha for task loss)')
    parser.add_argument('--teacher_model_path', type=str, default='./Teacher_model_J3.pth', help='Path to teacher model')
    
    # Time-dependent weight parameters
    parser.add_argument('--time_weights', type=float, default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        help='num_pred+1 Time-dependent weights for knowledge distillation')
    
    # Feature-based Knowledge Distillation options
    parser.add_argument('--kd_mode', type=int, default=5, 
                        choices=[0, 1, 2, 3, 4, 5],
                        help='Knowledge distillation mode: 0--5 for no_kd, logits KD, input_features KD, output_features KD, both_features KD, relational KD')
    parser.add_argument('--feature_loss_weight', type=float, default=10.0, help='Weight for feature distillation loss')
    parser.add_argument('--input_feature_weight', type=float, default=0.5, help='Weight for input feature loss when using both_features')
    parser.add_argument('--output_feature_weight', type=float, default=0.5, help='Weight for output feature loss when using both_features')
    
    # Relational Knowledge Distillation options
    parser.add_argument('--rkd_pairs_per_anchor', type=int, default=4, help='Number of pairs per anchor sample for relational KD')
    parser.add_argument('--rkd_distance_weight', type=float, default=1.0, help='Weight for Euclidean distance loss in relational KD')
    parser.add_argument('--rkd_angle_weight', type=float, default=2.0, help='Weight for cosine angle loss in relational KD')
    
    # Model parameters
    parser.add_argument('--feature_size', type=int, default=64, help='Feature size')
    parser.add_argument('--gru_hidden_size', type=int, default=64, help='GRU hidden size')
    parser.add_argument('--gru_num_layers_teacher', type=int, default=1, help='Number of GRU layers for teacher')
    parser.add_argument('--gru_num_layers_student', type=int, default=1, help='Number of GRU layers for student')
    parser.add_argument('--num_classes', type=int, default=64, help='Number of classes')
    parser.add_argument('--seq_length_teacher', type=int, default=8, help='Sequence length for teacher model')
    parser.add_argument('--seq_length_student', type=int, default=8, help='Sequence length for student model')
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
    
    # CosineAnnealingWarmRestarts scheduler parameters
    parser.add_argument('--T_0', type=int, default=10, help='Number of iterations for the first restart')
    parser.add_argument('--T_mult', type=int, default=2, help='A factor increases T_i after a restart')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate')
    
    # Multistep scheduler parameters
    parser.add_argument('--milestones', type=int, nargs='+', default=[20, 50, 80, 100], help='Learning rate decay milestones')
    parser.add_argument('--gamma', type=float, default=0.5, help='Learning rate decay factor')
    
    return parser.parse_args()



class DistillationLoss(nn.Module):
    """
    Enhanced Knowledge Distillation Loss supporting both logits and feature-based distillation, including relational KD
    """
    def __init__(self, task_criterion, args):
        super(DistillationLoss, self).__init__()
        self.task_criterion = task_criterion
        self.args = args
        self.kl_div = nn.KLDivLoss(reduction='none')  # Changed to 'none' to handle per-sample weights
        self.mse_loss = nn.MSELoss()
        

    def feature_distillation_loss(self, student_features, teacher_features):
        """
        Calculate feature distillation loss using Euclidean distance (MSE)
        """
        # Ensure features have the same shape
        if student_features.shape == teacher_features.shape:     
            feature_loss = self.mse_loss(student_features, teacher_features)
        else:
            print(f"Student features shape: {student_features.shape}, Teacher features shape: {teacher_features.shape}")
            raise ValueError("Student and teacher features must have the same shape")
        return feature_loss
    
    def select_pairs(self, batch_size, k):
        """
        Select k pairs for each anchor sample in the batch
        Args:
            batch_size: size of the batch
            k: number of pairs per anchor
        Returns:
            pairs: tensor of shape [batch_size, k, 2] containing anchor-positive pairs
        """
        pairs = []
        for i in range(batch_size):
            # Get all possible positive indices (excluding the anchor itself)
            positive_indices = list(range(batch_size))
            positive_indices.remove(i)
            
            # Randomly select k pairs (or all available if k > available)
            k_actual = min(k, len(positive_indices))
            if k_actual > 0:
                selected_positives = torch.randperm(len(positive_indices))[:k_actual]
                selected_indices = [positive_indices[idx] for idx in selected_positives]
                
                # Create pairs [anchor, positive]
                anchor_pairs = [[i, j] for j in selected_indices]
                pairs.extend(anchor_pairs)
        
        return torch.tensor(pairs) if pairs else torch.empty(0, 2, dtype=torch.long)
    
    def compute_euclidean_distance(self, features, pairs):
        """
        Compute normalized Euclidean distances for given pairs
        Args:
            features: tensor of shape [batch_size, seq_len, feature_dim]
            pairs: tensor of shape [num_pairs, 2] containing indices
        Returns:
            distances: tensor of shape [num_pairs] containing normalized distances
        """
        if pairs.numel() == 0:
            return torch.empty(0, device=features.device)
            
        # Flatten features for distance computation: [batch_size, seq_len * feature_dim]
        features_flat = features.reshape(features.size(0), -1)
        
        # Get feature vectors for pairs
        anchor_features = features_flat[pairs[:, 0]]  # [num_pairs, seq_len * feature_dim]
        positive_features = features_flat[pairs[:, 1]]  # [num_pairs, seq_len * feature_dim]
        
        # Compute Euclidean distances
        distances = torch.norm(anchor_features - positive_features, p=2, dim=1)
        
        # Normalize distances by mean of all d_ij in batch
        mean_distance = distances.mean() if distances.numel() > 0 else torch.tensor(1.0, device=features.device)
        if mean_distance > 0:
            distances = distances / mean_distance
        
        return distances
    
    def compute_cosine_distance(self, features, pairs):
        """
        Compute cosine distances (1 - cosine_similarity) for given pairs
        Args:
            features: tensor of shape [batch_size, seq_len, feature_dim]
            pairs: tensor of shape [num_pairs, 2] containing indices
        Returns:
            distances: tensor of shape [num_pairs] containing cosine distances
        """
        if pairs.numel() == 0:
            return torch.empty(0, device=features.device)
            
        # Flatten features for distance computation: [batch_size, seq_len * feature_dim]
        features_flat = features.reshape(features.size(0), -1)
        
        # Get feature vectors for pairs
        anchor_features = features_flat[pairs[:, 0]]  # [num_pairs, seq_len * feature_dim]
        positive_features = features_flat[pairs[:, 1]]  # [num_pairs, seq_len * feature_dim]
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(anchor_features, positive_features, dim=1)
        
        # Convert to cosine distance (1 - cosine_similarity)
        cosine_distances = 1 - cos_sim
        
        return cosine_distances
    
    def relational_knowledge_distillation_loss(self, student_features, teacher_features):
        """
        Compute Relational Knowledge Distillation loss
        Args:
            student_features: tensor of shape [batch_size, seq_len, feature_dim]
            teacher_features: tensor of shape [batch_size, seq_len, feature_dim]
        Returns:
            rkd_loss: scalar tensor representing the RKD loss
        """
        batch_size = student_features.size(0)
        
        # Select pairs for relational learning
        pairs = self.select_pairs(batch_size, self.args.rkd_pairs_per_anchor)
        
        if pairs.numel() == 0:
            return torch.tensor(0.0, device=student_features.device)
        
        pairs = pairs.to(student_features.device)
        
        # Compute distances for student features
        student_euclidean = self.compute_euclidean_distance(student_features, pairs)
        student_cosine = self.compute_cosine_distance(student_features, pairs)
        
        # Compute distances for teacher features
        teacher_euclidean = self.compute_euclidean_distance(teacher_features, pairs)
        teacher_cosine = self.compute_cosine_distance(teacher_features, pairs)
        
        # Compute distance-based loss (MSE between normalized distances)
        distance_loss = self.mse_loss(student_euclidean, teacher_euclidean)
        
        # Compute angle-based loss (MSE between cosine distances)
        angle_loss = self.mse_loss(student_cosine, teacher_cosine)
        
        # Combine losses with weights
        rkd_loss = self.args.rkd_distance_weight * distance_loss + self.args.rkd_angle_weight * angle_loss
        
        return rkd_loss
    
    def forward(self, student_logits, teacher_logits, targets, 
                student_input_features=None, teacher_input_features=None,
                student_output_features=None, teacher_output_features=None,
                input_shape_mapping=None):
        """
        Args:
            student_logits: logits from student model [batch_size * seq_len, num_classes]
            teacher_logits: logits from teacher model [batch_size * seq_len, num_classes]
            targets: ground truth labels [batch_size * seq_len]
            student_input_features: input features from student model [batch_size, seq_len, feature_size]
            teacher_input_features: input features from teacher model [batch_size, seq_len, feature_size]
            student_output_features: output features from student model [batch_size, seq_len, hidden_size]
            teacher_output_features: output features from teacher model [batch_size, seq_len, hidden_size]
            input_shape_mapping: shape mapping network for input features
        """
        # Task loss (standard cross-entropy or focal loss)
        task_loss = self.task_criterion(student_logits, targets)
        
        # Initialize distillation loss
        distillation_loss = torch.tensor(0.0, device=student_logits.device)
        
        if self.args.kd_mode == 0:
            # No knowledge distillation, only task loss
            total_loss = task_loss
            return total_loss, task_loss, distillation_loss
        
        elif self.args.kd_mode == 1:
            # Standard logits-based knowledge distillation with time-dependent weights
            student_soft = F.log_softmax(student_logits / self.args.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.args.temperature, dim=1)
            
            # Get the sequence length from the reshaped logits
            batch_size = student_logits.size(0) // (self.args.num_pred + 1)
            
            # Generate time-dependent weights
            time_weights = torch.tensor(self.args.time_weights, device=student_logits.device).repeat(batch_size)  # Repeat for each sample in batch
            
            # Calculate weighted KL divergence
            kl_loss = self.kl_div(student_soft, teacher_soft)  # [batch_size * seq_len, num_classes]
            kl_loss = kl_loss.mean(dim=1)  # Average over classes [batch_size * seq_len]
            kl_loss = (kl_loss * time_weights).mean()  # Weighted average over sequence
            
            distillation_loss = kl_loss * (self.args.temperature ** 2)
        
        elif self.args.kd_mode == 2:
            # Feature distillation using input features only
            if student_input_features is None or teacher_input_features is None:
                raise ValueError("Input features required for input_features KD mode")
            teacher_input_features_mapping = input_shape_mapping(teacher_input_features)
            distillation_loss = self.feature_distillation_loss(student_input_features, teacher_input_features_mapping)
            distillation_loss *= self.args.feature_loss_weight
        
        elif self.args.kd_mode == 3:
            # Feature distillation using output features only
            if student_output_features is None or teacher_output_features is None:
                raise ValueError("Output features required for output_features KD mode")
            distillation_loss = self.feature_distillation_loss(student_output_features, teacher_output_features)
            distillation_loss *= self.args.feature_loss_weight
        
        elif self.args.kd_mode == 4:
            # Feature distillation using both input and output features with separate shape mappings
            if (student_input_features is None or teacher_input_features is None or
                student_output_features is None or teacher_output_features is None):
                raise ValueError("Both input and output features required for both_features KD mode")
            
            teacher_input_features_mapping = input_shape_mapping(teacher_input_features)
            input_feature_loss = self.feature_distillation_loss(student_input_features, teacher_input_features_mapping)
            output_feature_loss = self.feature_distillation_loss(student_output_features, teacher_output_features)
            
            distillation_loss = (self.input_feature_weight * input_feature_loss + 
                               self.output_feature_weight * output_feature_loss) * self.args.feature_loss_weight
        
        elif self.args.kd_mode == 5:
            # Relational Knowledge Distillation using output features
            if student_output_features is None or teacher_output_features is None:
                raise ValueError("Output features required for relational KD mode")
            distillation_loss = self.relational_knowledge_distillation_loss(student_output_features, teacher_output_features)
            distillation_loss *= self.args.feature_loss_weight
        
        # Combined loss
        total_loss = (1 - self.args.alpha) * task_loss + self.args.alpha * distillation_loss
        
        return total_loss, task_loss, distillation_loss


def train_model(teacher_model, student_model, dataloaders, args, optimizer, scheduler, device, save_path, input_linear_mapping=None):
    """Main training function with knowledge distillation using linear feature mapping"""
    # Initialize loss function
    if args.loss_type == 'focal':
        task_criterion = FocalLoss(alpha=1, gamma=2)
        print(f"=====Using Focal Loss (alpha=1, gamma=2)=====")
    else:
        task_criterion = nn.CrossEntropyLoss()
        print(f"=====Using CrossEntropy Loss=====")
    
    # Initialize distillation loss
    distillation_criterion = DistillationLoss( task_criterion=task_criterion, args=args)
    
    if args.kd_mode == 0:
        print(f"=====Training without Knowledge Distillation (task loss only)=====")
    else:
        print(f"=====Using Knowledge Distillation mode: {args.kd_mode}=====")
        print(f"=====Temperature: {args.temperature}, Alpha: {args.alpha}=====")
        if args.kd_mode in [2, 3, 4, 5]:
            print(f"=====Feature loss weight: {args.feature_loss_weight}=====")
            if args.kd_mode == 4:
                print(f"=====Input feature weight: {args.input_feature_weight}, Output feature weight: {args.output_feature_weight}=====")
            elif args.kd_mode == 5:
                print(f"=====RKD pairs per anchor: {args.rkd_pairs_per_anchor}=====")
                print(f"=====RKD distance weight: {args.rkd_distance_weight}, RKD angle weight: {args.rkd_angle_weight}=====")
        if args.kd_mode == 1:
            print(f"=====Time-dependent weights: {args.time_weights}=====")
    
    # Set teacher model to evaluation mode (no gradient updates)
    if args.kd_mode != 0:
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
    print('Start training...', flush=True)

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
                d1,d2,d3,d4,d5 = img.shape
                image_batch_student = torch.cat([img[:,1-args.seq_length_student:, ...], torch.zeros(d1, args.num_pred, d3, d4, d5)], dim=1).to(device)
                image_batch_teacher = torch.cat([img[:,1-args.seq_length_teacher:, ...], torch.zeros(d1, args.num_pred, d3, d4, d5)], dim=1).to(device)
                label = torch.cat([beam_downsampled[..., -1:], label_downsampled[:, :args.num_pred]], dim=-1).to(device)
                beam_opt = beam_downsampled[:, 0].type(torch.LongTensor).to(device)
                # Forward pass - Student model
                optimizer.zero_grad()
                student_outputs, student_input_features, student_out_features = student_model(image_batch_student, beam_opt) # beam_opt is not used when using only image for training, left for extension
                # Output_s_valid = student_outputs[:, -4:, :]
                InFeature_s_valid = student_input_features[:,:args.seq_length_student-1:,:] # extract valid input features
                OutFeature_s_valid = student_out_features[:,-args.num_pred-1:,:] # extract valid output features
       
                # Forward pass - Teacher model (no gradients) - only if using KD
                if args.kd_mode != 0:
                    with torch.no_grad():
                        teacher_outputs, teacher_input_features, teacher_out_features = teacher_model(image_batch_teacher, beam_opt)
                        # Output_t_valid = teacher_outputs[:, -4:, :]
                        InFeature_t_valid = teacher_input_features[:,:args.seq_length_teacher-1,:] # extract valid input features
                        OutFeature_t_valid = teacher_out_features[:,-args.num_pred-1:,:] # extract valid output features
                else:
                    # Create dummy teacher outputs for no_kd mode
                    # Use actual batch size from current data instead of fixed args.batch_size
                    current_batch_size = image_batch_teacher.shape[0]
                    teacher_outputs = torch.zeros([current_batch_size, args.seq_length_teacher-1, args.num_classes], device=device)
                    InFeature_t_valid = torch.zeros([current_batch_size, args.num_pred+1, args.num_classes], device=device)
                    OutFeature_t_valid = torch.zeros([current_batch_size, args.num_pred+1, args.num_classes], device=device)

                
                student_outputs = student_outputs[:, -(args.num_pred + 1):, :]
                if args.kd_mode != 0:
                    teacher_outputs = teacher_outputs[:, -(args.num_pred + 1):, :]

                # print(f"student_outputs.shape: {student_outputs.shape}")
                # print(f"teacher_outputs.shape: {teacher_outputs.shape}")
                # print(f"label.shape: {label.shape}")

                # Reshape for loss calculation
                student_logits = student_outputs.reshape(-1, args.num_classes)
                teacher_logits = teacher_outputs.reshape(-1, args.num_classes)
                targets = label.flatten()
                
                
                # Calculate distillation loss
                total_loss, task_loss, distill_loss = distillation_criterion(
                    student_logits, teacher_logits, targets,
                    InFeature_s_valid, InFeature_t_valid,
                    OutFeature_s_valid, OutFeature_t_valid,
                    input_linear_mapping
                )
                
                total_loss.backward()

                # Clip gradients for student model and linear mapping networks
                all_params = list(student_model.parameters())
                if input_linear_mapping is not None:
                    all_params.extend(list(input_linear_mapping.parameters()))

                torch.nn.utils.clip_grad_norm_(all_params, 10)
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

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': student_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'test_loss': val_loss,
        }, save_path, f'Final_model_KD.pth')

        # Combined model saving and early stopping logic
        current_test_loss = val_loss
        improvement_threshold = best_test_loss - (args.min_delta if args.use_early_stopping else 0)
        
        if current_test_loss < improvement_threshold:
            # Validation loss improved - save model and reset early stopping counter
            best_test_loss = current_test_loss
            torch.save(student_model.state_dict(), os.path.join(save_path, 'model_best.pth'))
            print(f"New best model saved! Validation loss: {best_test_loss:.4f}")
           
            
            if args.use_early_stopping:
                epochs_without_improvement = 0
        else:
            # No improvement
            if args.use_early_stopping:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epochs (best: {best_test_loss:.4f})")
                
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                    print(f"Best validation loss: {best_test_loss:.4f}")
                    break

    if args.tensorboard:
        writer.close()

    time_elapsed = time.time() - start_time
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Finished Training with mode {args.kd_mode} and loss type {args.loss_type}')

    return train_acc_all, train_loss_all, val_acc_all, val_loss_all, lrs, train_task_loss_all, train_distill_loss_all


def validate_model(epoch, model, dataloader, args, device, save_path):
    """Test function with comprehensive evaluation
    
    Args:
        test_seq_length: Optional different sequence length for testing. If None, uses args.seq_length_student
        test_num_pred: Optional different number of predictions for testing. If None, uses args.num_pred
    """
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

        # Prepare data with flexible sequence and output lengths
        beam_downsampled = torch.floor(beam.float() / args.downsample_ratio).to(torch.int64)
        label_downsampled = torch.floor(label.float() / args.downsample_ratio).to(torch.int64)

        img = img.unsqueeze(2)
        d1,d2,d3,d4,d5 = img.shape
        image_batch = torch.cat([img[:,1-args.seq_length_student:, ...], torch.zeros(d1, args.num_pred, d3, d4, d5)], dim=1).to(device)
        
        # Adjust label based on test_num_pred - only take as many predictions as we're testing
        if args.num_pred <= label_downsampled.shape[1]:
            test_label = torch.cat([beam_downsampled[..., -1:], label_downsampled[:, :args.num_pred]], dim=-1).to(device)
        else:
            print("Error: num_pred is greater than the number of predictions in the label")
            exit()
        
        beam_opt = beam_downsampled[:, 0].type(torch.LongTensor).to(device)
        
        with torch.no_grad():
            outputs, _, _ = model(image_batch, beam_opt)  # Unpack the three outputs
        

        outputs = outputs[:, -(args.num_pred + 1):, :]

        val_loss += criterion(outputs.reshape(-1, args.num_classes), test_label.flatten()).item()
        
        all_outputs.append(outputs)
        all_labels.append(test_label)

    # Concatenate all outputs and labels
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    topk_acc, total = calculate_topk_accuracy(all_outputs, all_labels)
    dba_score = calculate_dba_score(all_outputs, all_labels)
    
    val_loss /= len(dataloader)
    
    param_info = ""
    param_info = f" (seq_len={args.seq_length_student}, num_pred={args.num_pred})"
    
    print(f'Epoch {epoch} Test Loss{param_info}: {val_loss:.4f}', flush=True)
    print("DBA-Score (Top-3):", dba_score)
    print('Top-K Accuracy:', flush=True)
    for k, acc in topk_acc.items():
        print(f'Top-{k}: {acc}', flush=True)
    

     # Save results
    with open(os.path.join(save_path, 'test_results.txt'), "a") as f:
        f.write(f"Epoch {epoch} Results Summary{param_info}\n\n")
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
            d1,d2,d3,d4,d5 = img.shape
            image_batch = torch.cat([img[:,1-args.seq_length_student:, ...], torch.zeros(d1, args.num_pred, d3, d4, d5)], dim=1).to(device)
            
            # Adjust label based on test_num_pred - only take as many predictions as we're testing
            if args.num_pred <= label_downsampled.shape[1]:
                test_label = torch.cat([beam_downsampled[..., -1:], label_downsampled[:, :args.num_pred]], dim=-1).to(device)
            else:
                print('Error: More predictions than available lables')
                exit()
            beam_opt = beam_downsampled[:, 0].type(torch.LongTensor).to(device)
            
            # Forward pass
            with torch.no_grad():
                student_outputs, _, _ = model(image_batch, beam_opt)
            
            student_outputs = student_outputs[:, -(args.num_pred + 1):, :]

            val_loss += criterion(student_outputs.reshape(-1, args.num_classes), test_label.flatten()).item()
            
            all_outputs.append(student_outputs)
            all_labels.append(test_label)
        
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
        args.teacher_model_path = './Single_modality/T_L8J8_ema.pth'
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
    for file in ['train_RKD.py', 'MyFunc.py', 'model.py']:
        if os.path.exists(os.path.join(script_dir, file)):
            shutil.copy(os.path.join(script_dir, file), save_directory)
    
    print(f"=====Save directory: {save_directory}=====")
    print(f"=====Training started at: {dayTime} {hourTime}=====")
    
    # Setup data
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    data_root = parent_dir + '/dataset/scenario9'
    
    train_csv_name = 'train_seqs_3.csv'
    test_csv_name = 'test_seqs_3.csv'

    train_dir = os.path.join(data_root, train_csv_name)
    test_dir = os.path.join(data_root, test_csv_name)
    
    # Data preprocessing
    img_resize = transf.Resize((224, 224))
    proc_pipe = transf.Compose([transf.ToPILImage(), img_resize])

    
    # Create data loaders
    train_loader = DataLoader(
        DataFeed(data_root, train_dir, args.seq_length_teacher, transform=proc_pipe,portion=args.dataset_pct),
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        DataFeed(data_root, test_dir, args.seq_length_teacher, transform=proc_pipe, portion=args.dataset_pct),
        batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    print(f'TrainDataSize: {len(train_loader.dataset)}, TestDataSize: {len(test_loader.dataset)}')
    dataloaders = {'train': train_loader, 'test': test_loader}
    
    # Setup device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
  
    print(f"Using device: {device}")
    
    # Define GRU parameters
    GRU_PARAMS_teacher = (args.feature_size, args.gru_hidden_size, args.gru_num_layers_teacher)
    GRU_PARAMS_student = (args.feature_size, args.gru_hidden_size, args.gru_num_layers_student)

    # Create teacher model
    teacher_model = ImageModalityNet(args.feature_size, args.num_classes, GRU_PARAMS_teacher)
    teacher_model.load_state_dict(torch.load(args.teacher_model_path, map_location=device))
    teacher_model.to(device)
    print(f"=====Teacher model loaded from: {args.teacher_model_path}=====")

    # Create student model (identical architecture)
    student_model = StudentModalityNet(args.feature_size, args.num_classes, GRU_PARAMS_student)
    student_model.to(device)
    print("=====Student model with different architecture to teacher=====")


    # Create shape mapping models based on KD mode
    input_linear_mapping = None
    
    if args.seq_length_student != args.seq_length_teacher and args.kd_mode in [2, 4]:

        # Only input features KD - create single mapping for input features
        input_linear_mapping = LinearMapping(args.seq_length_teacher-1, args.seq_length_student-1)
        input_linear_mapping.to(device)
        print("=====Input linear mapping model created for mode 2 or 4 (input features only)=====")
        


    
    # Save model summary and parameters to file
    with open(os.path.join(save_directory, 'params.txt'), 'a') as f:
        f.write("\n\nModel Architecture Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Capture model summary
        import io
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        # Print model summary using student model
        image_input = torch.randn(args.train_batch_size, args.seq_length_student-1, 1, 224, 224).to(device)
        beam = torch.randint(low=0, high=args.num_classes, size=(args.train_batch_size,), dtype=torch.long).to(device)
        try:
            print(summary(student_model, image_input, beam, show_input=True, show_hierarchical=True))
        except:
            print("Model summary could not be generated")
        
        sys.stdout = old_stdout
        model_summary = buffer.getvalue()
        
        f.write(model_summary)
        f.write("\n" + "=" * 50 + "\n")
        
        # Calculate and write parameter information
        trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in student_model.parameters())
        
        # Add shape mapping parameters if they exist
        if input_linear_mapping is not None:
            input_mapping_params = sum(p.numel() for p in input_linear_mapping.parameters())
            trainable_params += input_mapping_params
            total_params += input_mapping_params
        
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
        f.write(f"KD Mode: {args.kd_mode}\n")
        if args.kd_mode == 5:
            f.write(f"RKD pairs per anchor: {args.rkd_pairs_per_anchor}\n")
            f.write(f"RKD distance weight: {args.rkd_distance_weight}\n")
            f.write(f"RKD angle weight: {args.rkd_angle_weight}\n")
        if input_linear_mapping is not None:
            f.write(f"Input linear mapping: {input_mapping_params:,} parameters\n")
        f.write(f"\nTraining Mode:\n")
        f.write(f"Debug mode: {args.debug}\n")
        f.write(f"Save directory: {save_directory}\n")

    print(f"Total trainable parameters in student model: {trainable_params:,}")
    
    if args.debug:
        print(f"=====DEBUG MODE: All outputs saved to {save_directory}=====")
    else:
        print(f"=====TRAINING MODE: All outputs saved to {save_directory}=====")
        
    # Setup optimizer and scheduler for student model and linear mapping networks
    optimizer_params = list(student_model.parameters())
    if input_linear_mapping is not None:
        optimizer_params.extend(list(input_linear_mapping.parameters()))
        
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
        print(f"Using {args.scheduler} scheduler with T_0={args.T_0}, T_mult={args.T_mult}, eta_min={args.eta_min}")
    elif args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        print(f"Using {args.scheduler} scheduler with milestones={args.milestones}, gamma={args.gamma}")
    else:
        raise ValueError(f"Invalid scheduler type: {args.scheduler}")
    
    
    print(f"Gradient clipping enabled with max norm: {args.grad_clip}")
    
    if args.use_early_stopping:
        print(f"Early stopping enabled with patience: {args.patience} epochs, min_delta: {args.min_delta}")
    else:
        print("Early stopping disabled")
    # Train student model with knowledge distillation
    train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist, lrs, train_task_loss_hist, train_distill_loss_hist = train_model(
        teacher_model, student_model, dataloaders, args, optimizer, scheduler, device, save_directory, input_linear_mapping
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
    print(f"=====Combined Training outputs saved to numpy files in {save_directory}=====")

    
    # Test student model
    print('\nStart testing student model...\n', flush=True)
    student_model.load_state_dict(torch.load(os.path.join(save_directory, 'model_best.pth')))
    _, test_acc, DBA_score = test_model(student_model, test_loader, args, device, save_directory)
    
    # Plot training curves
    plot_training_curves(train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist, lrs, save_directory, train_task_loss_hist, train_distill_loss_hist)
    
    print(f"Training completed. Results saved to: {save_directory}")



if __name__ == "__main__":
    main()


