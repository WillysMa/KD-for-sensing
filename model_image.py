#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.com
@file: MyFuncs.py
@time: 2025/12/12 17:47
"""
from pytorch_model_summary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch

  
class ImageFeatureExtractor(nn.Module):
    def __init__(self, n_feature, in_channel=1):
        super(ImageFeatureExtractor, self).__init__()


        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=4, kernel_size=(3, 3), stride=1,padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))

        )

        self.channel_attention = True
        self.spatial_attention = True

        if self.channel_attention:
            # Channel Attention Module
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(64, 64 // 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64 // 2, 64, kernel_size=1),
                nn.Sigmoid()
            )
        
        if self.spatial_attention:
            # Spatial Attention Module
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            )

        # 全局平均池化层

        self.flatten = nn.Flatten()

        # 全连接层用于减少特征维度
        self.fc_layer = nn.Sequential(
             nn.Linear(64 * 7 * 7, 512),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(512, 128),
             nn.ReLU(),
             nn.Dropout(0.3),
             nn.Linear(128, 64),
             nn.ReLU(),
             nn.Dropout(0.2),
             nn.Linear(64, n_feature)
        )




    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()

        # 合并 batch 和时间维度以向量化处理
        frames = x.reshape(batch_size * seq_length, channels, height, width)

        # Apply CNN layers
        frame_features = self.cnn_layers(frames)

        # Apply channel attention
        if self.channel_attention:
            channel_att = self.channel_attention(frame_features)
            frame_features = frame_features * channel_att

        # Apply spatial attention
        if self.spatial_attention:
            spatial_att = self.spatial_attention(frame_features)
            frame_features = frame_features * spatial_att

        # Flatten and process through FC layers
        frame_features = self.flatten(frame_features)
        frame_features = self.fc_layer(frame_features)

        # 还原为 (batch_size, seq_length, n_feature)
        spatial_features = frame_features.view(batch_size, seq_length, -1)
        return spatial_features



class ImageModalityNet(nn.Module):
    def __init__(self, feature_size, num_classes, gru_params):
        super(ImageModalityNet, self).__init__()
        '''
        This model uses only image or radar as input for learning.
        image=True indicates the use of image data; image=False implies using radar data.
        '''
        self.name = 'ImageModalityNet'
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        assert gru_input_size == feature_size, f"Error: gru_input_size ({gru_input_size}) must be equal to feature_size ({feature_size})"

        self.feature_extraction = ImageFeatureExtractor(feature_size) # image input only

  
        self.GRU = nn.GRU(input_size=gru_input_size, hidden_size=gru_hidden_size, num_layers=gru_num_layers,
                          dropout=0.8, batch_first=True)

        # Temporal attention module
        self.temporal_attention = nn.Sequential(
            nn.Linear(gru_hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Add LayerNorm before GRU input
        self.layer_norm = nn.LayerNorm(gru_input_size)
        # Classifier

        self.classifier = nn.Sequential(
             nn.Linear(gru_hidden_size, 64),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(64, 64),
             nn.ReLU(),
             nn.Dropout(0.3),
             nn.Linear(64, num_classes)
        )



    def forward(self, image_batch):
        batch_size, seq_len, _, _, _ = image_batch.size()
        # Extract features using the feature extraction network


        features = self.feature_extraction(image_batch)

        # Apply LayerNorm to the features
        features = self.layer_norm(features)
        Seq_out, _ = self.GRU(features)

        # Apply temporal attention
        attn_weights = self.temporal_attention(Seq_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention weights to sequence output
        # This creates a weighted sum across the time dimension
        context_vector = torch.sum(Seq_out * attn_weights, dim=1)
        
        # Expand context vector to match sequence length for residual connection
        context_vector_expanded = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine with original sequence using residual connection
        enhanced_seq_out = Seq_out + context_vector_expanded


        Pred = self.classifier(enhanced_seq_out) # Final classification layer

        return Pred, features, enhanced_seq_out


class ImageStudentModalityNet(nn.Module):
    """
    Student for image-only learning. Uses a lighter CNN + GRU.
    """
    def __init__(self, feature_size, num_classes, gru_params, width_multiplier=1.5, image_channels=1):
        super(ImageStudentModalityNet, self).__init__()
        '''
        This model uses only image or radar as input for learning.
        Minimal CNN architecture with GlobalAveragePool and GlobalMaxPool.
        '''
        self.name = 'ImageStudentModalityNet'
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        assert gru_input_size == feature_size, f"Error: gru_input_size ({gru_input_size}) must be equal to feature_size ({feature_size})"

        # Depthwise-separable conv block to reduce FLOPs
        def ds_conv_block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        # Lightweight CNN layers (moderate downsample + mid channels)
        # Targets ~5x-10x FLOPs reduction while preserving features.
        c1 = int(12*width_multiplier)
        c2 = int(24*width_multiplier)
        c3 = int(48*width_multiplier)
        c4 = int(96*width_multiplier)
        self.image_cnn_layers = nn.Sequential(
            nn.Conv2d(image_channels, c1, 3, stride=2, padding=1, bias=False),  # 224->112
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ds_conv_block(c1, c2, stride=1),   # 112->56
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ds_conv_block(c2, c3, stride=1),   # 56->28
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ds_conv_block(c3, c4, stride=1),   # 28->14
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Channel and spatial attention on CNN features (like teacher's ImageFeatureExtractor)
        reduction = max(c4 // 2, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c4, reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, c4, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c4, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

        # Global pooling layers
        self.image_global_max_pool = nn.AdaptiveMaxPool2d(1)



        # Feature fusion and projection
        self.fusion_layer = nn.Sequential(
            nn.Linear(c4 , 64),  
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            nn.Linear(64, feature_size)
        )

        self.GRU = nn.GRU(input_size=gru_input_size, hidden_size=gru_hidden_size, num_layers=gru_num_layers,
                          dropout=0.8, batch_first=True)

        # Lightweight temporal attention (same role as teacher) so RKD can match relation structure
        self.temporal_attention = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Add LayerNorm before GRU input
        self.layer_norm = nn.LayerNorm(gru_input_size)
        
        # Classifier
        self.classifier = nn.Sequential(
             nn.Linear(gru_hidden_size, 64),
             nn.ReLU(),
             nn.Dropout(0.3),
             nn.Linear(64, num_classes)
        )

    def forward(self, image_batch):
        B, T, C, H, W = image_batch.shape

        # Flatten time into batch
        img = image_batch.view(B*T, C, H, W)          # (B*T, 1, 224, 224)

        # CNN forward
        img_feat = self.image_cnn_layers(img)

        # Channel and spatial attention (same structure as teacher)
        channel_att = self.channel_attention(img_feat)
        img_feat = img_feat * channel_att
        spatial_att = self.spatial_attention(img_feat)
        img_feat = img_feat * spatial_att

        # Global pool then flatten
        img_pooled = self.image_global_max_pool(img_feat).flatten(1)  # (B*T, c4)


        # Projection to feature_size, then reshape back to sequence
        fused_features = self.fusion_layer(img_pooled).view(B, T, -1)   # (B, T, feature_size)

        # LayerNorm + GRU
        features = self.layer_norm(fused_features)
        Seq_out, _ = self.GRU(features)

        # Temporal attention + residual (align with teacher's enhanced_seq_out for RKD)
        attn_weights = F.softmax(self.temporal_attention(Seq_out), dim=1)
        context_vector = torch.sum(Seq_out * attn_weights, dim=1)
        context_vector_expanded = context_vector.unsqueeze(1).expand(-1, T, -1)
        enhanced_seq_out = Seq_out + context_vector_expanded

        Pred = self.classifier(enhanced_seq_out)
        return Pred, features, enhanced_seq_out
    

