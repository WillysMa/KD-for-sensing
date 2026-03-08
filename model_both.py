
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class RadarFeatureExtractor(nn.Module):
    def __init__(self, n_feature, in_channels=1):
        super().__init__()
        
        self.net = nn.Sequential(
            # Output: 64 x 64
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(4), 
            nn.ReLU(),

            # Output: 32 x 32
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),

            # Output: 16 x 16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),

            # Output: 8 x 8
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        # 全连接层用于减少特征维度
        self.fc_layer = nn.Sequential(
             nn.Linear(64 * 8 * 4, 512),
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

        batch_size, seq_length, _, _, _ = x.size()
        spatial_features = []

        # 对每个时间步分别处理
        for t in range(seq_length):
            frame = x[:, t, :, :,:]  # 获取第t个时间步的帧
            
            # Apply CNN layers
            frame_features = self.net(frame)  # 应用2D CNN
            
            frame_features = self.flatten(frame_features)
            frame_features = self.fc_layer(frame_features)            
            spatial_features.append(frame_features)

        # 将所有时间步的特征拼接在一起
        spatial_features = torch.stack(spatial_features, dim=1)  # 形状: (batch_size, seq_length, n_feature)
        return spatial_features



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
        spatial_features = []

        # 对每个时间步分别处理
        for t in range(seq_length):
            frame = x[:, t, :, :,:]  # 获取第t个时间步的帧
            
            # Apply CNN layers
            frame_features = self.cnn_layers(frame)  # 应用2D CNN
            
            # Flatten and process through FC layers
            frame_features = self.flatten(frame_features)
            frame_features = self.fc_layer(frame_features)
            
            spatial_features.append(frame_features)

        # 将所有时间步的特征拼接在一起
        spatial_features = torch.stack(spatial_features, dim=1)  # 形状: (batch_size, seq_length, n_feature)
        return spatial_features



class FusionModalityNet(nn.Module):
    def __init__(self, feature_size, num_classes, gru_params, image_channels=1, radar_channels=2, num_heads=8):
        super(FusionModalityNet, self).__init__()
        '''
        This model uses both image and radar as input for learning.
        '''
        self.name = 'FusionModalityNet'
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        assert gru_input_size == feature_size, f"Error: gru_input_size ({gru_input_size}) must be equal to feature_size ({feature_size})"


        self.image_feature_extractor = ImageFeatureExtractor(n_feature=feature_size, in_channel=image_channels)
        self.radar_feature_extractor = RadarFeatureExtractor(n_feature=feature_size, in_channels=radar_channels)
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + 64, feature_size),  # Concat image and radar features
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # GRU for temporal modeling
        self.GRU = nn.GRU(input_size=gru_input_size, hidden_size=gru_hidden_size, 
                         num_layers=gru_num_layers, dropout=0.5, batch_first=True)
        
        # Multi-head attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size, num_heads=num_heads, 
            dropout=0.1, batch_first=True
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


    def forward(self, image_batch, radar_batch):
        """
        Args:
            image_batch: [B, T, C_img, H_img, W_img]
            radar_batch: [B, T, C_rad, H_rad, W_rad]
        """
        

        # Extract features: (B, T, feature_size)
        image_features = self.image_feature_extractor(image_batch)  # (B, T, feature_size)
        radar_features = self.radar_feature_extractor(radar_batch)  # (B, T, feature_size)


        # Fuse image and radar features
        fused_features = torch.cat([image_features, radar_features], dim=2)  # (B, T, feature_size*2)
        features = self.fusion_layer(fused_features)  # (B, T, feature_size)

        # Apply LayerNorm to the features
        features = self.layer_norm(features)
        
        # GRU for temporal modeling
        Seq_out, _ = self.GRU(features)

        # Apply multi-head attention to GRU output
        attn_output, attn_weights = self.multihead_attention(
            query=Seq_out,
            key=Seq_out, 
            value=Seq_out
        )
        
        # Attention output modes
        enhanced_seq_out = attn_output + Seq_out


        # Classification
        Pred = self.classifier(enhanced_seq_out)

        return Pred, features, enhanced_seq_out



class StudentModalityNet(nn.Module):
    def __init__(self, feature_size, num_classes, gru_params, image_channels=1, radar_channels=2):
        super(StudentModalityNet, self).__init__()
        '''
        This model uses both image and radar as input for learning.
        Minimal CNN architecture with GlobalAveragePool and GlobalMaxPool.
        '''
        self.name = 'StudentModalityNet'
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
        self.image_cnn_layers = nn.Sequential(
            nn.Conv2d(image_channels, 12, 3, stride=2, padding=1, bias=False),  # 224->112
            nn.BatchNorm2d(12), nn.ReLU(inplace=True),

            ds_conv_block(12, 16, stride=2),   # 112->56
            ds_conv_block(16, 24, stride=2),   # 56->28
            ds_conv_block(24, 40, stride=2),   # 28->14
            ds_conv_block(40, 96, stride=2),   # 14->7
        )
        
        self.radar_cnn_layers = nn.Sequential(
            nn.Conv2d(radar_channels, 12, 3, stride=2, padding=1, bias=False),  # 128x64->64x32
            nn.BatchNorm2d(12), nn.ReLU(inplace=True),

            ds_conv_block(12, 16, stride=2),   # ->32x16
            ds_conv_block(16, 24, stride=2),   # ->16x8
            ds_conv_block(24, 96, stride=2),   # ->8x4
        )
        
        # Global pooling layers
        self.image_global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.image_global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.radar_global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.radar_global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Feature fusion a,nd projection
        self.fusion_layer = nn.Sequential(
            nn.Linear(96*4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, feature_size),
            )

        

        self.GRU = nn.GRU(input_size=gru_input_size, hidden_size=gru_hidden_size, num_layers=gru_num_layers,
                          dropout=0.5 if gru_num_layers > 1 else 0, batch_first=True)

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

    def forward(self, image_batch, radar_batch, beam=None):
        B, T, C, H, W = image_batch.shape
        Br, Tr, Cr, Hr, Wr = radar_batch.shape
        assert B == Br and T == Tr

        # Flatten time into batch
        img = image_batch.view(B*T, C, H, W)          # (B*T, 1, 224, 224)
        rad = radar_batch.view(B*T, Cr, Hr, Wr)       # (B*T, 1, 128, 64) or your radar size

        # CNN forward
        img_feat = self.image_cnn_layers(img)         # (B*T, 96, 7, 7)
        rad_feat = self.radar_cnn_layers(rad)         # (B*T, 96, 8, 4)

        # Avg + Max pool then concat per modality
        img_avg = self.image_global_avg_pool(img_feat).flatten(1)  # (B*T, 96)
        img_max = self.image_global_max_pool(img_feat).flatten(1)  # (B*T, 96)
        img_pooled = torch.cat([img_avg, img_max], dim=1)          # (B*T, 192)

        rad_avg = self.radar_global_avg_pool(rad_feat).flatten(1)  # (B*T, 96)
        rad_max = self.radar_global_max_pool(rad_feat).flatten(1)  # (B*T, 96)
        rad_pooled = torch.cat([rad_avg, rad_max], dim=1)          # (B*T, 192)

        # Fuse modalities
        fused = torch.cat([img_pooled, rad_pooled], dim=1)         # (B*T, 384)

        # Projection to feature_size, then reshape back to sequence
        fused_features = self.fusion_layer(fused).view(B, T, -1)   # (B, T, feature_size)

        # LayerNorm + GRU
        features = self.layer_norm(fused_features)
        Seq_out, _ = self.GRU(features)

        Pred = self.classifier(Seq_out)
        return Pred, features, Seq_out

