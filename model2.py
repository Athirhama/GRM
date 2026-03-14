import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_graph_feature

class DGCNN_PartSeg(nn.Module):
    def __init__(self, k=20, num_parts=50, num_categories=16):
        super(DGCNN_PartSeg, self).__init__()
        self.k = k
        

        # 1. EdgeConv Blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        # Le 3ème EdgeConv est bien un mlp(64) dans le papier
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )


        # 2. Global aggregation
        self.conv4 = nn.Sequential(
            nn.Conv1d(192, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        

        # 3. Category label branch
        self.label_conv = nn.Sequential(
            nn.Conv1d(num_categories, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )


        # 4. Segmentation head
        self.conv5 = nn.Sequential(
            nn.Conv1d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Ajout du Dropout(0.5)
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5) 
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        self.conv8 = nn.Conv1d(128, num_parts, kernel_size=1)

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x).max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x).max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x).max(dim=-1, keepdim=False)[0]

        # Fusion pour le contexte global
        x_concat = torch.cat((x1, x2, x3), dim=1)
        x_glob = self.conv4(x_concat)
        x_max = x_glob.max(dim=-1, keepdim=True)[0]

        # Préparation du categorical vector
        l = l.view(batch_size, -1, 1)
        l = self.label_conv(l)

        # Construction du super-tenseur final de dimension (Batch, 1280, N_points)
        x_combined = torch.cat((x_max, l), dim=1).repeat(1, 1, num_points)
        x_final = torch.cat((x_combined, x1, x2, x3), dim=1)

        # Couches de segmentation point par point
        x_final = self.conv5(x_final)
        x_final = self.conv6(x_final)
        x_final = self.conv7(x_final)
        
        return self.conv8(x_final)
