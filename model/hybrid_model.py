import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
from torch_geometric.data import Data, Batch  # for graph data handling
import torch_geometric.nn as geom_nn        # for GCN layers



# %% [code]
# Define hyperparameters for the GCN
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 256

# --- Feature Extractor ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # We'll use the entire features sequence, but also grab an intermediate feature.
        self.features = efficientnet.features
        # You can experiment with the index – here, we take the output after layer 2 as the skip.
    def forward(self, x):
        skip = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 2:  # change this index as appropriate
                skip = x
        return x, skip  # x: final feature, skip: intermediate high-res feature

# --- Graph Constructor ---
class GraphConstructor(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(GraphConstructor, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool2d((16, 16))  # Reduce spatial dimensions

    def forward(self, features):
        features = self.pooling(features)
        batch_size, _, h, w = features.shape
        node_features = self.node_embedding(features)  # B x hidden_dim x 16 x 16
        # Flatten spatial dimensions
        node_features_flat = node_features.permute(0, 2, 3, 1).reshape(-1, self.hidden_dim)

        graph_batch = []
        for b in range(batch_size):
            edge_index = []
            for i in range(h):
                for j in range(w):
                    node_idx = i * w + j
                    # 4-neighborhood connectivity
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            neighbor_idx = ni * w + nj
                            edge_index.append([node_idx, neighbor_idx])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            start_idx = b * h * w
            end_idx = (b + 1) * h * w
            x = node_features_flat[start_idx:end_idx]
            graph_data = Data(x=x, edge_index=edge_index)
            graph_batch.append(graph_data)
        batch_data = Batch.from_data_list(graph_batch)
        return batch_data

# --- GCN Module ---
class GCNModule(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(GCNModule, self).__init__()
        self.gcn1 = geom_nn.GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = geom_nn.GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, graph_batch):
        x, edge_index = graph_batch.x, graph_batch.edge_index
        x = self.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        return x, graph_batch.batch

# --- Generator ---
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 6, 12, 18)):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        for dilation in dilations:
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
            )
        self.project = nn.Conv2d(len(dilations)*out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        x = torch.cat(features, dim=1)
        x = self.project(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Generator(nn.Module):
    def __init__(self, input_dim, output_classes, skip_channels):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_classes = output_classes
        # Initial conv reshape for GCN features
        self.conv_reshape = nn.Conv2d(input_dim, 256, kernel_size=1)
        # ASPP for multi-scale context
        self.aspp = ASPP(256, 256)
        # Process skip features with a 1x1 convolution to match dimensions
        self.skip_conv = nn.Conv2d(skip_channels, 64, kernel_size=1)
        # Upsampling layers; first layer fuses skip features
        # Concatenate the ASPP output (256 channels) with the skip feature (64 channels)
        self.upconv1 = nn.ConvTranspose2d(256+64, 128, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.upconv4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(32)
        # Final attention block (SE block)
        self.se_block = SEBlock(32)
        self.final_conv = nn.Conv2d(32, output_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, original_size, skip_features):
        # x is the GCN output, flatten to (B, input_dim, 16, 16)
        x = x.view(-1, self.input_dim, 16, 16)
        x = self.conv_reshape(x)
        x = self.aspp(x)  # multi-scale features

        # Process skip features from the backbone
        skip = self.skip_conv(skip_features)
        # Resize skip to match x if necessary (assuming x is 16x16)
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate along the channel dimension
        x = torch.cat([x, skip], dim=1)  # Now channel dimension is 256+64=320

        # Upsample
        x = self.relu(self.bn1(self.upconv1(x)))
        x = self.relu(self.bn2(self.upconv2(x)))
        x = self.relu(self.bn3(self.upconv3(x)))
        x = self.relu(self.bn4(self.upconv4(x)))

        # Apply attention
        x = self.se_block(x)

        masks = self.final_conv(x)
        if masks.shape[2:] != original_size:
            masks = F.interpolate(masks, size=original_size, mode='bilinear', align_corners=False)
        return masks

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.instance_norm2 = nn.InstanceNorm2d(64)
        self.instance_norm3 = nn.InstanceNorm2d(128)
        self.instance_norm4 = nn.InstanceNorm2d(256)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.instance_norm2(self.conv2(x)))
        x = self.leaky_relu(self.instance_norm3(self.conv3(x)))
        x = self.leaky_relu(self.instance_norm4(self.conv4(x)))
        x = self.conv5(x)
        return x

# --- Full Hybrid Model ---
class HybridGCNGAN(nn.Module):
    def __init__(self, num_classes, skip_channels):
        super(HybridGCNGAN, self).__init__()
        self.feature_extractor = FeatureExtractor()
        # The final feature dimension from EfficientNet remains the same:
        feature_dim = 1280
        self.graph_constructor = GraphConstructor(feature_dim, GCN_HIDDEN_DIM)
        self.gcn_module = GCNModule(GCN_HIDDEN_DIM, GCN_OUTPUT_DIM)
        # Pass the number of skip channels to the Generator.
        self.generator = Generator(GCN_OUTPUT_DIM, num_classes, skip_channels)
        self.discriminator = Discriminator(num_classes + 3)

        # Store num_classes as an attribute of the model
        self.num_classes = num_classes  # Add this line

    def forward(self, x):
        final_features, skip_features = self.feature_extractor(x)
        original_size = (x.shape[2], x.shape[3])
        graph_batch = self.graph_constructor(final_features)
        graph_batch = graph_batch.to(x.device)
        gcn_features, _ = self.gcn_module(graph_batch)
        # Pass skip_features to the generator along with gcn_features
        seg_masks = self.generator(gcn_features, original_size, skip_features)
        return seg_masks

    def discriminate(self, masks, images):
        disc_input = torch.cat([masks, images], dim=1)
        return self.discriminator(disc_input)

# --- Loss Functions ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten predictions and targets
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class HybridLoss(nn.Module):
    def __init__(self, num_classes):
        super(HybridLoss, self).__init__()
        self.num_classes = num_classes
        class_freq = torch.tensor([7308, 735, 378, 455, 1176, 441, 1365, 574, 91, 2877, 511], dtype=torch.float32)
        weights = 1.0 / (class_freq / class_freq.sum())
        weights = weights
        # criterion = nn.CrossEntropyLoss(weight=weights)
        self.ce_loss = nn.CrossEntropyLoss()
        # self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred_masks, true_masks, discriminator_pred_fake=None, adversarial_weight=0.01):
        # Create one-hot encoding of true masks for dice loss
        if len(true_masks.shape) == 3:  # (batch, H, W)
            true_masks_one_hot = F.one_hot(true_masks, self.num_classes).permute(0, 3, 1, 2).float()
        else:
            true_masks_one_hot = true_masks.float()
        ce_loss = self.ce_loss(pred_masks, true_masks.long())
        dice = self.dice_loss(F.softmax(pred_masks, dim=1), true_masks_one_hot)
        seg_loss = ce_loss + dice
        if discriminator_pred_fake is not None:
            adv_target = torch.ones_like(discriminator_pred_fake)
            adv_loss = self.bce_loss(discriminator_pred_fake, adv_target)
            total_loss = seg_loss + adversarial_weight * adv_loss
            return total_loss, seg_loss, adv_loss
        return seg_loss, seg_loss, torch.tensor(0.0).to(pred_masks.device)

