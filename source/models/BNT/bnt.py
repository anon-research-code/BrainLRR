import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel
import math


class AdaptiveFrequencyModule(nn.Module):
    """Frequency module with learnable frequency cutoff"""
    def __init__(self, channels, init_freq_ratio=0.5, learnable_cutoff=True):
        super().__init__()
        self.channels = channels
        self.learnable_cutoff = learnable_cutoff
        
        if learnable_cutoff:
            # Learnable frequency cutoff point (constrained to 0-1 via sigmoid)
            self.freq_ratio_logit = nn.Parameter(
                torch.tensor(self._ratio_to_logit(init_freq_ratio))
            )
        else:
            self.register_buffer('freq_ratio', torch.tensor(init_freq_ratio))
        
        # Learnable frequency weights
        self.freq_weight_low = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.freq_weight_high = nn.Parameter(torch.ones(1, channels, 1, 1))
        
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def _ratio_to_logit(self, ratio):
        # Convert ratio (0-1) to logit for initialization
        ratio = max(min(ratio, 0.99), 0.01)  # Clamp to avoid inf
        return math.log(ratio / (1 - ratio))
    
    def get_freq_ratio(self):
        if self.learnable_cutoff:
            return torch.sigmoid(self.freq_ratio_logit)
        else:
            return self.freq_ratio
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # FFT
        x_freq = torch.fft.rfft2(x, norm='ortho')
        x_freq_real = torch.view_as_real(x_freq)
        real_part = x_freq_real[..., 0]
        imag_part = x_freq_real[..., 1]
        
        amplitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        phase = torch.atan2(imag_part, real_part)
        
        H_freq, W_freq = amplitude.shape[2], amplitude.shape[3]
        crow, ccol = H_freq // 2, W_freq // 2
        
        # Get current frequency ratio
        freq_ratio = self.get_freq_ratio()
        r = (min(crow, ccol) * freq_ratio).int()
        
        # Create masks
        y, x_coord = torch.meshgrid(
            torch.arange(H_freq, device=x.device), 
            torch.arange(W_freq, device=x.device), 
            indexing='ij'
        )
        
        # Soft mask for differentiability
        distance = torch.sqrt((y - crow) ** 2 + (x_coord - ccol) ** 2)
        max_dist = math.sqrt(crow**2 + ccol**2)
        
        # Use sigmoid for soft transition
        sharpness = 10.0  # Control transition sharpness
        mask_low = torch.sigmoid(sharpness * (freq_ratio * max_dist - distance))
        mask_high = 1 - mask_low
        
        # Apply masks and weights
        amp_low = amplitude * mask_low.unsqueeze(0).unsqueeze(0) * self.freq_weight_low
        amp_high = amplitude * mask_high.unsqueeze(0).unsqueeze(0) * self.freq_weight_high
        
        # Reconstruct
        real_low = amp_low * torch.cos(phase)
        imag_low = amp_low * torch.sin(phase)
        x_freq_low = torch.complex(real_low, imag_low)
        
        real_high = amp_high * torch.cos(phase)
        imag_high = amp_high * torch.sin(phase)
        x_freq_high = torch.complex(real_high, imag_high)
        
        # IFFT
        x_low = torch.fft.irfft2(x_freq_low, s=(H, W), norm='ortho')
        x_high = torch.fft.irfft2(x_freq_high, s=(H, W), norm='ortho')
        
        # Fuse
        x_freq_features = torch.cat([x_low, x_high], dim=1)
        x_out = self.freq_conv(x_freq_features)
        
        return x_out

# NEW MODULES FOR FREQUENCY DOMAIN PROCESSING ###################################
class FrequencySpatialComplementary(nn.Module):
    """Complementary frequency and spatial processing"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Spatial branch - handles local spatial patterns
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Frequency branch - global frequency-domain enhancement
        self.freq_weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.freq_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Dynamic fusion
        self.fusion_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Spatial processing
        x_spatial = self.spatial_branch(x)
        
        # Frequency processing
        x_freq = torch.fft.rfft2(x, norm='ortho')
        x_freq_real = torch.view_as_real(x_freq)
        real_part = x_freq_real[..., 0]
        imag_part = x_freq_real[..., 1]
        
        amplitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        phase = torch.atan2(imag_part, real_part)
        
        # Simple amplitude enhancement (no hard cutoff)
        amplitude_enhanced = amplitude * self.freq_weight
        
        # Reconstruct
        real_enh = amplitude_enhanced * torch.cos(phase)
        imag_enh = amplitude_enhanced * torch.sin(phase)
        x_freq_enh = torch.complex(real_enh, imag_enh)
        x_freq_out = torch.fft.irfft2(x_freq_enh, s=(H, W), norm='ortho')
        x_freq_out = self.freq_branch(x_freq_out)
        
        # Adaptive fusion (learns when to use frequency domain vs. spatial domain)
        combined = torch.cat([x_spatial, x_freq_out], dim=1)
        weights = self.fusion_weight(combined)  # (B, 2, 1, 1)
        
        x_out = weights[:, 0:1] * x_spatial + weights[:, 1:2] * x_freq_out
        
        # Residual
        x_out = x_out + x
        
        return x_out
# NEW MODULES FOR FREQUENCY DOMAIN PROCESSING ###################################




class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True,\
                 use_frequency=False, freq_ratio=0.5):
        super().__init__()

        # Add frequency-domain processing module
        self.use_frequency = use_frequency
        if use_frequency:
            # # Reshape (batch, nodes, features) to (batch, 1, nodes, features) for frequency-domain processing
            # self.freq_module = FrequencyModule(
            #     channels=1,  # treat the entire feature matrix as a single channel
            #     freq_ratio=freq_ratio,
            #     learnable=True
            # )

            # Reshape (batch, nodes, features) to (batch, 1, nodes, features) for frequency-domain processing
            self.freq_module = FrequencySpatialComplementary(
                channels=1,  # treat the entire feature matrix as a single channel
            )


        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=4,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):

        # Frequency-domain processing
        if self.use_frequency:
            B, N, F = x.shape
            # Reshape to (batch, 1, nodes, features) for frequency processing
            x_freq_input = x.unsqueeze(1)  # (B, 1, N, F)
            x_freq = self.freq_module(x_freq_input)  # (B, 1, N, F)
            x = x_freq.squeeze(1)  # (B, N, F)

        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class BrainNetworkTransformer(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling

        # Get whether to use frequency-domain processing from the config
        use_frequency = config.model.get('use_frequency', False)
        freq_ratio = config.model.get('freq_ratio', 0.5)

        for index, size in enumerate(sizes):

            # Only use frequency domain in the first layer (when full brain-region info is preserved)
            use_freq_this_layer = (index == 0) if use_frequency else False

            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=1024,
                                    output_node_num=size,
                                    pooling=do_pooling[index],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment,
                                    use_frequency=use_freq_this_layer,
                                    freq_ratio=freq_ratio)
                                    )

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):

        bz, _, _, = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
