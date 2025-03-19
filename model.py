import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
from typing import Optional
import smplx
from smplx.lbs import vertices2joints
from smplx.utils import SMPLOutput

class Encoder(nn.Module):
    def __init__(self, input_dim_x, output_dim_y, latent_dim):
        super(Encoder, self).__init__()
        # Input layer: concatenation of input representation (x) and pose parameters (y)
        self.fc1 = nn.Linear(input_dim_x + output_dim_y, 512)
        self.fc2 = nn.Linear(512, 256)
        # Output layers for mean (mu) and log variance (logvar) of latent distribution
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x, y):
        # Concatenate input representation and pose parameters
        combined = torch.cat([x, y], dim=1)
        h = F.relu(self.fc1(combined))
        h = F.relu(self.fc2(h))
        # Compute mean and log variance for latent variables
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, input_dim_x, output_dim_y, latent_dim):
        super(Decoder, self).__init__()
        # Input layer: concatenation of latent variable (z) and input representation (x)
        self.fc1 = nn.Linear(latent_dim + input_dim_x, 256)
        self.fc2 = nn.Linear(256, 512)
        # Output layer: reconstruct pose parameters (y_hat)
        self.fc_out = nn.Linear(512, output_dim_y)

    def forward(self, z, x):
        # Concatenate latent variable and input representation
        combined = torch.cat([z, x], dim=1)
        h = F.relu(self.fc1(combined))
        h = F.relu(self.fc2(h))
        # Reconstruct pose parameters
        y_hat = self.fc_out(h)
        return y_hat

class PoseCVAE(nn.Module):
    def __init__(self, input_dim_x, output_dim_y, latent_dim):
        super(PoseCVAE, self).__init__()
        self.encoder = Encoder(input_dim_x, output_dim_y, latent_dim)
        self.decoder = Decoder(input_dim_x, output_dim_y, latent_dim)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample z from the latent distribution
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)    # Random normal noise
        z = mu + eps * std             # Sampled latent variable
        return z

    def forward(self, x, y):
        # Encode input and pose parameters to obtain latent distribution parameters
        mu, logvar = self.encoder(x, y)
        # Sample latent variable z
        z = self.reparameterize(mu, logvar)
        # Decode z and x to reconstruct pose parameters y_hat
        y_hat = self.decoder(z, x)
        return y_hat, mu, logvar

class SMPL(smplx.SMPLLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, update_hips: bool = False, **kwargs):
        """
        Extension of the official SMPL implementation to support more joints.
        Args:
            Same as SMPLLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(SMPL, self).__init__(*args, **kwargs)
        smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                            7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
            
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))
        self.update_hips = update_hips

    def forward(self, *args, **kwargs) -> SMPLOutput:
        """
        Run forward pass. Same as SMPL and also append an extra set of joints if joint_regressor_extra is specified.
        """
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_map, :]
        if self.update_hips:
            joints[:,[9,12]] = joints[:,[9,12]] + \
                0.25*(joints[:,[9,12]]-joints[:,[12,9]]) + \
                0.5*(joints[:,[8]] - 0.5*(joints[:,[9,12]] + joints[:,[12,9]]))
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, smpl_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        smpl_output.joints = joints
        return smpl_output
