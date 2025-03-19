import torch
from torch import nn
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig,
)
import numpy as np
import wandb
import argparse
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score

from loader import InteractionDataset
from trainer import CustomTrainer
from model import PoseCVAE, SMPL
from pose_utils import rotmat_to_rot6d, rot6d_to_rotmat, Keypoint3DLoss, ParameterLoss, eval_pose

def str2bool(value):
    if value.lower() in ('true', '1'):
        return True
    elif value.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Script to run an experiment")
    parser.add_argument('--expt_name', type=str, required=True, help="Name of the experiment")
    parser.add_argument('--training_task', type=str, required=True, choices=['object-only', 'pose-only', 'multi-task'], help="Type of the training")
    parser.add_argument('--future_time', type=int, default=60, help="Future time for some processing (default: 60 seconds)")
    parser.add_argument('--observation_time', type=int, default=30, help="Future time for some processing (default: 30 seconds)")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate for the model (default: 5e-5)")
    parser.add_argument('--use_video', type=str2bool, default=True, help="Flag to indicate whether to use ego video features (default: True)")
    parser.add_argument('--use_pose', type=str2bool, default=True, help="Flag to indicate whether to use pose features (default: True)")
    parser.add_argument('--use_location', type=str2bool, default=True, help="Flag to indicate whether to use location features (default: True)")
    parser.add_argument('--use_env', type=str2bool, default=True, help="Flag to indicate whether to use environment mappings (default: True)")

    args = parser.parse_args()
    return args

# Define the configuration class
class CustomConfig(PretrainedConfig):
    model_type = "custom_transformer"

    def __init__(
        self,
        training_task='object-only',
        visual_feature_dim=512,
        pose_feature_dim=256,
        object_vocab_size=1000,
        location_vocab_size=32,
        orientation_feature_dim=9,
        transformer_hidden_size=768,
        num_transformer_layers=6,
        num_heads=8,
        output_dim=4096,
        cvae_latent_dim=32,
        use_video=True,
        use_pose=True,
        use_location=True,
        use_env=True,
        observation_time = 30.,
        anticipation_time = 5.,
        future_time = 60.,
        num_decoding_samples=5,
        scenarios = "all",
        return_smpl_vertices = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.training_task = training_task
        self.visual_feature_dim = visual_feature_dim
        self.pose_feature_dim = pose_feature_dim
        self.object_vocab_size = object_vocab_size
        self.location_vocab_size = location_vocab_size
        self.orientation_feature_dim = orientation_feature_dim
        self.transformer_hidden_size = transformer_hidden_size
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.cvae_latent_dim = cvae_latent_dim
        self.use_video = use_video
        self.use_pose = use_pose
        self.use_location = use_location
        self.use_env = use_env
        self.observation_time = observation_time
        self.anticipation_time = anticipation_time
        self.future_time = future_time
        self.num_decoding_samples = num_decoding_samples
        self.scenarios = scenarios
        self.return_smpl_vertices = return_smpl_vertices

def torch_default_data_collator(features):
    '''
    Copied from the current transformers version src/transformers/data/data_collator.py
    '''
    first = features[0]
    batch = {}

    assert "label" not in first.keys(), "We are using custom collator and have removed handling label..."
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            elif isinstance(v, dict):
                # This is the custom part, we have keys in labels and want to collate them
                batch[k] = {}
                for sub_keys in v:
                    batch[k][sub_keys] = torch.tensor(np.stack([f[k][sub_keys] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch

# Define the custom transformer model
class CustomTransformerModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)

        # Define the training task
        self.training_task = config.training_task
        self.num_decoding_samples = config.num_decoding_samples

        # Modality switches
        self.use_video = config.use_video
        self.use_pose = config.use_pose
        self.use_location = config.use_location
        self.use_env = config.use_env
        self.return_smpl_vertices = config.return_smpl_vertices

        # Visual mapper: maps visual features to transformer input dimension
        if self.use_video:
            self.visual_mapper = nn.Linear(
                config.visual_feature_dim, config.transformer_hidden_size
            )

        # Pose mapper: maps pose features to transformer input dimension
        if self.use_pose:
            self.pose_mapper = nn.Linear(
                config.pose_feature_dim, config.transformer_hidden_size
            )

        # Object embedding: converts object indices to embeddings
        if self.use_env:
            self.object_embedding = nn.Embedding(
                config.object_vocab_size, config.transformer_hidden_size
            )

        if self.use_location or self.training_task in ['pose-only', 'multi-task']:
            self.location_embedding_x = nn.Embedding(
                config.location_vocab_size, config.transformer_hidden_size
            )
            self.location_embedding_y = nn.Embedding(
                config.location_vocab_size, config.transformer_hidden_size
            )
            self.location_embedding_z = nn.Embedding(
                config.location_vocab_size, config.transformer_hidden_size
            )
            self.orientation_mapper = nn.Linear(
                config.orientation_feature_dim, config.transformer_hidden_size
            )

        # Separator token embedding
        self.sep_token = nn.Parameter(
            torch.randn(config.transformer_hidden_size)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_hidden_size, nhead=config.num_heads
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_transformer_layers
        )

        # Output layer: converts transformer output to probability map
        if self.training_task in ['object-only', 'multi-task']:
            self.output_layer = nn.Linear(
                config.transformer_hidden_size, config.output_dim
            )
            # Define the loss function
            self.loss_func_interaction = nn.BCEWithLogitsLoss()
        if self.training_task in ['pose-only', 'multi-task']:
            # Define loss functions
            self.smpl_parameter_loss = ParameterLoss()
            self.smpl_keypoint_loss = Keypoint3DLoss()
            self.smpl_cfg = {'model_path': '/path/to/4DHumans/data/smpl',
                             'gender': 'neutral',
                             'num_body_joints': 23,
                             'joint_regressor_extra': '/path/to/4DHumans/data/SMPL_to_J19.pkl',
                             'mean_params': '/path/to/4DHumans/data/smpl_mean_params.npz'
                            }
            self.smpl = None
            # Define CVAE
            self.latent_dim = config.cvae_latent_dim
            # 4 times = x_embed, y_embed, z_embed, txfm_output, 6 times = 6d for num_joints and one for orientation
            self.pose_sampler = PoseCVAE(config.transformer_hidden_size * 4, 6 * (self.smpl_cfg['num_body_joints'] + 1), self.latent_dim)

        self.main_input_name = "index"

    def set_return_smpl_vertices(self):
        self.return_smpl_vertices = True

    def forward(
        self,
        index, visual_features, visual_mask, location_voxels, location_voxels_mask, orientations, orientations_mask,
        pose_features, pose_features_mask, env_voxels, sampled_future_loc, sampled_future_betas, labels=None
    ):
        """
        Forward pass of the model.

        Args:
            visual_features (torch.Tensor): Visual features tensor of shape [batch_size, num_visual_tokens, visual_feature_dim].
            pose_features (torch.Tensor): Pose features tensor of shape [batch_size, num_pose_tokens, pose_feature_dim].
            env_voxels (torch.Tensor): Object indices tensor of shape [batch_size, num_object_tokens].
            labels (torch.Tensor, optional): Labels tensor of shape [batch_size, output_dim].

        Returns:
            dict: A dictionary containing 'loss' (if labels are provided) and 'logits'.
        """
        batch_size = visual_features.size(0)

        # Map visual features
        if self.use_video:
            visual_mapped = self.visual_mapper(visual_features)

        # Embed location indices
        if self.use_location:
            loc_x_embedded = self.location_embedding_x(location_voxels[:, :, 0])  # Shape: [batch_size, max_num_visual_tokens, transformer_hidden_size]
            loc_y_embedded = self.location_embedding_y(location_voxels[:, :, 1])
            loc_z_embedded = self.location_embedding_z(location_voxels[:, :, 2])
            orientation_mapped = self.orientation_mapper(orientations)

        # Map pose features
        if self.use_pose:
            pose_mapped = self.pose_mapper(pose_features)

        # Embed object indices
        if self.use_env:
            object_embedded = self.object_embedding(env_voxels)

        # Separator token expanded for batch
        sep_token_expanded = self.sep_token.unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, -1
        )

        # Concatenate tokens with separator tokens
        input_sequence = torch.cat(
            [
                *([visual_mapped, sep_token_expanded] if self.use_video else []),
                *([loc_x_embedded, loc_y_embedded, loc_z_embedded, sep_token_expanded, orientation_mapped, sep_token_expanded] if self.use_location else []),
                *([pose_mapped, sep_token_expanded] if self.use_pose else []),
                *([object_embedded, sep_token_expanded] if self.use_env else []),
            ],
            dim=1,
        )

        # Find the masks
        device = visual_features.device
        # Separator tokens are not padding, so their mask is False
        sep_padding_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        # Object features are assumed to be of fixed length; no padding needed
        if self.use_env:
            object_padding_mask = torch.zeros(
                batch_size, object_embedded.size(1), dtype=torch.bool, device=device
            )

        # Convert visual_mask to padding mask (True where padding)
        # visual_mask is of shape [batch_size, max_num_visual_tokens], with 1s for valid tokens and 0s for padding
        if self.use_video:
            visual_padding_mask = ~visual_mask.bool()  # Shape: [batch_size, max_num_visual_tokens]
        if self.use_location:
            location_voxels_mask = ~location_voxels_mask.bool()
            orientations_mask = ~orientations_mask.bool()
        if self.use_pose:
            pose_features_mask = ~pose_features_mask.bool()

        # Concatenate masks
        attention_mask = torch.cat(
            [
                *([visual_padding_mask, sep_padding_mask] if self.use_video else []),
                *([location_voxels_mask, location_voxels_mask, location_voxels_mask, sep_padding_mask, orientations_mask, sep_padding_mask] if self.use_location else []),
                *([pose_features_mask, sep_padding_mask] if self.use_pose else []),
                *([object_padding_mask, sep_padding_mask] if self.use_env else []),
            ],
            dim=1,
        )  # Shape: [batch_size, total_seq_length]

        # Permute dimensions to match transformer input shape
        input_sequence = input_sequence.permute(1, 0, 2)

        # Pass through transformer
        transformer_output = self.transformer(input_sequence, src_key_padding_mask=attention_mask)

        # Get aggregate feature (using the first token)
        aggregate_feature = transformer_output[0]

        # Output layer
        if self.training_task in ['object-only', 'multi-task']:
            logits = self.output_layer(aggregate_feature)
        if self.training_task in ['pose-only', 'multi-task']:
            # concat aggregate_feature with location feature
            future_loc_x_embedded = self.location_embedding_x(sampled_future_loc[:, 0])
            future_loc_y_embedded = self.location_embedding_y(sampled_future_loc[:, 1])
            future_loc_z_embedded = self.location_embedding_z(sampled_future_loc[:, 2])
            future_input_embedded = torch.cat([
                future_loc_x_embedded,
                future_loc_y_embedded,
                future_loc_z_embedded,
                aggregate_feature,
            ], dim=1)
            # Shape: [B, 3072] that is 768x4
            # Sample 6d pose
            pose_6d = rotmat_to_rot6d(labels['pose'].view(-1, 3, 3)).view(batch_size, -1) # [b, 23, 3, 3] -> [b*23, 3, 3] --func--> [b*23, 6] -> [b, 138]
            if self.training:
                # When training we need to use both encoder and decoder
                predicted_pose, mu, logvar = self.pose_sampler(future_input_embedded, pose_6d)
                predicted_pose = rot6d_to_rotmat(predicted_pose).view(batch_size, -1, 3, 3)
            else:
                # Sample candidates
                predicted_pose = []
                for _ in range(self.num_decoding_samples):
                    z = torch.randn(batch_size, self.latent_dim).to(device)
                    # Generate pose parameters y_hat using the decoder
                    y_hat = self.pose_sampler.decoder(z, future_input_embedded)
                    predicted_pose.append(y_hat)
                predicted_pose = torch.stack(predicted_pose, dim=1) # op shape - (B, num_decoding_samples, num_pose_params)
                predicted_pose = rot6d_to_rotmat(predicted_pose).view(batch_size * self.num_decoding_samples, -1, 3, 3)

            # Check for the validity of the rotation transformation -- can be removed later
            assert torch.allclose(labels['pose'], rot6d_to_rotmat(pose_6d).view(batch_size, -1, 3, 3), atol=1e-6)
            # Convert SMPL params to joints_3d
            if self.smpl is None:
                self.smpl = SMPL(**self.smpl_cfg)
            smpl_model_device = self.smpl.to(device)
            smpl_params = {}
            smpl_params['global_orient'] = predicted_pose[:, :1, :, :]
            smpl_params['body_pose'] = predicted_pose[:, 1:, :, :]
            smpl_params['betas'] = sampled_future_betas if self.training else sampled_future_betas.repeat(self.num_decoding_samples, 1)
            smpl_output = smpl_model_device(**{k: v.float() for k,v in smpl_params.items()}, pose2rot=False)
            pred_smpl_vertices = smpl_output.vertices.contiguous()
            pred_keypoints_3d = smpl_output.joints.contiguous()
            # We use the same betas
            smpl_params['global_orient'] = labels['pose'][:, :1, :, :]
            smpl_params['body_pose'] = labels['pose'][:, 1:, :, :]
            smpl_params['betas'] = sampled_future_betas
            smpl_output = smpl_model_device(**{k: v.float() for k,v in smpl_params.items()}, pose2rot=False)
            gt_smpl_vertices = smpl_output.vertices.contiguous()
            gt_keypoints_3d = smpl_output.joints.contiguous()

        if labels is not None:
            # Compute loss using BCEWithLogitsLoss for binary classification
            if self.training_task in ['object-only', 'multi-task']:
                loss_interaction = self.loss_func_interaction(logits, labels['interaction'].float())
            if self.training_task in  ['pose-only', 'multi-task']:
                if self.training:
                    param_loss = self.smpl_parameter_loss(predicted_pose, labels['pose'], torch.ones(batch_size).to(device))
                    joints3d_loss = self.smpl_keypoint_loss(pred_keypoints_3d, gt_keypoints_3d)
                    # KL Divergence loss
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss_pose = param_loss + joints3d_loss + 10. * kl_loss
                else:
                    pose_gt_rep = labels['pose'].unsqueeze(1).repeat(1, self.num_decoding_samples, 1, 1, 1)
                    # Correct the shape of the validation variables before passing to outputs
                    predicted_pose = predicted_pose.view(batch_size, self.num_decoding_samples, *predicted_pose.shape[1:])
                    pred_keypoints_3d = pred_keypoints_3d.view(batch_size, self.num_decoding_samples, *pred_keypoints_3d.shape[1:])
                    pred_smpl_vertices = pred_smpl_vertices.view(batch_size, self.num_decoding_samples, *pred_smpl_vertices.shape[1:])
                    loss_pose = self.smpl_parameter_loss(predicted_pose, pose_gt_rep, torch.ones(batch_size).to(device))

            if self.training_task == 'object-only':
                loss = loss_interaction
                outputs = {"interaction": logits}
            elif self.training_task == 'pose-only':
                loss = loss_pose
                outputs = {"pose": predicted_pose, "keypoints3d": pred_keypoints_3d, "keypoints3d_gt": gt_keypoints_3d,  **({"vertices": pred_smpl_vertices, "vertices_gt": gt_smpl_vertices} if self.return_smpl_vertices else {})} # GT is passed in the output
            elif self.training_task == 'multi-task':
                raise NotImplementedError("Remove this line to test multi-task: Only do when individual ones are running good...")
                loss = loss_interaction + loss_pose
                outputs = {"interaction": logits, "pose": predicted_pose}
            else:
                raise ValueError("Invalid training task...")
            if loss.dim() == 0:
                # Unsqueeze to add a dimension
                loss = loss.unsqueeze(0)
            outputs["loss"] = loss

        return outputs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def list_model_parameters(model):
    print("Listing Model Parameters:\n")
    total_params = 0
    total_trainable_params = 0
    for name, param in model.named_parameters():
        param_size = param.numel()
        trainable = param.requires_grad
        print(f"Name: {name}")
        print(f" - Shape: {param.shape}")
        print(f" - Size: {param_size} parameters")
        print(f" - Trainable: {trainable}\n")
        total_params += param_size
        if trainable:
            total_trainable_params += param_size
    print(f"Total Parameters: {total_params}")
    print(f"Total Trainable Parameters: {total_trainable_params}")

def list_model_parameters_sorted(model):
    print("Listing Model Parameters:\n")
    total_params = 0
    total_trainable_params = 0
    params = []  # Initialize list to store parameter details
    
    for name, param in model.named_parameters():
        param_size = param.numel()
        trainable = param.requires_grad
        # Collect parameter details
        params.append((param_size, name, param.shape, trainable))
        total_params += param_size
        if trainable:
            total_trainable_params += param_size
    
    # Sort parameters by size in decreasing order
    params.sort(key=lambda x: x[0], reverse=True)
    
    # Print sorted parameters
    for param_size, name, shape, trainable in params:
        print(f"Name: {name}")
        print(f" - Shape: {shape}")
        print(f" - Size: {param_size} parameters")
        print(f" - Trainable: {trainable}\n")
    
    print(f"Total Parameters: {total_params}")
    print(f"Total Trainable Parameters: {total_trainable_params}")

class ComputeMetricsWrapper:
    def __init__(self, eval_dataset, training_task, save_vertices=False, save_locations=False):
        self.eval_dataset = eval_dataset
        self.training_task = training_task
        self.save_vertices = save_vertices
        self.save_locations = save_locations

    def ensure_top_k_ones(self, predictions, threshold=0.1, top_k=3):
        # We ensure at least k samples are one for every sample
        # Step 1: Binarize based on the threshold
        binary_prediction = (predictions > threshold).astype(int)

        # Step 2: Ensure at least top_k values are 1 in each row
        for i in range(predictions.shape[0]):
            if np.sum(binary_prediction[i]) < top_k:
                # Get the indices of the top_k values (use argsort twice to get top values in descending order)
                top_k_indices = np.argsort(predictions[i])[-top_k:]
                # Set the corresponding top_k indices to 1
                binary_prediction[i, top_k_indices] = 1

        return binary_prediction

    def process_and_calculate_chamfer(self, labels, binary_predictions):
        labels_3d = labels.reshape(-1, 16, 16, 16)
        predictions_3d = binary_predictions.reshape(-1, 16, 16, 16)
        batch_size = labels_3d.shape[0]

        # Prepare lists to store point sets for each sample
        labels_points_list = []
        predictions_points_list = []

        for i in range(batch_size):
            # Get the occupied voxel indices for labels and predictions
            label_voxels = np.argwhere(labels_3d[i] == 1)
            prediction_voxels = np.argwhere(predictions_3d[i] == 1)

            # Convert voxel indices to spatial coordinates
            # Optionally, you can scale coordinates based on voxel size if necessary
            labels_points_list.append(label_voxels)
            predictions_points_list.append(prediction_voxels)

        chamfer_distances = []
        for i in range(batch_size):
            label_points = labels_points_list[i]
            prediction_points = predictions_points_list[i]
            cd = self.chamfer_distance(label_points, prediction_points)
            chamfer_distances.append(cd)
        average_cd = np.mean(chamfer_distances)
        return average_cd, chamfer_distances

    def chamfer_distance(self, point_set_1, point_set_2, grid_size=16, normalize=True):
        # Compute the distance from each point in set 1 to the nearest point in set 2
        from scipy.spatial import cKDTree

        if len(point_set_1) == 0 or len(point_set_2) == 0:
            # Handle empty point sets
            if normalize:
                return 0.0 if len(point_set_1) == 0 and len(point_set_2) == 0 else 1.0
            else:
                return 0.0 if len(point_set_1) == 0 and len(point_set_2) == 0 else 1000. #inf

        tree_1 = cKDTree(point_set_1)
        tree_2 = cKDTree(point_set_2)

        distances_1, _ = tree_1.query(point_set_2, k=1)
        distances_2, _ = tree_2.query(point_set_1, k=1)

        cd = np.mean(distances_1 ** 2) + np.mean(distances_2 ** 2)

        if normalize:
            # Normalization -- maximum cd can be 2*d_{max}**2 -- one gt and prediction and maximum separation where d_max is sqrt(3)*grid_size
            d_max = np.sqrt(3 * (grid_size - 1) ** 2)
            cd_max = 2 * (d_max ** 2)
            cd = cd / cd_max

        return cd

    def find_threshold_independent_metrics(self, predictions, labels):
        # Lists to store per-sample AUC-ROC and AUC-PR values
        aucs_roc = []
        aucs_pr = []

        # Loop through each sample
        for i in range(predictions.shape[0]):
            y_true = labels[i]  # True labels for the ith sample
            y_pred = predictions[i]  # Predictions for the ith sample

            # Check if the sample has both positive and negative examples (required for AUC calculation)
            if len(np.unique(y_true)) > 1:
                # Compute AUC-ROC and AUC-PR for the current sample
                auc_roc = roc_auc_score(y_true, y_pred)
                auc_pr = average_precision_score(y_true, y_pred)
            else:
                assert False, "Check again to make sure all samples as at least some GT as positive"
                # If only one class present in y_true, assign AUC scores as NaN or some default
                auc_roc = np.nan
                auc_pr = np.nan

            aucs_roc.append(auc_roc)
            aucs_pr.append(auc_pr)
        # Final result by averaging across all samples, ignoring NaNs
        mean_auc_roc = np.nanmean(aucs_roc)
        mean_auc_pr = np.nanmean(aucs_pr)

        return {
            'roc_auc': mean_auc_roc,
            'pr_auc': mean_auc_pr,
        }

    def calculate_iou(self, labels, binary_predictions, per_sample=False, tol=1e-10):
        if not per_sample:
            return (np.logical_and(labels, binary_predictions) + tol) / (np.logical_or(labels, binary_predictions) + tol)
        else:
            # Assumes shape is of the form (N, 4096)
            intersection = np.logical_and(binary_predictions, labels).sum(axis=1)  # Sum over each row
            union = np.logical_or(binary_predictions, labels).sum(axis=1)  # Sum over each row
            return np.mean((intersection + tol) / (union + tol))

    def compute_pose_metrics(self, pred_keypoints_3d, gt_keypoints_3d, pred_vertices, gt_vertices, indexes):
        batch_size, num_candidates, num_joints, _ = pred_keypoints_3d.shape
        gt_keypoints_3d_expanded = torch.tensor(gt_keypoints_3d).unsqueeze(1).expand(*pred_keypoints_3d.shape).contiguous()
        mpjpe, pa_mpjpe = eval_pose(
            torch.tensor(pred_keypoints_3d).view(batch_size * num_candidates, num_joints, 3),
            gt_keypoints_3d_expanded.view(batch_size * num_candidates, num_joints, 3))

        mpjpe = mpjpe.reshape(batch_size, num_candidates)
        min_mpjpe = np.min(mpjpe, axis=1)
        pa_mpjpe = pa_mpjpe.reshape(batch_size, num_candidates)
        min_pa_mpjpe = np.min(pa_mpjpe, axis=1)
        if self.save_vertices:
            min_mpjpe_indices = np.argmin(mpjpe, axis=1)
            best_pred_vertices = pred_vertices[np.arange(batch_size), min_mpjpe_indices]
            vertices = {'gt_verts': gt_vertices, 'pred_verts': best_pred_vertices, 'MPJPE_uncompressed': min_mpjpe, 'PA-MPJPE_uncompressed': min_pa_mpjpe}
        return {
            'indexes': indexes,
            'MPJPE': np.mean(min_mpjpe),
            'PA-MPJPE': np.mean(min_pa_mpjpe),
            **(vertices if self.save_vertices else {})
        }

    def compute_metrics(self, eval_pred):
        predictions, labels, indexes = eval_pred  # eval_pred contains model's predictions and labels

        if self.training_task == 'pose-only':
            assert (self.save_vertices and len(predictions) == 5) or (not self.save_vertices and len(predictions) == 3), "Check for change in the implementation or the save_smpl flags"
            if len(predictions) == 5:
                _, pred_keypoints_3d, gt_keypoints_3d, pred_vertices, gt_vertices = predictions
                return self.compute_pose_metrics(pred_keypoints_3d, gt_keypoints_3d, pred_vertices, gt_vertices, indexes)
            else:
                _, pred_keypoints_3d, gt_keypoints_3d = predictions
                return self.compute_pose_metrics(pred_keypoints_3d, gt_keypoints_3d, None, None, indexes)
        elif self.training_task == 'object-only':
            return self.compute_object_metrics(predictions, labels['interaction'], indexes)
        else:
            raise NotImplementedError

    def compute_object_metrics(self, predictions, labels, indexes):

        scenario_mapping = self.eval_dataset.get_scenario_mapping()

        # Map scenario mapping values to the index in the prediction and labels
        for parent_task in scenario_mapping:
            original_list = scenario_mapping[parent_task]
            scenario_mapping[parent_task] = [list(indexes).index(x) for x in original_list]

        # Apply sigmoid since there is no sigmoid in the model and we use BCEWithLogitLoss
        predictions = 1 / (1 + np.exp(-predictions))

        # Compute the MSE loss
        mse_loss = np.mean((predictions - labels) ** 2)

        # find_threshold_independent_metrics
        aucs = self.find_threshold_independent_metrics(predictions, labels)

        # Threshold the predictions at 0.5 to convert to binary predictions
        thresh_params = [{'thresh': 0.01, 'top_k': 5}, {'thresh': 0.05, 'top_k': 5}, {'thresh': 0.1, 'top_k': 5}, {'thresh': 0.2, 'top_k': 5}]
        binary_predictions_list = [self.ensure_top_k_ones(predictions, threshold=x['thresh'], top_k=x['top_k']) for x in thresh_params]
        metrics = {}
        for param_idx, binary_predictions in enumerate(binary_predictions_list):
            param_desc = f"thresh-{thresh_params[param_idx]['thresh']}-topk-{thresh_params[param_idx]['top_k']}"

            # Calculate IoU
            iou = self.calculate_iou(labels, binary_predictions)
            mean_iou_per_sample = self.calculate_iou(labels, binary_predictions, per_sample=True)

            # Do per scenario
            for key in scenario_mapping:
                # Also add other metrics if doing all scenario
                metrics[f"{key}_iou_{param_desc}"]= self.calculate_iou(
                    labels[scenario_mapping[key]], binary_predictions[scenario_mapping[key]],
                    per_sample=True
                )
                per_scenario_cd, _ = self.process_and_calculate_chamfer(labels[scenario_mapping[key]], binary_predictions[scenario_mapping[key]])
                metrics[f"{key}_chamfer_dist_{param_desc}"] = per_scenario_cd

            # Calculate chamfer distance
            average_cd, chamfer_distances = self.process_and_calculate_chamfer(labels, binary_predictions)

            metrics[f"iou_{param_desc}"] = iou
            metrics[f"iou_{param_desc}"] = mean_iou_per_sample
            metrics[f"chamfer_dist_{param_desc}"] = average_cd
            if self.save_locations:
                metrics[f"chamfer_per_sample_{param_desc}"] = chamfer_distances
        metrics['best_chamfer_dist'] = min(value for key, value in metrics.items() if key.startswith('chamfer_dist_'))

        if self.save_locations:
            save_data = {
                'thresh_params': thresh_params,
                'binary_preds': binary_predictions_list,
                'labels': labels
            }

        return {
            "mse_loss": mse_loss,  # Return MSE as part of the metrics
            **aucs,
            **metrics,
            "indexes": indexes,
            **(save_data if self.save_locations else {}),
        }

if __name__ == "__main__":
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args = get_args()
    if '2024' not in args.expt_name:
        expt_name_with_time = f"{current_time}_{args.expt_name}"
    else:
        expt_name_with_time = args.expt_name

    # Initialize wandb with custom project and run name
    wandb.init(project="InteractionAnticipation", name=expt_name_with_time)

    # Example configuration (replace with actual dimensions)
    training_task = args.training_task
    visual_feature_dim = 4096  # Dimension of visual features
    pose_feature_dim = 207    # Dimension of pose features
    object_vocab_size = 1204  # Size of the object vocabulary (including 0 for empty space)
    transformer_hidden_size = 768 # Size of the input embedding of the transformer
    num_transformer_layers = 6
    num_heads = 8
    output_dim = 4096
    cvae_latent_dim = 32
    location_vocab_size=17 #16+1, 0 is for unknown location
    orientation_feature_dim = 9
    use_video = args.use_video
    use_pose = args.use_pose
    use_location = args.use_location
    use_env = args.use_env
    observation_time = args.observation_time
    anticipation_time = 5.
    future_time = args.future_time
    num_decoding_samples = 5
    scenarios = "Cooking"
    learning_rate = args.learning_rate
    return_smpl_vertices = False # Set it true in the eval code to save vertices

    # Create the configuration instance
    config = CustomConfig(
        training_task=training_task,
        visual_feature_dim=visual_feature_dim,
        pose_feature_dim=pose_feature_dim,
        object_vocab_size=object_vocab_size,
        location_vocab_size=location_vocab_size,
        orientation_feature_dim=orientation_feature_dim,
        transformer_hidden_size=transformer_hidden_size,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        output_dim=output_dim,
        cvae_latent_dim=cvae_latent_dim,
        use_video=use_video,
        use_pose=use_pose,
        use_location=use_location,
        use_env=use_env,
        observation_time=observation_time,
        anticipation_time=anticipation_time,
        future_time=future_time,
        num_decoding_samples=num_decoding_samples,
        scenarios=scenarios,
        return_smpl_vertices=return_smpl_vertices,
    )

    # Add config to wandb
    wandb.config.update({
        "training_task": training_task,
        "use_video": use_video,
        "use_pose": use_pose,
        "use_location": use_location,
        "use_env": use_env,
        "future_time": future_time,
        "scenarios": scenarios,
        "learning_rate": learning_rate,
    })

    # Instantiate the model
    model = CustomTransformerModel(config)

    print(f"Total number of trainable parameters: {count_parameters(model)}")
    list_model_parameters_sorted(model)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{expt_name_with_time}/",
        num_train_epochs=3,
        learning_rate=learning_rate,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=10,
        save_steps=0.05,
        eval_steps=0.05,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,  # Ensure the best model is loaded at the end of training
        metric_for_best_model="pr_auc" if args.training_task in ['object-only', 'multi-task'] else "MPJPE",  # Metric to track for saving the best model
        greater_is_better=True if args.training_task in ['object-only', 'multi-task'] else False,
        include_inputs_for_metrics=True,
    )

    # Assuming you have train_dataset and eval_dataset ready
    # The datasets should return a dictionary with keys:
    # 'visual_features', 'pose_features', 'object_indices', 'labels'

    # Instantiate datasets (replace with your actual datasets)
    train_dataset = InteractionDataset('train', observation_time=observation_time, anticipation_time=anticipation_time, future_time=future_time, scenarios=scenarios)
    eval_dataset = InteractionDataset('val', observation_time=observation_time, anticipation_time=anticipation_time, future_time=future_time, scenarios=scenarios)

    # Instantiate the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=torch_default_data_collator,
        compute_metrics=ComputeMetricsWrapper(eval_dataset, args.training_task).compute_metrics,
    )

    # Start training
    trainer.train()
