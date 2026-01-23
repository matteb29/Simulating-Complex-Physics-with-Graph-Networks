"""
Graph Network-based Physics Simulator.

This module implements the pipeling of both training and evaluation of our Simulator,
divided in two distint functions.

References:
    "Learning to Simulate Complex Physics with Graph Networks", Sanchez-Gonzalez et al.

"""

import collections
import glob
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

# Internal import
import learned_simulator
import noise_utils
import numpy as np
import reading_utils
import torch
from absl import app, flags
from torch.utils.data import DataLoader

# Configuration Flags
flags.DEFINE_enum(
    "mode", "train", ["train", "eval_rollout"], help="Train model or rollout evaluation"
)
flags.DEFINE_enum(
    "eval_split",
    "test",
    ["train", "valid", "test"],
    help="Split to use when running evaluation",
)
flags.DEFINE_string("data_path", None, help="The dataset directory")
flags.DEFINE_integer("batch_size", 2, help="The batch size")
flags.DEFINE_integer("num_steps", int(2e6), help="Number of training steps")
flags.DEFINE_float("noise_std", 6.7e-4, help="The standard deviation of the noise")
flags.DEFINE_string(
    "model_path",
    "model_checkpoints",
    help="The path for saving checkpoints of the model",
)
flags.DEFINE_string(
    "output_path", "rollouts", help="The path for saving outputs (e.g. rollouts)"
)
flags.DEFINE_integer("log_steps", 500, help="Log info every N steps")
flags.DEFINE_integer("save_steps", 5000, help="Save model every N steps")

FLAGS = flags.FLAGS

# Named tuple per statistiche
Stats = collections.namedtuple("Stats", ["mean", "std"])

# Constants
INPUT_SEQUENCE_LENGTH = 6
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
EMBEDDING_DIM = 16


def get_kinematic(particle_types: torch.Tensor) -> torch.Tensor:
    """Identifies kinematic particles (e.g., walls) based on type ID.

    Args:
        particle_types: Tensor of shape [N_particles] containing type indices.

    Returns:
        Boolean mask of shape [N_particles] where True indicates a kinematic particle.
    """
    return torch.eq(particle_types, KINEMATIC_PARTICLE_ID)


def load_metadata(data_path: str) -> Dict[str, Any]:
    """Loads dataset metadata from JSON file.

    Args:
        data_path: Directory containing the metadata.json file.

    Returns:
        Dictionary containing simulation bounds, sequence length, and normalization stats.
    """
    metadata_path = os.path.join(data_path, "metadata.json")
    with open(metadata_path, "rt", encoding="utf-8") as fp:
        return json.loads(fp.read())


def get_device() -> torch.device:
    """Selects the compute device, CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(data_path: str, split: str) -> List[Dict[str, Any]]:
    """Loads simulation data (particles, positions, types).

    Args:
        data_path: Root directory of the dataset.
        split: One of 'train', 'valid', 'test'.

    Returns:
        A list of dictionaries, where each dictionary represents a trajectory.
    """

    # Check for single file structure e.g. data_path/train.npz
    single_file_path = os.path.join(data_path, split)

    if os.path.isfile(single_file_path):
        with np.load(single_file_path, allow_pickle=True) as data:
            return [dict(x) for x in data["data"]]

    # Check for directory structure e.g. data_path/train/*.npz
    split_directory = os.path.join(data_path, split)
    if os.path.isdir(split_directory):

        files = sorted(glob.glob(os.path.join(split_directory, "*.npz")))

        if len(files) > 0:
            print(f"Found {len(files)} files in split: {split}")
            data_list = []

            for f in files:
                with np.load(f, allow_pickle=True) as data:

                    trajectory = {
                        "position": data["pos"],
                        "particle_type": data["particle_type"],
                    }

                    if "global" in data:
                        trajectory["global_context"] = data["global"]

                    data_list.append(trajectory)

            return data_list

    raise ValueError(f"No data found in {data_path} for split {split}")


def build_model(
    metadata: Dict[str, Any], noise_std: float, device: torch.device
) -> torch.nn.Module:
    """Initializes the Graph Network Simulator with normalized statistics.

    Args:
        metadata: Dataset metadata containing global stats (mean/std).
        noise_std: Standard deviation of noise. Added to stats to prevent numerical
                   instabilities during training with noisy inputs.
        device: Torch device.

    Returns:
        An initialized Simulator model (nn.Module).
    """

    # Helper function to convert data to tensor on device
    def convert_data(data):
        return torch.tensor(data, dtype=torch.float32, device=device)

    # Load accelerations stats
    acc_mean = convert_data(metadata["acc_mean"])

    # The model expects inputs to be normalized. Since we add noise during training,
    # we adjust the normalization std to include this noise variance.
    acc_std = torch.sqrt(convert_data(metadata["acc_std"]) ** 2 + noise_std**2)

    # Load velocities stats
    vel_mean = convert_data(metadata["vel_mean"])
    vel_std = torch.sqrt(convert_data(metadata["vel_std"]) ** 2 + noise_std**2)

    # Dictionary of stats for normalization
    normalization_stats = {
        "accelerations": {"mean": acc_mean, "std": acc_std},
        "velocity": {"mean": vel_mean, "std": vel_std},
    }

    # Handle optional context stats
    if "context_mean" in metadata and "context_std" in metadata:
        context_mean = convert_data(metadata["context_mean"])
        context_std = convert_data(metadata["context_std"])
        normalization_stats["context"] = {"mean": context_mean, "std": context_std}

    # Define dimensionality
    dim = metadata["dim"]

    # Calculate input feature size based on sequence length
    # Features = (Past Velocities) + (Distance to boundaries)
    # Past Velocities count = Sequence_length - 1 (current position is target, not feature)
    computed_node_in = (INPUT_SEQUENCE_LENGTH - 1) * dim + (2 * dim)

    print(f"Building model: Dim={dim}, Input Features={computed_node_in}")

    # Initialize the Simulator
    model = learned_simulator.Simulator(
        particle_dimensions=dim,
        node_in=computed_node_in,
        edge_in=dim + 1,
        latent_size=128,
        num_layers=2,
        message_passing_steps=10,
        connectivity_radius=metadata["default_connectivity_radius"],
        boundaries=np.array(metadata["bounds"]),
        normalization_stats=normalization_stats,
        num_particle_types=NUM_PARTICLE_TYPES,
        dim_particle_embedding=EMBEDDING_DIM,
    ).to(device)

    return model


def collate_function(
    batch: List[Dict[str, Any]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Custom collate function for GNN DataLoaders.

    Flattens a batch of simulations into a single large graph (disjoint union).
    A `batch_index` tensor is created to track which particle belongs to which simulation.

    Args:
        batch: List of trajectory dictionaries from load_data.

    Returns:
        Tuple containing:
        - batch_position: [Total_Particles, Time, Dim]
        - batch_index: [Total_Particles] (Simulation ID mapping)
        - batch_type: [Total_Particles]
        - batch_context: [Total_Particles, Context_Dim] or None
    """
    position_list = []
    type_list = []
    context_list = []
    batch_index_list = []

    for i, item in enumerate(batch):
        # Convert positions: [Time, Particles, Dim] -> [Particles, Time, Dim]
        # We use torch.as_tensor to avoid unnecessary copies if input is already compatible
        positions = torch.as_tensor(item["position"], dtype=torch.float32).permute(
            1, 0, 2
        )
        position_list.append(positions)

        particle_type = torch.as_tensor(item["particle_type"], dtype=torch.long)
        type_list.append(particle_type)

        # Create batch index (simulation ID) for each particle
        # If simulation 'i' has 100 particles, we create a vector of [i, i, ..., i] (len 100)
        num_particles = positions.shape[0]
        batch_index = torch.full((num_particles,), i, dtype=torch.long)
        batch_index_list.append(batch_index)

        # Handle global context
        if "global_context" in item:
            # Assuming context is per-simulation [Context_Dim]
            # We repeat it for each particle to allow easy concatenation
            context = torch.as_tensor(item["global_context"], dtype=torch.float32)
            # Reshape to [1, Context_Dim] then repeat
            context_expanded = context.unsqueeze(0).repeat(num_particles, 1)
            context_list.append(context_expanded)

    # Concatenate all lists to create the batched tensors
    batch_position = torch.cat(position_list, dim=0)
    batch_type = torch.cat(type_list, dim=0)
    batch_index = torch.cat(batch_index_list, dim=0)

    batch_context = None
    if len(context_list) > 0:
        batch_context = torch.cat(context_list, dim=0)

    return batch_position, batch_index, batch_type, batch_context


def train(metadata: Dict[str, Any], device: torch.device) -> None:
    """
    Executes the training loop with One-Step-Ahead prediction.

    Training is performed by steps (not epochs), with noise injection to make
    the model robust to error accumulation during rollouts.
    """

    print(f"Loading training data from {FLAGS.data_path}...")

    # Load simulation for training
    simulation_training = load_data(FLAGS.data_path, "train")

    # Create the dataset
    dataset_obj = reading_utils.SimulationDataset(
        simulation_training, window_length=INPUT_SEQUENCE_LENGTH + 1
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset_obj,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=collate_function,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize model
    model = build_model(metadata, FLAGS.noise_std, device)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1e6))

    model.train()

    step = 0
    loss_history = []

    # Infinite iterator for step-based training
    data_iter = iter(dataloader)

    print(f"Start training for {FLAGS.num_steps} steps...")

    while step < FLAGS.num_steps:
        try:
            batch_data = next(data_iter)
        except StopIteration:
            # Restart iterator if dataset ends
            data_iter = iter(dataloader)
            batch_data = next(data_iter)

        # Move data to device
        batch_pos, batch_index, batch_type, batch_context = batch_data
        batch_pos = batch_pos.to(device)
        batch_index = batch_index.to(device)
        batch_type = batch_type.to(device)
        if batch_context is not None:
            batch_context = batch_context.to(device)

        # Split input (past sequence) and target (next step)
        # batch_pos shape: [Particles, Time, Dim]
        input_positions = batch_pos[:, :-1, :]
        target_positions = batch_pos[:, -1, :]

        # Noise Injection
        # Add noise to input positions to improve rollout stability
        noisy_input_positions = input_positions.clone()

        # Get random walk noise
        sampled_noise = noise_utils.get_random_walk_noise(
            noisy_input_positions, noise_std_last_step=FLAGS.noise_std
        ).to(device)

        # Mask noise for kinematic particles (walls don't vibrate)
        is_kinematic = get_kinematic(batch_type)
        sampled_noise[is_kinematic] = 0.0

        noisy_input_positions += sampled_noise

        # Optimization Step
        optimizer.zero_grad()

        # Compute predictions
        # The helper_training method calculates predicted acceleration and target acceleration
        predicted_acc, target_acc = model.helper_training(
            next_positions=target_positions,
            positions_sequence=noisy_input_positions,
            positions_sequence_noise=sampled_noise,
            particle_types=batch_type,
            batch=batch_index,
        )

        # Calculate Loss (MSE on acceleration)
        loss = (predicted_acc - target_acc) ** 2
        loss = loss.sum(dim=-1)  # Sum over coordinate dimensions (x, y)

        # Average loss only over non-kinematic particles
        num_non_kinematic = (~is_kinematic).sum()
        masked_loss = loss[~is_kinematic].sum() / (num_non_kinematic + 1e-8)

        masked_loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if step % FLAGS.log_steps == 0:
            print(f"Step {step}/{FLAGS.num_steps} | Loss: {masked_loss.item():.6f}")
            loss_history.append({"step": step, "loss": masked_loss.item()})

        # Save Checkpoints
        if step % FLAGS.save_steps == 0 and step > 0:
            os.makedirs(FLAGS.model_path, exist_ok=True)
            path = os.path.join(FLAGS.model_path, f"model_{step}.pt")
            torch.save(model.state_dict(), path)

            # Save loss history
            with open(os.path.join(FLAGS.model_path, "loss_history.json"), "w", encoding="utf-8") as f:
                json.dump(loss_history, f, indent=4)
            print(f"Saved checkpoint: {path}")

        step += 1

    # Final Save
    torch.save(model.state_dict(), os.path.join(FLAGS.model_path, "final_model.pt"))
    print("Training finished.")


def rollout(metadata: Dict[str, Any], device: torch.device) -> None:
    """Performs autoregressive rollout evaluation on the test set."""

    print(f"Starting rollout evaluation on split: {FLAGS.eval_split}")

    # Initialize model with zero noise for evaluation
    model = build_model(metadata, noise_std=0.0, device=device)

    if FLAGS.model_path.endswith(".pt"):
        if not os.path.exists(FLAGS.model_path):
            raise ValueError(f"Checkpoint file not found: {FLAGS.model_path}")
        checkpoint_to_load = FLAGS.model_path
    else:
        # If it is a directory, look for the latest checkpoint inside
        checkpoints = glob.glob(os.path.join(FLAGS.model_path, "*.pt"))
        if not checkpoints:
            raise ValueError(f"No checkpoints found in directory {FLAGS.model_path}")

        # Pick the latest modified file
        checkpoint_to_load = max(checkpoints, key=os.path.getctime)

    print(f"Loading model checkpoint: {checkpoint_to_load}")
    model.load_state_dict(torch.load(checkpoint_to_load, map_location=device))
    model.eval()

    # Load data
    data = load_data(FLAGS.data_path, FLAGS.eval_split)
    os.makedirs(FLAGS.output_path, exist_ok=True)

    with torch.no_grad():
        for i, example in enumerate(data):
            # Example structure: {'position': (Time, Part, Dim), 'particle_type': ...}
            positions = torch.tensor(
                example["position"], device=device, dtype=torch.float32
            )
            particle_type = torch.tensor(
                example["particle_type"], device=device, dtype=torch.long
            )

            # Extract dimensions
            # positions shape is [Time, Particles, Dim]
            total_time = positions.shape[0]
            num_particles = positions.shape[1]

            # Initial input sequence (first 6 frames)
            # Reshape to [Particles, Sequence, Dim] for the model
            current_positions = positions[:INPUT_SEQUENCE_LENGTH].permute(1, 0, 2)

            predictions_list = []

            # Identify kinematic particles once
            kinematic_mask = get_kinematic(particle_type)  # [Particles]

            print(f"Rolling out trajectory {i} ({total_time} steps)...")

            # Autoregressive loop
            for t in range(INPUT_SEQUENCE_LENGTH, total_time):
                # Create a dummy batch index (all particles belong to simulation 0)
                batch_index = torch.zeros(
                    num_particles, dtype=torch.long, device=device
                )

                # Predict next position
                # Model takes [Particles, Seq, Dim] -> Returns [Particles, Dim]
                next_position_pred = model(
                    positions_sequence=current_positions,
                    particle_types=particle_type,
                    batch=batch_index,
                )

                # Enforce boundary conditions / Kinematic particles
                # If kinematic, use ground truth (don't move walls). Else use prediction.
                ground_truth_current = positions[t]

                # where(condition, if_true, if_false)
                next_position_corrected = torch.where(
                    kinematic_mask.unsqueeze(-1),  # Broadcast to [Particles, 1]
                    ground_truth_current,  # True: use GT
                    next_position_pred,  # False: use Pred
                )

                predictions_list.append(next_position_corrected.cpu().numpy())

                # Update sliding window for next step:
                # Remove oldest frame ([:, 1:, :]) and append new prediction
                next_pos_expanded = next_position_corrected.unsqueeze(
                    1
                )  # [Part, 1, Dim]
                current_positions = torch.cat(
                    [current_positions[:, 1:, :], next_pos_expanded], dim=1
                )

            # Save results
            output_dict = {
                "initial_positions": positions[:INPUT_SEQUENCE_LENGTH].cpu().numpy(),
                "predicted_rollout": np.stack(
                    predictions_list
                ),  # [Time-Seq, Part, Dim]
                "ground_truth_rollout": positions[INPUT_SEQUENCE_LENGTH:].cpu().numpy(),
                "particle_types": particle_type.cpu().numpy(),
            }

            save_name = os.path.join(FLAGS.output_path, f"rollout_{i}.pkl")
            with open(save_name, "wb") as f:
                pickle.dump(output_dict, f)

            print(f"Saved rollout to {save_name}")


def main(_):
    """
    Main entry point governed by absl.app.
    """

    device = get_device()
    print(f"Running on device: {device}")

    if FLAGS.data_path is None:
        raise ValueError("You must provide a --data_path")

    metadata = load_metadata(FLAGS.data_path)

    if FLAGS.mode == "train":
        train(metadata, device)

    elif FLAGS.mode == "eval_rollout":
        rollout(metadata, device)


if __name__ == "__main__":
    app.run(main)
