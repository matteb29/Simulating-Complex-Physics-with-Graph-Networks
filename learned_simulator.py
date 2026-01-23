"""
Graph Network-based Physics Simulator.

This module implements a Learnable Simulator based on the "Encoder-Processor-Decoder"
architecture. It includes:
    A Normalizer class for online computation of input/output statistics.
    A Simulator class orchestrating the GNN, feature extraction, and physical integration.

References:
    "Learning to Simulate Complex Physics with Graph Networks", Sanchez-Gonzalez et al.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

# Internal import
from graph_net import EncoderProcessorDecoder
from torch_cluster import radius_graph


class Normalizer(nn.Module):
    """
    Feature normalizer that accumulates statistics online.

    It computes the running mean and standard deviation of the input data
    using an online accumulation strategy. It is implemented as an nn.Module
    to ensure buffers (mean, std) are automatically moved to the correct device
    and saved within the model state_dict.

    """

    def __init__(
        self, size: int, max_accumulation: int = 10**6, std_epsilon: float = 1e-8
    ):
        """
        Initializes the Normalizer.

        Args:
            size: The dimension of the features to normalize.
            max_accumulation: The maximum number of samples to use for statistics accumulation.
                              After this count, stats are frozen.
            std_epsilon: A small epsilon value for numerical stability to avoid division by zero.
        """
        super().__init__()

        self.max_accumulation = max_accumulation
        self.std_epsilon = std_epsilon
        self.accumulating = True

        # Buffers for online statistics accumulation
        self.register_buffer("accumulated_sum", torch.zeros(size))
        self.register_buffer("accumulated_sum_squared", torch.zeros(size))
        self.register_buffer("accumulated_count", torch.zeros(1))

        # The final statistics used for normalization
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std_dev", torch.ones(size))

    def update_stats(self, batch: torch.Tensor) -> None:
        """
        Updates the running statistics with a new batch of data.

        This method should only be called during the training phase where
        `self.accumulating` is True.

        Args:
            batch: Input tensor of shape [B, size].
        """
        if not self.accumulating:
            return

        count = batch.shape[0]
        self.accumulated_count += count

        self.accumulated_sum += batch.sum(dim=0)
        self.accumulated_sum_squared += (batch**2).sum(dim=0)

        # Update running mean: E[x]
        self.mean[:] = self.accumulated_sum / self.accumulated_count

        # Update running std: sqrt(E[x^2] - E[x]^2)
        # Note: We clamp the variance to be non-negative before sqrt.
        variance = (
            self.accumulated_sum_squared / self.accumulated_count
        ) - self.mean**2
        variance = torch.max(variance, torch.tensor(0.0, device=variance.device))

        self.std_dev[:] = torch.sqrt(variance + self.std_epsilon)

        if self.accumulated_count > self.max_accumulation:
            self.accumulating = False

    def normalize(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Applies z-score normalization.

        Args:
            batch: Input tensor.

        Returns:
            Normalized tensor of the same shape.
        """
        return (batch - self.mean) / self.std_dev

    def inverse(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse transformation (denormalization).

        Args:
            batch: Normalized input tensor.

        Returns:
            Denormalized tensor.
        """
        return batch * self.std_dev + self.mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.normalize(x)


class Simulator(nn.Module):
    """
    Physics Simulator using Graph Neural Networks.

    This module handles the extraction of graph features from particle features,
    predicts accelerations using an Encoder-Processor-Decoder architecture,
    and integrates the system state using an Euler integrator.
    """

    def __init__(
        self,
        particle_dimensions: int = 2,
        node_in: int = 14,
        edge_in: int = 3,
        latent_size: int = 128,
        num_layers: int = 2,
        message_passing_steps: int = 10,
        connectivity_radius: float = 0.015,
        boundaries: List[List[float]] = None,
        normalization_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        num_particle_types: int = 9,
        dim_particle_embedding: int = 16,
    ):
        """
        Args:
            particle_dimensions: Dimensionality of the physical space (2D).
            node_in: Dimension of the raw node features (velocities + boundary distances).
            edge_in: Dimension of the raw edge features (relative displacement + distance).
            latent_size: Size of the latent vectors in the MLP layers.
            num_layers: Number of hidden layers in the MLPs.
            message_passing_steps: Number of Message Passing steps in the Processor.
            connectivity_radius: Radius for graph construction (interaction range).
            boundaries: Box boundaries [[min_x, max_x], [min_y, max_y]].
            normalization_stats: Dictionary containing pre-computed 'mean' and 'std'
                                 for 'nodes', 'edges', and 'accelerations' or 'velocities'.
            num_particle_types: Number of distinct particle types (materials).
            dim_particle_embedding: Size of the learnable embedding for particle types.
        """
        super().__init__()

        self.connectivity_radius = connectivity_radius
        self.particle_dimension = particle_dimensions
        self.num_particle_types = num_particle_types

        # Boundary information is fixed; registered as buffer to persist with model.
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float))

        self.actual_node_dim = node_in

        # Learnable Particle Type Embedding
        if self.num_particle_types > 1:
            self.particle_embedding = nn.Embedding(
                num_particle_types, dim_particle_embedding
            )
            self.actual_node_dim += dim_particle_embedding

        # Initialize Normalizers for inputs (nodes, edges) and output (acceleration)
        self.node_normalizer = Normalizer(node_in)
        self.edge_normalizer = Normalizer(edge_in)
        self.output_normalizer = Normalizer(particle_dimensions)

        # Load pre-computed normalization statistics if provided
        self._load_normalization_stats(normalization_stats)

        # Core GNN (Encoder-Processor-Decoder)
        self.framework = EncoderProcessorDecoder(
            input_node_dim=self.actual_node_dim,
            input_edge_dim=edge_in,
            latent_size=latent_size,
            num_layers_mlp=num_layers,
            num_message_passing_steps=message_passing_steps,
            output_size=particle_dimensions,
        )

    def _load_normalization_stats(self, normalization_stats: Optional[Dict]) -> None:
        """Helper to safely load normalization statistics into buffers."""
        if normalization_stats is None:
            return

        normalizers_map = {
            "accelerations": self.output_normalizer,
            "nodes": self.node_normalizer,
            "edges": self.edge_normalizer,
        }

        for key, normalizer in normalizers_map.items():
            if key in normalization_stats:
                stats = normalization_stats[key]

                # size matching test
                if normalizer.mean.shape != stats["mean"].shape:
                    raise ValueError(
                        f"Dimension mismatch for normalization stats: {key}"
                    )

                device = normalizer.mean.device

                # Load Mean
                normalizer.mean.data = stats["mean"].to(device)

                # Load Std (avoid KeyError handling both 'std' and 'std_dev')
                std_key = "std" if "std" in stats else "std_dev"
                if std_key in stats:
                    normalizer.std_dev.data = stats[std_key].to(device)

                # Disable accumulation since we loaded "metadata" stats
                normalizer.accumulating = False
                print(f"DEBUG: Loaded normalization stats for {key}")

    def get_edge_features(
        self, positions: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Constructs edge features based on relative positions.

        Args:
            positions: Tensor of shape [N, D].
            edge_index: Tensor of shape [2, Number of Edges] (Coordinate list format).

        Returns:
            edge_features: Tensor of shape [Number of Edges, edge_dim].
        """
        sender_pos = positions[edge_index[0]]
        receiver_pos = positions[edge_index[1]]

        relative_pos = sender_pos - receiver_pos
        # Normalize distance by connectivity radius for scale invariance
        relative_pos_norm = relative_pos / self.connectivity_radius
        distance = torch.norm(relative_pos_norm, dim=-1, keepdim=True)

        edge_features = torch.cat([relative_pos_norm, distance], dim=-1)
        return edge_features

    def get_boundary_distances(self, current_positions: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance of each particle to the box boundaries.

        The distances are normalized to the connectivity radius and clipped to [-1, 1].

        Args:
            current_positions: Tensor of shape [N, D].

        Returns:
            Clipped distances tensor of shape [N, 2*D].
        """
        lower_bounds = self.boundaries[:, 0].unsqueeze(0)  # [1, D]
        upper_bounds = self.boundaries[:, 1].unsqueeze(0)  # [1, D]

        distances_to_lower = current_positions - lower_bounds  # [N, D]
        distances_to_upper = upper_bounds - current_positions  # [N , D]

        distances_to_boundaries = torch.cat(
            [distances_to_lower, distances_to_upper], dim=-1
        )

        # Clamp distance between -1 and 1 normalized relative to the radius
        norm_clipped_distances = torch.clamp(
            distances_to_boundaries / self.connectivity_radius, min=-1.0, max=1.0
        )

        return norm_clipped_distances

    def _construct_graph(
        self, positions: torch.Tensor, batch: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Constructs the connectivity graph, handling hardware compatibility.

        Attempts to build the graph on the current device.
        If the specific CUDA kernel is missing , it automatically
        falls back to CPU execution without crashing.
        NOTE: This was a problem we faced since we worked with SWAN ambient

        Args:
            positions: [N, D]
            batch: [N]

        Returns: edge index [2, Number of Edges]
        """
        try:
            # First attempt: Optimal path (GPU if available)
            return radius_graph(
                positions, r=self.connectivity_radius, batch=batch, loop=False
            )
        except RuntimeError:
            # Fallback attempt: Execute on CPU, then move result back to original device
            return radius_graph(
                positions.cpu(),
                r=self.connectivity_radius,
                batch=batch.cpu() if batch is not None else None,
                loop=False,
            ).to(positions.device)

    def predict_accelerations(
        self,
        position_sequence: torch.Tensor,
        particle_types: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            position_sequence: Historical positions [N, Time, D].
            particle_types: Particle material types [N].
            batch: Batch vector [N] mapping particles to simulations

        Returns:
            Unnormalized acceleration predictions [N, D].
        """
        current_positions = position_sequence[:, -1]

        # Compute recent velocities from the sequence
        # velocity[t] = pos[t] - pos[t-1]
        velocities = [
            position_sequence[:, t] - position_sequence[:, t - 1]
            for t in range(1, position_sequence.shape[1])
        ]
        velocities_tensor = torch.cat(velocities, dim=-1)

        # Compute boundary features
        boundary_distances = self.get_boundary_distances(current_positions)

        # Concatenate node inputs
        node_input_features = torch.cat([velocities_tensor, boundary_distances], dim=-1)

        # Construct the Interaction Graph
        edge_index = self._construct_graph(current_positions, batch)
        edge_input_features = self.get_edge_features(current_positions, edge_index)

        # Update normalization stats (only during training)
        if self.training:
            self.node_normalizer.update_stats(node_input_features)
            self.edge_normalizer.update_stats(edge_input_features)

        #  Normalize Inputs
        norm_node_input_features = self.node_normalizer.normalize(node_input_features)
        norm_edge_input_features = self.edge_normalizer.normalize(edge_input_features)

        # Append Particle Embeddings (if applicable)
        if self.num_particle_types > 1:
            embedded_types = self.particle_embedding(particle_types)
            norm_node_input_features = torch.cat(
                [norm_node_input_features, embedded_types], dim=-1
            )

        # Forward Pass (GNN)
        prediction_norm = self.framework(
            norm_node_input_features, edge_index, norm_edge_input_features
        )

        # Denormalize output to get physical acceleration
        return self.output_normalizer.inverse(prediction_norm)

    def euler_integrator(
        self,
        positions_sequence: torch.Tensor,
        particle_types: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs a single step of Euler integration based on GNN predictions.

        Update rule:
        $$ v_t = p_t - p_{t-1} $$
        $$ p_{t+1} = p_t + v_t + a_{pred} $$

        Args:
            positions_sequence: Input position history [N, T, D].
            particle_types: [N].
            batch: [N].

        Returns:
            Next positions [N, D].
        """
        accelerations = self.predict_accelerations(
            positions_sequence, particle_types, batch
        )

        current_positions = positions_sequence[:, -1]
        last_positions = positions_sequence[:, -2]

        # Estimate current velocity (assuming dt=1)
        current_velocities = current_positions - last_positions

        next_positions = current_positions + current_velocities + accelerations

        return next_positions

    def helper_training(
        self,
        next_positions: torch.Tensor,
        positions_sequence_noise: torch.Tensor,
        positions_sequence: torch.Tensor,
        particle_types: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training step with Noise Injection.

        This method adds noise to the input trajectory but trains the model to
        predict the correct target acceleration that compensates for this noise.
        This improves rollout stability (according to Sanchez-Gonzalez's paper).

        Args:
            next_positions: Ground truth positions at t+1 [N, D].
            positions_sequence_noise: Sampled noise added to inputs [N, T, D].
            positions_sequence: Ground truth input history [N, T, D].
            particle_types: [N].

        Returns:
            Tuple(predicted_normalized_acc, target_normalized_acc)
        """

        # Inject noise into input sequence
        noisy_pos_sequence = positions_sequence + positions_sequence_noise
        current_noisy_positions = noisy_pos_sequence[:, -1]

        # Recompute input features based on noisy positions
        velocities = [
            noisy_pos_sequence[:, t] - noisy_pos_sequence[:, t - 1]
            for t in range(1, noisy_pos_sequence.shape[1])
        ]

        noisy_velocities = torch.cat(velocities, dim=-1)
        boundary_positions = self.get_boundary_distances(current_noisy_positions)

        noisy_physical_features = torch.cat(
            [noisy_velocities, boundary_positions], dim=-1
        )

        # Update node stats (based on noisy inputs to match inference distribution shifts)
        if self.training:
            self.node_normalizer.update_stats(noisy_physical_features)

        norm_physical_features = self.node_normalizer.normalize(noisy_physical_features)

        # Append embeddings
        if self.num_particle_types > 1:
            embedded_types = self.particle_embedding(particle_types)
            norm_nodes_input_features = torch.cat(
                [norm_physical_features, embedded_types], dim=-1
            )
        else:
            norm_nodes_input_features = norm_physical_features

        # Construct Graph on noisy positions
        noisy_edge_index = self._construct_graph(current_noisy_positions, batch)

        noisy_edge_input = self.get_edge_features(
            current_noisy_positions, noisy_edge_index
        )

        if self.training:
            self.edge_normalizer.update_stats(noisy_edge_input)

        norm_edges_input_features = self.edge_normalizer.normalize(noisy_edge_input)

        # GNN Forward Pass
        pred_norm_accelerations = self.framework(
            norm_nodes_input_features, noisy_edge_index, norm_edges_input_features
        )

        # Compute Target Acceleration (Noise Cancellation)
        # We want the model to predict an acceleration that takes us from
        # (current_noisy_pos + noisy_vel) to (true_next_pos).

        # Correct the "next" target by the noise at the last step to keep consistency
        next_pos_target_adjusted = next_positions + positions_sequence_noise[:, -1]

        # Target Velocity = (Target Next Pos) - (Current Noisy Pos)
        target_velocities = next_pos_target_adjusted - noisy_pos_sequence[:, -1]

        # Previous Velocity (from noisy input)
        previous_velocities_noisy = (
            noisy_pos_sequence[:, -1] - noisy_pos_sequence[:, -2]
        )

        # Target Acceleration = Target Delta V
        target_accelerations_unnormalized = (
            target_velocities - previous_velocities_noisy
        )

        # Update output stats
        if self.training:
            self.output_normalizer.update_stats(target_accelerations_unnormalized)

        target_norm_accelerations = self.output_normalizer.normalize(
            target_accelerations_unnormalized
        )

        return pred_norm_accelerations, target_norm_accelerations

    def forward(
        self,
        positions_sequence: torch.Tensor,
        particle_types: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass acts as a standard integrator step for inference.
        """
        return self.euler_integrator(positions_sequence, particle_types, batch)
