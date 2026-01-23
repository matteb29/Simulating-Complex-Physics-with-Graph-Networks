"""
Graph Network-based Physics Simulator.

This module provides functions to generate noise sequences (random walks) 
to be added to the input of the Graph Network during training.
This technique forces the model to correct its own prediction errors over long rollouts.

Reference:
    "Learning to Simulate Complex Physics with Graph Networks", Sanchez-Gonzalez et al.
"""

import torch


def get_random_walk_noise(
    positions_sequence: torch.Tensor, noise_std_last_step: float
) -> torch.Tensor:
    """
    Generates a random walk noise sequence for input positions.

    The noise is generated such that it starts at 0 and accumulates over time,
    simulating drift error. The magnitude is scaled so that the standard deviation
    of the noise at the final step matches `noise_std_last_step`.


    Args:
        positions_sequence: The ground truth positions tensor of shape [Batch, Time, Dim].
                            Used to determine the shape and device of the noise.
        noise_std_last_step: The target standard deviation of the accumulated noise
                             at the last time step.

    Returns:
        position_sequence_noise: Tensor of shape [Batch, Time, Dim] containing the
                                 generated random walk noise (zeros at t=0).
    """

    # Extract dimensions
    # Shape: [Batch, Time, Dim]
    batch_size, num_time_steps, dim = positions_sequence.shape
    device = positions_sequence.device
    dtype = positions_sequence.dtype

    # We need velocities for T-1 steps
    num_velocity_steps = num_time_steps - 1

    # Compute the standard deviation per step to satisfy the total variance constraint.
    # Var(Sum(X_i)) = Sum(Var(X_i)) = N * sigma_step^2.
    # We want N * sigma_step^2 = sigma_final^2  so we have sigma_step = sigma_final / sqrt(N)
    single_step_std = noise_std_last_step / (num_velocity_steps**0.5)

    # Generate white noise for velocities (uncorrelated steps)
    # Shape: [Batch, Time-1, Dim]
    velocity_noise = (
        torch.randn((batch_size, num_velocity_steps, dim), device=device, dtype=dtype)
        * single_step_std
    )

    # Integrate velocity noise over time to get position noise (Random Walk)
    # Shape: [Batch, Time-1, Dim]
    position_noise_integrated = torch.cumsum(velocity_noise, dim=1)

    # Create zero noise for the initial state (t=0)
    # The simulation assumes exact knowledge of the initial state.
    start_zero_noise = torch.zeros((batch_size, 1, dim), device=device, dtype=dtype)

    # Concatenate to match the full sequence shape
    # Shape: [Batch, Time, Dim]
    position_sequence_noise = torch.cat(
        [start_zero_noise, position_noise_integrated], dim=1
    )

    return position_sequence_noise
