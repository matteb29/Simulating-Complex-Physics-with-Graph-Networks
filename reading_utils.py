"""
Graph Network-based Physics Simulator.

This module implements a Dataset that retrieves simulation windows on-the-fly.
This is done not duplicate data in memory,
allowing for training on datasets much larger than available RAM.

References:
    "Learning to Simulate Complex Physics with Graph Networks", Sanchez-Gonzalez et al.
"""

import bisect
from typing import Dict, List
import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    """
    Memory-efficient Dataset that slices trajectories on demand.

    Instead of pre-computing all windows (which multiplies memory usage by window_length),
    this class stores a cumulative index map. When __getitem__ is called, it:
     Locates the correct simulation index using binary search.
     Calculates the local offset within that simulation.
     Returns a view/slice of the tensors without copying data unnecessarily.
    """

    def __init__(
        self, data_list: List[Dict[str, torch.Tensor]], window_length: int = 7
    ):
        """
        Args:
            data_list: List of dictionaries containing full trajectories.
            window_length: The temporal window size for the graph network input.
        """
        self.data_list = data_list
        self.window_length = window_length

        # Pre-calculate the number of valid windows for each simulation
        self.n_windows_per_sim = []

        print(f"INFO: Indexing {len(data_list)} simulations...")

        for sim in data_list:
            num_steps = sim["position"].shape[0]
            # Valid windows = T - Window + 1
            num_valid_windows = num_steps - window_length + 1

            # Handle edge case: simulation shorter than window
            if num_valid_windows < 1:
                # We count 0 samples for this simulation (it will be skipped)
                self.n_windows_per_sim.append(0)
            else:
                self.n_windows_per_sim.append(num_valid_windows)

        # Create a cumulative sum array (prefix sum).
        # This allows us to map a global index (e.g., 500) to a specific simulation.
        # Example: sims have [100, 100, 100] windows. cumsum -> [100, 200, 300].
        self.cumulative_sizes = (
            torch.tensor(self.n_windows_per_sim).cumsum(dim=0).tolist()
        )

        self.total_samples = self.cumulative_sizes[-1] if self.cumulative_sizes else 0
        print(
            f"INFO: Lazy Dataset ready. Virtual training samples: {self.total_samples}"
        )

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves the window corresponding to the global index `idx`.
        Uses binary search for O(log N) complexity.
        """

        # Find which simulation this index belongs to.
        # bisect_right returns the insertion point to maintain order.
        simulation_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        # Find the local index within that simulation.
        if simulation_idx == 0:
            sample_internal_idx = idx
        else:
            # Subtract the total count of previous simulations
            sample_internal_idx = idx - self.cumulative_sizes[simulation_idx - 1]

        # Retrieve the reference to the full simulation data
        simulation_data = self.data_list[simulation_idx]

        # Slice the tensors on-the-fly
        # Position Tensor: [Time, N, D] -> Slice: [Window, N, D]
        pos_window = simulation_data["position"][
            sample_internal_idx : sample_internal_idx + self.window_length
        ]

        # Particle Type: [N]. Static, so we just return as it is.
        # The DataLoader will handle batching later.
        particle_type = simulation_data["particle_type"]

        sample = {"position": pos_window, "particle_type": particle_type}

        # Handle global context if present
        if "step_context" in simulation_data:
            # Context: [Time, Context_Dim] -> Slice: [Window, Context_Dim]
            context_window = simulation_data["step_context"][
                sample_internal_idx : sample_internal_idx + self.window_length
            ]
            sample["step_context"] = context_window

        return sample
