"""
Graph Network-based Physics Simulator - Visualization Module.

This module provides tools to render the rollout results (saved as .pkl files) 
into animated GIFs for qualitative evaluation. It generates a side-by-side 
comparison between the Ground Truth physics and the Learned Simulator prediction.

References:
    "Learning to Simulate Complex Physics with Graph Networks", Sanchez-Gonzalez et al.
"""

import os
import pickle

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags


# Configurations Flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "rollout_path", None, "Path to the .pkl file containing the rollout data."
)
flags.DEFINE_string(
    "output_path", "output.gif", "Path where the output GIF will be saved."
)
flags.DEFINE_integer(
    "step_stride", 1, "Stride for frame sampling (1=all frames, 2=every other)."
)

# Ensure the user provides at least the input file
flags.mark_flag_as_required("rollout_path")


def render_gif(rollout_path: str, output_path: str, step_stride: int = 1):
    """
    Renders a side-by-side comparison of the ground truth and the predicted simulation.

    Args:
        rollout_path: Path to the .pkl file containing the rollout data.
        output_path: Path where the output .gif file will be saved.
        step_stride: Stride for frame sampling (e.g., 1=every step, 2=every other step).
                     Useful to reduce GIF size for long simulations.
    """

    print(f"Loading rollout data from: {rollout_path}")
    with open(rollout_path, "rb") as f:
        data = pickle.load(f)

    # Extract Trajectories
    # Shape: [Time_Steps, Num_Particles, Dimensions]
    pred_pos = data["predicted_rollout"]
    true_pos = data["ground_truth_rollout"]
    particle_types = data["particle_types"]

    # Setup the Figure
    # We use a dual-subplot setup: Left=Ground Truth, Right=Prediction
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # BOUNDING BOX CALCULATION
    # We compute the global min/max across the ENTIRE trajectory (both true and pred).
    # This ensures the camera remains fixed and doesn't zoom in/out during animation.
    all_pos = np.concatenate([pred_pos, true_pos], axis=0)
    x_min, x_max = all_pos[..., 0].min(), all_pos[..., 0].max()
    y_min, y_max = all_pos[..., 1].min(), all_pos[..., 1].max()

    # Add a 5% visual margin to prevent particles from clipping the edges
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05

    for ax in axes:
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)
        ax.set_aspect("equal")  # Crucial for physics simulations to preserve geometry

    axes[0].set_title("Ground Truth")
    axes[1].set_title("Our GNN Prediction")

    # COLOR MAPPING
    # set blue as default color, could be setted based on particle type
    default_color = "blue"

    # Map each particle to its color based on type
    c = [default_color for _ in particle_types]

    # INITIALIZATION
    # We create the scatter objects once at frame 0.
    # During the update loop, we will only modify their data (set_offsets).
    scat_true = axes[0].scatter(true_pos[0, :, 0], true_pos[0, :, 1], c=c, s=10)
    scat_pred = axes[1].scatter(pred_pos[0, :, 0], pred_pos[0, :, 1], c=c, s=10)

    def update(frame):
        """
        Animation loop callback.
        Updates the position of the scatter plot particles for the current frame.
        """
        # Apply stride to skip frames if requested
        idx = frame * step_stride

        # Safety check to avoid index out of bounds
        if idx >= len(pred_pos):
            return scat_true, scat_pred

        # Update
        # Instead of clearing the axis and replotting (slow), we just update
        # the internal coordinate data of the matplotlib object.
        scat_true.set_offsets(true_pos[idx])
        scat_pred.set_offsets(pred_pos[idx])

        fig.suptitle(f"Time Step: {idx}/{len(pred_pos)}")
        return scat_true, scat_pred

    # Create Animation
    num_frames = len(pred_pos) // step_stride

    # Interval is in milliseconds. 42ms ~= 24 FPS
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=42, blit=True
    )

    # Save Output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # We use 'pillow' as it doesn't require installing external tools like ffmpeg
    ani.save(output_path, writer="pillow", fps=24)
    print("Done! Visualization saved.")
    plt.close()


def main(_):
    """
    Main entry point governed by absl.app.
    """
    render_gif(
        rollout_path=FLAGS.rollout_path,
        output_path=FLAGS.output_path,
        step_stride=FLAGS.step_stride,
    )


if __name__ == "__main__":
    app.run(main)
