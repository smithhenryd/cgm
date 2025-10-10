import matplotlib.pyplot as plt

import torch

from typing import Any, Callable, List, Optional, Tuple

from neural_sde import NeuralSDE


def plot_1d_density(
    density: Callable[[torch.Tensor], float],
    ax: Optional[plt.Axes] = None,
    xlim=(-1, 1),
    Npts=100,
    figsize=(3, 3),
    include_labels: bool = True,
    labelsize: float = 12.0,
    axis_labelsize: float = 14.0,
    **plt_kwargs: Any,
):
    """
    Function for plotting a 1D density
    """

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Grid of x values
    xs = torch.linspace(xlim[0], xlim[1], Npts)

    # Plot
    ax.plot(xs, density(xs), **plt_kwargs)

    # Label axes
    if include_labels:
        # Axis labels with fontsize
        ax.set_xlabel("x", fontsize=axis_labelsize)
        ax.set_ylabel("density", fontsize=axis_labelsize)

        # Tick labels fontsize
        ax.tick_params(axis="both", which="major", labelsize=labelsize)

    return fig, ax


def plot_nsde_marginals(
    model: NeuralSDE,
    ts: list[float] = [0.0, 0.75, 0.9, 0.95, 0.99, 1.0],
    Nsamps: int = 1000,
    xlim: Tuple[float, float] = (-1, 1),
    figsize: Tuple[float, float] = (3, 3),
    include_labels: bool = True,
    labelsize: float = 12.0,
    axis_labelsize: float = 14.0,
    bins: int = 100,
    component: int = 0,
    density: bool = True,
    hist_kwargs: dict = None,
):
    """
    Function for plotting the marginals of a neural-SDE
    """

    if hist_kwargs is None:
        hist_kwargs = {}

    # Sample from the model
    with torch.no_grad():
        out = model.sample(Nsamps)
        xs = out.xs
        ts_grid = out.ts
        xs, ts_grid = xs.detach().cpu(), ts_grid.cpu()

    d = xs.shape[-1]
    if component < 0 or component >= d:
        raise ValueError(f"'component' must be in [0, {d-1}], got {component}")

    # Create grid for plotting
    num_requested = len(ts)
    cols = 3
    rows_needed = (num_requested + (cols - 1)) // cols
    rows = min(4, rows_needed)  # cap at 4 rows
    max_panels = rows * cols
    if num_requested > max_panels:
        print(
            f"Note: truncating to {max_panels} panels (got {num_requested} time points)."
        )
        ts = ts[:max_panels]

    w, h = figsize
    fig_w = w * cols
    fig_h = h * rows

    fig, axs = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1:
        axs = axs.reshape(1, -1)

    bin_edges = torch.linspace(xlim[0], xlim[1], bins + 1)

    # Plot each time
    for i, t in enumerate(ts):
        row, col = divmod(i, cols)
        ax_i = axs[row, col]

        # Find closest time
        t_tensor = torch.tensor(t, dtype=ts_grid.dtype)
        t_idx = torch.argmin(torch.abs(ts_grid - t_tensor)).item()

        # Get samples from selected dimension
        samps_i = xs[:, t_idx, component]

        # Plot them
        ax_i.hist(
            samps_i, bins=bin_edges, histtype="step", density=density, **hist_kwargs
        )

        # Add labels
        if include_labels:
            ax_i.set_title(f"t = {ts_grid[t_idx].item():.3f}", fontsize=axis_labelsize)
            ax_i.set_xlabel("x", fontsize=axis_labelsize)
            ax_i.set_ylabel("density" if density else "count", fontsize=axis_labelsize)

        # Set limits and ticks
        ax_i.set_xlim(*xlim)
        ax_i.tick_params(axis="both", which="major", labelsize=labelsize)

    # Hide any unused panels
    total_panels = rows * cols
    for j in range(len(ts), total_panels):
        r, c = divmod(j, cols)
        axs[r, c].axis("off")

    fig.tight_layout()
    return fig, axs
