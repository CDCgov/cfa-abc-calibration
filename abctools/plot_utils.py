import os

import matplotlib.pyplot as plt
import polars as pl

from abctools.abc_classes import SimulationBundle


def marginal_posterior_plot(
    sim_bundle: SimulationBundle,
    fig_path: str,
    log_scale: bool = True,
    file_name: str = "marginal_pairs.jpg",
):
    n_parameters = len(sim_bundle.experiment_params)
    paired_fig, paired_axes = plt.subplots(
        nrows=n_parameters,
        ncols=n_parameters,
        figsize=(5 * n_parameters, 5 * n_parameters),
        sharex="col",
    )

    bundle_data = sim_bundle.accept_results
    alpha = min(1, (1.0 / len(sim_bundle.accept_results)) + 0.01)
    file_out = os.path.join(fig_path, file_name)

    for row_idx, experiment_param in enumerate(sim_bundle.experiment_params):
        diag_ax = paired_axes[row_idx, row_idx]
        diag_ax.hist(bundle_data[experiment_param])
        for col_idx, comp_param in enumerate(sim_bundle.experiment_params):
            if col_idx < row_idx:
                ax_lwr = paired_axes[row_idx, col_idx]
                ax_lwr.scatter(
                    bundle_data[comp_param],
                    bundle_data[experiment_param],
                    alpha=alpha,
                )
                ax_lwr.set_xlabel(comp_param)
                ax_lwr.set_ylabel(experiment_param)

                if log_scale:
                    ax_lwr.set_xscale("log")
                    ax_lwr.set_yscale("log")

                ax_upr = paired_axes[col_idx, row_idx]
                ax_upr.scatter(
                    bundle_data[experiment_param],
                    bundle_data[comp_param],
                    alpha=alpha,
                )
                ax_upr.set_xlabel(experiment_param)
                ax_upr.set_ylabel(comp_param)

                if log_scale:
                    ax_upr.set_xscale("log")
                    ax_upr.set_yscale("log")

        diag_ax.set_xlabel(experiment_param)
        diag_ax.set_ylabel("Frequency")

    for row in range(n_parameters):
        for col in range(n_parameters):
            if row != col:
                if row != 0:
                    paired_axes[row, col].sharey(paired_axes[row, 0])
                else:
                    paired_axes[row, col].sharey(paired_axes[row, 1])

    paired_fig.savefig(file_out)
    plt.close(paired_fig)


def target_comparison_plot(
    sim_bundle: SimulationBundle,
    target_data: pl.DataFrame,
    fig_path: str,
    x_col,
    y_col,
    x_lab: str = None,
    y_lab: str = None,
    file_name: str = None,
    alpha_by_weight=True,
):
    if file_name is None:
        file_name = (f"timeseries_{sim_bundle.step_number}.jpg",)

    input_weights = sim_bundle.accept_results.filter(
        pl.col("simulation").is_in(sim_bundle.results.keys())
    )

    data_list = []
    for sim_id, df in sim_bundle.results.items():
        result = df.filter(
            pl.col(x_col) <= target_data[x_col].max()
        ).with_columns(pl.lit(sim_id).alias("simulation"))
        result = result.join(input_weights, on="simulation", how="left")
        data_list.append(result)

    # Label axes
    if x_lab is None:
        x_lab = x_col.replace("_", " ").capitalize()
    if y_lab is None:
        y_lab = y_col.replace("_", " ").capitalize()

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)

    for df in data_list:
        if alpha_by_weight:
            alpha = min(1, (df["weight"][0] * 3))
            plt.plot(df[x_col], df[y_col], color="blue", alpha=alpha)
        else:
            alpha = min(1, (1.0 / df.height) + 0.01)
            plt.plot(
                df[x_col],
                df[y_col],
                color="blue",
                alpha=alpha,
            )

    plt.scatter(
        target_data[x_col],
        target_data[y_col],
        color="red",
        label="Target",
        zorder=len(data_list) + 1,
    )

    # Assign figure to object and close plot
    fig_simulations = plt.gcf()
    plt.close()

    file_out = os.path.join(
        fig_path,
        file_name,
    )
    fig_simulations.savefig(file_out)


def plot_xy_data(
    data_list, x_col, y_col, plot_args_list=None, xlabel=None, ylabel=None
):
    """
    Plot multiple datasets on the same axes for comparison.

    Parameters:
    - data_list: List of DataFrames containing the data to be plotted.
    - x_col: Column name for the x-axis values.
    - y_col: Column name for the y-axis values.
    - plot_args_list: List of dictionaries with keyword arguments for styling each plot (optional).
                      Each dictionary in the list corresponds to a DataFrame in `data_list`.
                      If None, a default style will be applied to all plots.
    - xlabel: Label for the x-axis (optional).
    - ylabel: Label for the y-axis (optional).

    Returns:
        None
    """

    # Define a default plotting style if no custom styles are provided
    default_plot_args = {"color": "blue", "alpha": 0.7}

    # If no custom plotting arguments are provided, use the default style for all plots
    if plot_args_list is None:
        plot_args_list = [default_plot_args] * len(data_list)

    # Plot each dataset with its corresponding style
    for df, plot_args in zip(data_list, plot_args_list):
        plt.plot(
            df[x_col],
            df[y_col],
            **plot_args,
        )

    # Set axis labels if provided
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Assign figure to object and close plot
    fig = plt.gcf()
    plt.close()

    return fig
