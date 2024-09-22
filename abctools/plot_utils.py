import matplotlib.pyplot as plt


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
