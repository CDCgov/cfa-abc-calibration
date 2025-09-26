import logging
import os
import pickle
from typing import Callable

import polars as pl

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SimulationBundle:
    """
    A class to keep track of an iteration/generation of simulations (particles)
    for ABC/SMC.

    Attributes:
        inputs (pl.DataFrame): Input parameters for the simulations.
        results (pl.DataFrame): Results for the simulations, initialized as an empty DataFrame.
        step_number (int): Keeps track of the ABC step (a.k.a. generation/iteration)
        baseline_params (dict): Unchanging parameters needed for the simulation
        experiment_params (list): Derived from 'inputs'--list of experiment parameter names
        status (str): Current status in the ABC process
        distances (pl.DataFrame): Calculated distances from target
        accepted (pl.DataFrame): Accepted simulations with experiment parameters
        n_accepted (int): Calculated from 'accepted'--number of accepted simulations
        weights (pl.DataFrame): Simulation weights for resampling
        merge_history (dict): History of merges with other SimulationBundle objects
        summary_metrics (pl.DataFrame): Summary metrics calculated for each simulation
        accept_results (pl.DataFrame): DataFrame of accepted results
    """

    def __init__(
        self,
        inputs: pl.DataFrame,
        step_number: int,
        baseline_params: dict,
        status: str = "initialized",
        seed_variable_name: str = "randomSeed",
    ):
        """
        Initialize a new instance of SimulationsBundle.

        Args:
            inputs (pl.DataFrame): Input parameters for the simulations (optionally including randomSeed).
            step_number (int): Step/iteration/generation number.
            baseline_params (dict): The baseline parameters for the simulations.
            status (str): Current status of the object. Defaults to "initialized".
            seed_variable_name (str): Name of the column containing random seeds. Defaults to "randomSeed".
        """

        # Public variables
        self.inputs = inputs
        self.status = status
        self.merge_history = {}
        self.results = pl.DataFrame()
        self.distances = pl.DataFrame()
        self.accepted = pl.DataFrame()
        self.weights = pl.DataFrame()
        self.summary_metrics = None
        self.seed_variable_name = seed_variable_name

        # Private variables
        self._step_number = step_number
        self._baseline_params = baseline_params
        self._experiment_params = [
            col
            for col in inputs.columns
            if col not in ["simulation", self.seed_variable_name]
        ]

    @property
    def step_number(self) -> int:
        """Getter for _step_number."""
        return self._step_number

    @property
    def n_simulations(self) -> int:
        """Getter for _n_simulations."""
        return self.inputs["simulation"].n_unique()

    @property
    def baseline_params(self) -> list:
        """Getter for _baseline_params."""
        return self._baseline_params

    @property
    def experiment_params(self) -> list:
        """Getter for _experiment_params."""
        return self._experiment_params

    @property
    def n_accepted(self) -> int:
        """Getter for number of accepted simulations"""
        return len(self.get_accepted())

    @property
    def writer_input_dict(self) -> dict:
        """Getter that outputs a dictionary with simulation details. Needed by gcm_python_wrappers.wrappers.gcm_experiments_writer"""
        return {
            "baseline_parameters": self._baseline_params,
            "experiment_parameters": self._experiment_params,
            "simulation_parameter_values": self.inputs,
        }

    @property
    def full_params_df(self) -> pl.DataFrame:
        """Getter that outputs a Polars DataFrame with the full parameters list (simulation number, random seed, baseline parameters, and experimental parameters)"""
        full_params_df = self.inputs
        for colname, value in self._baseline_params.items():
            full_params_df = full_params_df.with_columns(
                pl.lit(value).alias(colname)
            )
        return full_params_df

    def __getstate__(self):
        """
        Specifies what gets pickled when the save_state method is called.

        Returns:
            state (dict): The object's state without the 'results' attribute.
        """
        # Copy object's __dict__
        state = self.__dict__.copy()

        # Remove 'results'
        if "results" in state:
            del state["results"]

        return state

    def save_state(self, folder_path: str, filename: str):
        """
        Saves the current state of the simulation bundle to a file using pickle,
        excluding 'results'.

        Args:
            folder_path (str): The path to the folder where state should be saved.
            filename (str): The name of the file to save state into.

        Returns:
            None
        """
        # Check if folder exists, and create it if it doesn't
        os.makedirs(folder_path, exist_ok=True)

        # Create full path for the output file
        full_path = os.path.join(folder_path, filename)

        # Use 'with' statement to ensure that file is properly closed after writing
        with open(full_path, "wb") as file:
            # Pickle only selected parts of the object and write it to file
            pickle.dump(self.__getstate__(), file)

    def add_results(self, results_df, merge_params=True):
        """
        Adds results to the SimulationBundle object.

        Args:
            results_df (pl.DataFrame): The results DataFrame to be added.

        Returns:
            None
        """
        # Ensure results_df is a Polars DataFrame with all of the same values in the 'simulation' column as inputs
        # Note: there may be more than one row per simulation in results_df
        if not isinstance(results_df, pl.DataFrame):
            raise TypeError("results_df must be a Polars DataFrame.")

        if not results_df["simulation"].is_in(self.inputs["simulation"]).all():
            if merge_params:
                raise ValueError(
                    "results_df must contain all simulation numbers from inputs if merging inputs."
                )
            else:
                raise Warning(
                    "Warning: results_df does not contain all the simulation numbers from inputs."
                )

        # Recover params if applicable
        if merge_params:
            self.merge_params()

        self.results = results_df

    def merge_params(self):
        """
        Updates self.results by merging in columns from self.inputs onto self.results based on the 'simulation' column.

        Returns:
            None
        """
        if self.results is None:
            raise ValueError(
                "self.results is not set. Cannot merge parameters without results."
            )

        # Perform a left join to add input parameters to results based on 'simulation'
        merged_results = self.results.join(
            self.inputs, on=["simulation"], how="left"
        )

        # Ensure the DataFrame is unique on 'simulation'
        merged_results = merged_results.unique(subset=["simulation"])

        # Update self.results with merged data
        self.results = merged_results

    def calculate_summary_metrics(self, summary_function):
        """
        Applies a user-defined function to calculate summary metrics for each simulation.

        Args:
            summary_function (callable): A function that takes in per-simulation results (a Polars DataFrame, typically) and returns summary metrics.

        Returns:
            None
        """
        if self.results is None:
            raise ValueError("No results available to summarize.")

        self.summary_metrics = pl.DataFrame()

        grouped_results = apply_per_group_preserve_key(
            self.results,
            key="simulation",
            user_udf=summary_function,
        )

        self.summary_metrics = grouped_results

    def calculate_distances(
        self, target_data, distance_function, modifier, use_summary_metrics=False
    ):
        """
        Calculates distances between simulation results and target data using a user-defined distance function.

        Args:
            target_data (tuple): Target data to compare against.
            distance_function (callable): A user-defined function that takes results_data and target_data and returns a distance.
            use_summary_metrics (bool): Whether to use summary metrics or raw results. Defaults to False.

        Returns:
            None
        """

        # Check if summary metrics should be used
        if use_summary_metrics:
            data_to_use = self.summary_metrics
        else:
            data_to_use = self.results

        data_to_use = data_to_use.join(modifier, on="simulation", how="left")
        # Round the 'burn_in_column' to the nearest value divisible by 7
        data_to_use = data_to_use.with_columns(
            (pl.col("burn_in_period") // 7 * 7).alias("burn_in_period")
        )
        # Add the burn_in_period value to the t value
        data_to_use = data_to_use.with_columns(
            (pl.col("t").cast(int) + pl.col("burn_in_period").cast(pl.Int64)).alias("t")
        )
        self.distances = apply_per_group_preserve_key(
            data_to_use,
            key="simulation",
            user_udf=distance_function,
            result_column="distance",
            target_data=target_data,
        )

    def get_accepted(self, one_rep_per_parset: bool = False):
        if self.accepted.is_empty():
            raise ValueError(
                "Accepted simulations have not been determined yet."
            )
        if one_rep_per_parset:
            return (
                self.accepted.filter(pl.col("accept_bool"))
                .group_by(
                    self.inputs.drop(
                        ["simulation", self.seed_variable_name]
                    ).columns
                )
                .agg([pl.col("simulation").min()])
                .drop("simulation")
            )
        else:
            return self.accepted.filter(pl.col("accept_bool"))

    def accept_reject(self, tolerance):
        """
        Accepts or rejects simulations based on the calculated distances and given tolerance level.

        Args:
            tolerance (float): The tolerance level for accepting simulations.

        Returns:
            None
        """

        # Ensure distances have been calculated
        if self.distances.is_empty():
            raise ValueError("Distances have not been calculated.")
        # Filter and join to get accepted parameters
        self.accepted = self.distances.with_columns(
            (pl.col("distance") <= tolerance).alias("accept_bool")
        ).join(self.inputs, on="simulation")

    def accept_stochastic(
        self,
        tolerance,
    ):
        """
        Accepts the minimum simulation of each parameter set with greater than zero replicates under the tolerance level
        Sets the acceptance_weight proportion for each parameter set

        Args:
            tolerance (float): The tolerance level for accepting simulations.

        Returns:
            None

        Raises:
            ValueError: If distances have not been previously calculated.
        """

        if self.distances.is_empty():
            raise ValueError("Distances have not been calculated.")

        self.accept_reject(tolerance)
        # Group by parameters besides simulation and random seed
        param_names = self.inputs.drop(
            ["simulation", self.seed_variable_name]
        ).columns

        self.accepted = (
            self.accepted.group_by(param_names)
            .agg(
                [
                    pl.col("accept_bool").mean().alias("acceptance_weight"),
                ]
            )
            .filter(
                pl.col("acceptance_weight") > 0
            )  # Filter particles with accepted count > 0
            .join(self.accepted, on=param_names, how="right")
            .with_columns(
                pl.col("acceptance_weight").fill_null(strategy="zero")
            )
        )

    def accept_proportion(self, proportion: float):
        """
        Accepts a specified proportion of simulations with the smallest distances.
        This method ranks all simulations by their distance values in ascending order
        and selects the top-performing simulations up to the specified proportion.

        Args:
            proportion (float): The proportion of top simulations to accept based on their distances.
                                For example, 0.1 for the top 10%, or 0.25 for the top 25%.

        Returns:
            None

        Raises:
            ValueError: If distances have not been previously calculated.
        """

        # Ensure distances have been calculated
        if self.distances.is_empty():
            raise ValueError("Distances have not been calculated.")

        # Calculate the number of simulations to accept based on the given proportion (minimum of 1)
        num_to_accept = max(1, int(len(self.distances) * proportion))

        # Sort simulations by distance in ascending order and select the best ones
        sorted_distances = self.distances.sort("distance").head(num_to_accept)

        # Filter the accepted parameters, remove simulation and seed columns
        accepted_params = self.inputs.filter(
            pl.col("simulation").is_in(sorted_distances["simulation"])
        ).drop(["simulation", self.seed_variable_name], None)

        # Create accepted dataframe
        self.accepted = sorted_distances.join(
            accepted_params, on="simulation", how="left"
        ).with_columns(pl.lit(1.0).alias("acceptance_weight"))

    def merge_with(self, other_bundle):
        """
        Merges another SimulationBundle object into this one by combining their inputs,
        results, summary metrics, distances, and accepted simulations.

        Args:
            other_bundle (SimulationBundle): Another SimulationBundle instance to merge with this one.

        Returns:
            None
        """

        # Merge inputs DataFrames directly
        merged_inputs = pl.concat([self.inputs, other_bundle.inputs])

        # Check for duplicate simulation numbers after merging
        if (
            merged_inputs["simulation"].unique().len()
            != merged_inputs["simulation"].len()
        ):
            raise ValueError(
                "Duplicate simulation numbers found after merging. Merge aborted."
            )

        # If no duplicates are found, proceed with updating self.inputs
        self.inputs = merged_inputs

        # Merge results as DataFrames
        self.results = pl.concat([self.results, other_bundle.results])

        # Merge distances DataFrames directly
        self.distances = pl.concat([self.distances, other_bundle.distances])

        # Merge accepted simulations DataFrames directly
        self.accepted = pl.concat([self.accepted, other_bundle.accepted])

        # Merge summary metrics DataFrames directly
        self.summary_metrics = pl.concat(
            [self.summary_metrics, other_bundle.summary_metrics]
        )

        # Record the merge event in the history
        current_merge_index = len(self.merge_history) + 1
        number_merged = len(other_bundle.inputs)

        self.merge_history[current_merge_index] = number_merged


def apply_per_group_preserve_key(
    df: pl.DataFrame,
    key: str,
    user_udf: Callable[[pl.DataFrame], pl.DataFrame],
    *args,
    result_column: str = "result",  # Specify the column name for scalar outputs
    **kwargs
) -> pl.DataFrame:
    """
    Apply a user-defined function to each group of a DataFrame, preserving the key column.
    NOTE: This is surprisingly a bit tricky in polars, and a better solution is welcome.
    Handles both DataFrame and scalar outputs from the user-defined function.

    Args:
        df (pl.DataFrame): The input DataFrame.
        key (str): The column to group by.
        user_udf (Callable): The user-defined function to apply to each group.
        result_column (str): The column name to use if the user_udf returns a scalar.
        *args, **kwargs: Additional arguments to pass to the user_udf.

    Returns:
        pl.DataFrame: A DataFrame with the key column and the results of the user_udf.
    """
    parts = df.partition_by(key)
    mapped_parts = []

    for part in parts:
        result = user_udf(part, *args, **kwargs)
        if not isinstance(result, pl.DataFrame):
            # Wrap scalar output in a DataFrame
            result = pl.DataFrame({result_column: [result]})
        # Add the key column to the result
        result = result.with_columns(pl.lit(part[key][0]).alias(key))
        mapped_parts.append(result)

    # Concatenate all parts and reorder columns to place the key first
    result = pl.concat(mapped_parts, how="vertical")
    columns = [key] + [col for col in result.columns if col != key]
    return result.select(columns)
