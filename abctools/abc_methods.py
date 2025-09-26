import logging
import random
import warnings

import numpy as np
import polars as pl
from scipy.stats import qmc, truncnorm

from abctools.abc_classes import SimulationBundle

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def draw_simulation_parameters(
    params_inputs: dict,
    n_parameter_sets: int,
    method: str = "sobol",
    add_random_seed: bool = True,
    add_simulation_id: bool = True,
    starting_simulation_number: int = 0,
    seed=random.randint(0, 2**32 - 1),
    seed_variable_name: str = "randomSeed",
    replicates_per_particle=1,
) -> pl.DataFrame:
    """
    Draw samples of parameters for simulations based on the specified method.

    Args:
        params_inputs (dict): Dictionary containing parameters and their distributions.
        n_parameter_sets (int): Number of simulations to perform.
        method (str): Sampling method ("random", "sobol", or "latin_hypercube").
        add_random_seed (bool): If True, adds a random seed column with randomly generated numbers.
        add_simulation_id (bool): If True, adds a 'simulation' column with simulation IDs starting from `starting_simulation_number`.
        starting_simulation_number (int): The number at which to start numbering simulations. Defaults to 0.
        seed (int): Random seed passed in to ensure consistency between runs.
        replicates_per_particle (int): Number of replicates to generate per parameter set.

    Returns:
        pd.DataFrame: DataFrame containing arrays of sampled values for each parameter,
                      possibly including 'random_seed' and 'simulation' columns.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_params = len(params_inputs)

    if method == "random":
        # Random sampling for each distribution
        samples = {
            param_key: dist_obj.rvs(size=n_parameter_sets)
            for param_key, dist_obj in params_inputs.items()
        }

    elif method == "sobol" or method == "latin_hypercube":
        warnings.filterwarnings("ignore", category=UserWarning)

        # Create the appropriate sampler based on the chosen method
        sampler = (
            qmc.Sobol(d=num_params, seed=seed)
            if method == "sobol"
            else qmc.LatinHypercube(d=num_params, seed=seed)
        )

        # Generate uniform samples across all dimensions
        uniform_samples = sampler.random(n=n_parameter_sets)

        # Transform uniform samples using ppf for each distribution
        samples_transformed = np.column_stack(
            [
                dist.ppf(uniform_samples[:, i])
                for i, dist in enumerate(params_inputs.values())
            ]
        )

        # Convert array of samples into dictionary format
        samples = {
            param_key: samples_transformed[:, i]
            for i, param_key in enumerate(params_inputs.keys())
        }

    else:
        raise ValueError(f"Unsupported sampling method: {method}")

    # Convert to Polars DataFrame
    n_simulations = n_parameter_sets * replicates_per_particle
    simulation_parameters_df = pl.DataFrame(samples).select(
        pl.all().repeat_by(replicates_per_particle).flatten()
    )

    if add_random_seed:
        # Add a random seed column with integers between 0 and 2^32 - 1
        seed_column = [
            random.randint(0, 2**32) for _ in range(n_simulations)
        ]
        simulation_parameters_df = simulation_parameters_df.with_columns(
            pl.Series(seed_variable_name, seed_column)
        )

    # If specified, add a simulation ID column with integers from `starting_simulation_number` to `starting_simulation_number + n_parameter_sets - 1`
    if add_simulation_id:
        # Generate sequence using arange and offset by the starting number
        simulation_id_sequence = np.arange(
            starting_simulation_number,
            starting_simulation_number + n_simulations,
        )

        # Reorder columns to make 'simulation' the first one
        simulation_parameters_df = pl.concat(
            [
                pl.DataFrame({"simulation": simulation_id_sequence}),
                simulation_parameters_df,
            ],
            how="horizontal",
        )

    return simulation_parameters_df


def resample(
    accepted_simulations: pl.DataFrame,
    n_samples: int,
    prior_distributions: dict,
    replicates_per_sample: int = 1,
    perturbation_kernels: dict = None,
    weights: pl.DataFrame = None,
    add_random_seed: bool = True,
    starting_simulation_number: int = 0,
    seed: int = None,
    seed_variable_name: str = "randomSeed",
) -> pl.DataFrame:
    """
    Resamples parameters from accepted simulations with optional perturbation and reweighting.

    Args:
        accepted_simulations (pl.DataFrame): DataFrame of accepted simulations with parameters.
        n_samples (int): Number of additional samples to generate.
        replicates_per_sample (int): Number of replicates to generate per sample.
        perturbation_kernels (dict): Dictionary of perturbation kernels for each parameter.
        prior_distributions (dict): Dictionary of prior distributions for each parameter.
        weights (pl.DataFrame or None): Optional DataFrame of weights for each accepted simulation. If None, uniform weighting is assumed.
        add_random_seed (bool): If True, adds a random seed column with randomly generated numbers.
        starting_simulation_number (int): Starting number for new simulation IDs.
        seed (int): Random seed passed in to ensure consistency between runs.

    Returns:
        pl.DataFrame: DataFrame containing resampled and possibly perturbed parameters.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if weights is not None and not weights.is_empty():
        weights = weights.with_columns(
            (pl.col("weight") / pl.col("weight").sum()).alias(
                "normalized_weight"
            )
        )
        accepted_simulations = accepted_simulations.join(
            weights.select(["simulation", "normalized_weight"]),
            on="simulation",
            how="left",
        )
        sampling_weights = accepted_simulations["normalized_weight"].to_list()
    else:
        sampling_weights = [1.0 / len(accepted_simulations)] * len(
            accepted_simulations
        )

    # Sample 'simulation' values with replacement using sampling weights
    sampled_simulations = random.choices(
        accepted_simulations["simulation"].to_list(),
        weights=sampling_weights,
        k=n_samples,
    )
    # Filter rows based on sampled 'simulation' values, ensuring duplicates are included
    sampled_simulations_df = pl.concat(
        [
            accepted_simulations.filter(pl.col("simulation") == sim)
            for sim in sampled_simulations
        ]
    )

    # Apply perturbations
    if perturbation_kernels and prior_distributions:
        sampled_simulations_df = apply_perturbations(
            sampled_simulations_df, perturbation_kernels, prior_distributions
        )

    # Repeat rows by replicates_per_sample
    if replicates_per_sample > 1:
        resampled_df = sampled_simulations_df.select(
            pl.all().repeat_by(replicates_per_sample).flatten()
        )
    else:
        resampled_df = sampled_simulations_df

    # Add simulation column starting at specified number
    resampled_df = resampled_df.with_columns(
        pl.Series(
            "simulation",
            np.arange(
                starting_simulation_number,
                starting_simulation_number + len(resampled_df),
            ),
        )
    )

    if add_random_seed:
        # Add a random seed column with integers between 0 and 2^32 - 1
        resampled_df = resampled_df.with_columns(
            pl.Series(
                seed_variable_name,
                np.random.randint(0, 2**32 - 1, size=len(resampled_df)),
            )
        )
    # Keep all columns
    colnames = [k for k in prior_distributions.keys()]
    colnames.append("simulation")
    colnames.append(seed_variable_name)

    resampled_df = resampled_df.select(colnames)

    return resampled_df


def apply_perturbations(
    df: pl.DataFrame,
    perturbation_kernels: dict,
    prior_distributions: dict,
    seed: int = None,
) -> pl.DataFrame:
    """
    Applies perturbations to parameters using kernels and ensures validity within prior distributions.

    Args:
        df (pl.DataFrame): DataFrame of parameters to perturb.
        perturbation_kernels (dict): Dictionary of perturbation kernels for each parameter.
        prior_distributions (dict): Dictionary of prior distributions for each parameter.
        seed (int): Random seed for reproducibility.
    Returns:
        pl.DataFrame: DataFrame with perturbed parameters.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)  # Set the global NumPy random seed

    def perturb_column(values, kernel, prior, base_seed):
        """Perturb an entire column of values."""
        perturbed_values = []
        for i, value in enumerate(values):
            rng = np.random.default_rng(
                base_seed + i
            )  # Create a unique RNG for each element
            while True:
                perturbed_value = value + kernel.rvs(random_state=rng)
                try:
                    prior_value = prior.pdf(perturbed_value)
                except AttributeError:
                    prior_value = prior.pmf(perturbed_value)
                if (
                    
                    prior_value > 0
                ):  # Ensure the value is valid within the prior
                    perturbed_values.append(perturbed_value)
                    break
        return perturbed_values

    # Iterate over each parameter column and apply perturbations
    for param_name in df.columns:
        if (
            param_name in perturbation_kernels
            and param_name in prior_distributions
        ):
            kernel = perturbation_kernels[param_name]
            prior = prior_distributions[param_name]

            # Use a deterministic seed for the column if `seed` is provided
            column_seed = (
                seed if seed is not None else np.random.randint(0, 2**32 - 1)
            )

            # Apply perturbation to the column
            perturbed_values = perturb_column(
                df[param_name].to_numpy(), kernel, prior, column_seed
            )
            df = df.with_columns(pl.Series(param_name, perturbed_values))

    return df


def calculate_weights_abcsmc(
    current: SimulationBundle,
    previous: SimulationBundle,
    prior_distributions: dict,
    perturbation_kernels: dict,
    normalize: bool = True,
) -> pl.DataFrame:
    """
    Calculate weights for simulations in steps t > 0 of an ABC SMC algorithm (Toni et al. 2009)

    Args:
        current_accepted (pl.DataFrame): Accepted simulations for the current step
        prev_step_accepted (pl.DataFrame): Accepted simulations from the previous step.
        prev_weights (pl.DataFrame): Weights for each simulation from the previous step.
        prior_distributions (dict): Dictionary containing prior distribution objects for each parameter.
        perturbation_kernels (dict): Dictionary containing scipy.stats distributions used as perturbation kernels for each parameter.
        normalize (bool): If True, normalize the weights so they sum to 1.

    Returns:
        pl.DataFrame: DataFrame of calculated weights for each simulation in current_accepted.
    """

    # Ensure prev_weights weight column is numeric
    prev_weights = previous.weights.with_columns(
        pl.col("weight").cast(pl.Float64)
    )
    current_accepted = current.get_accepted()
    prev_accepted = previous.get_accepted()
    assert set(perturbation_kernels.keys()) == set(prior_distributions.keys())

    # Initialize list to store new weights
    new_weights = []
    # Loop over all accepted simulations in current step
    for current_particle in current_accepted.iter_rows(named=True):
        prior_prob = 1.0
        simulation_id = current_particle["simulation"]
        for key in prior_distributions.keys():
            try:
                prior_prob *= prior_distributions[key].pdf(current_particle[key])
            except AttributeError:
                prior_prob *= prior_distributions[key].pmf(current_particle[key])
            
        # Multiply prior probability by proportion of simulations accepted
        numerator = prior_prob * current_particle["acceptance_weight"]

        denominator = 0.0
        for past_particle in prev_accepted.iter_rows(named=True):
            perturbation_prob = past_particle["acceptance_weight"]
            for key in perturbation_kernels.keys():
                kernel = perturbation_kernels[key]
                dist = current_particle[key] - past_particle[key]
                try:
                    perturbation_prob *= kernel.pdf(dist)
                except AttributeError:
                    perturbation_prob *= kernel.pmf(dist)
            denominator += (
                perturbation_prob
                * prev_weights.filter(
                    pl.col("simulation") == past_particle["simulation"]
                )
                .select("weight")
                .item()
            )

        weight = numerator / denominator if denominator != 0 else 0
        new_weights.append(
            {
                "simulation": simulation_id,
                "weight": weight,
            }
        )

    weights_df = pl.DataFrame(new_weights)

    if normalize:
        # Ensure weights column is numeric before normalization
        weights_df = weights_df.with_columns(pl.col("weight").cast(pl.Float64))

        # Normalize weights so they sum up to 1
        total_new_weight = weights_df["weight"].sum()

        if total_new_weight == 0:
            raise ValueError(
                "Total weight is zero after normalization. Check input data and distributions."
            )

        weights_df = weights_df.with_columns(
            (pl.col("weight") / total_new_weight).alias("weight")
        )

    return weights_df


def get_truncated_normal(mean, sd, low=0, upp=1):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
