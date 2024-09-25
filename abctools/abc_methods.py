import random
import warnings

import numpy as np
import polars as pl
from scipy.stats import qmc, truncnorm


def draw_simulation_parameters(
    params_inputs: dict,
    n_simulations: int,
    method: str = "sobol",
    add_random_seed: bool = True,
    add_simulation_id: bool = True,
    starting_simulation_number: int = 0,
    seed = None,
) -> pl.DataFrame:
    """
    Draw samples of parameters for simulations based on the specified method.

    Args:
        params_inputs (dict): Dictionary containing parameters and their distributions.
        n_simulations (int): Number of simulations to perform.
        method (str): Sampling method ("random", "sobol", or "latin_hypercube").
        add_random_seed (bool): If True, adds a 'randomSeed' column with randomly generated numbers.
        add_simulation_id (bool): If True, adds a 'simulation' column with simulation IDs starting from `starting_simulation_number`.
        starting_simulation_number (int): The number at which to start numbering simulations. Defaults to 0.
        seed (int): Random seed passed in to ensure consistency between runs.

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
            param_key: dist_obj.rvs(size=n_simulations)
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
        uniform_samples = sampler.random(n=n_simulations)

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
    simulation_parameters_df = pl.DataFrame(samples)

    if add_random_seed:
        # Add a random seed column with integers between 0 and 2^32 - 1
        seed_column = [
            random.randint(0, 2**32) for _ in range(n_simulations)
        ]
        simulation_parameters_df = simulation_parameters_df.with_columns(
            pl.Series("randomSeed", seed_column)
        )

    # If specified, add a simulation ID column with integers from `starting_simulation_number` to `starting_simulation_number + n_simulations - 1`
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
    accepted_simulations,
    n_samples,
    perturbation_kernels=None,
    prior_distributions=None,
    weights=None,
    add_random_seed: bool = True,
    starting_simulation_number=0,
    seed = None
):
    """
    Resamples parameters from accepted simulations with optional perturbation and reweighting.

    Args:
        accepted_simulations (dict): Dictionary of Polar DataFrames or dictionaries of accepted simulations with parameters.
        n_samples (int): Number of additional samples to generate.
        perturbation_kernels (dict): Dictionary of perturbation kernels for each parameter.
        prior_distributions (dict): Dictionary of prior distributions for each parameter.
        weights (dict or None): Optional dictionary of weights for each accepted simulation. If None, uniform weighting is assumed.
        add_random_seed (bool): If True, adds a 'random_seed' column with randomly generated numbers.
        starting_simulation_number (int): Starting number for new simulation IDs.
        seed (int): Random seed passed in to ensure consistency between runs.

    Returns:
        pl.DataFrame: DataFrame containing resampled and possibly perturbed parameters.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # Prepare list to hold new samples
    new_samples = []

    # Normalize weights if provided (just to be safe)
    if weights is not None:
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        sim_numbers = list(normalized_weights.keys())
        weights = list(normalized_weights.values())
    else:
        sim_numbers = list(accepted_simulations.keys())
        weights = [1 / len(sim_numbers)] * len(sim_numbers)

    # Resampling loop
    for _ in range(n_samples):
        # Select a random index based on weights
        chosen_index = random.choices(sim_numbers, weights=weights, k=1)[0]

        # Retrieve the chosen parameters and apply perturbations
        chosen_params = accepted_simulations[chosen_index]

        # Perturb each parameter using its respective kernel and check against prior distribution
        selected_params = {}

        # Convert to dictionary if Polars DataFrame
        if isinstance(chosen_params, pl.DataFrame):
            chosen_params = chosen_params.to_dicts()[0]

        # Perturb and test against prior distribution, if specified
        for param_name, value in chosen_params.items():
            if perturbation_kernels and prior_distributions:
                kernel_dist = perturbation_kernels.get(param_name)
                prior_dist = prior_distributions.get(param_name)

                if kernel_dist is not None:
                    # Apply perturbation until a valid sample within the prior distribution is obtained
                    while True:
                        perturbed_value = value + kernel_dist.rvs()
                        if (
                            prior_dist.pdf(perturbed_value) > 0
                        ):  # Check if within prior distribution
                            break

                    selected_params[param_name] = perturbed_value

                else:
                    raise ValueError(
                        f"No perturbation kernel provided for {param_name}."
                    )
            else:
                selected_value = value
                selected_params[param_name] = selected_value

        new_samples.append(selected_params)

    # Convert list of dictionaries to Polars DataFrame and add 'simulation' column starting at desired number
    resampled_df = pl.DataFrame(new_samples).with_columns(
        pl.Series(
            "simulation",
            np.arange(
                starting_simulation_number,
                starting_simulation_number + n_samples,
            ),
        )
    )

    if add_random_seed:
        # Add a random seed column with integers between 0 and 2^32 - 1
        seed_column = [random.randint(0, 2**32) for _ in range(n_samples)]
        resampled_df = resampled_df.with_columns(
            pl.Series("randomSeed", seed_column)
        )

    return resampled_df


def calculate_weights_abcsmc(
    current_accepted,
    prev_step_accepted,
    prev_weights,
    prior_distributions,
    perturbation_kernels,
    normalize=True,
):
    """
    Calculate weights for simulations in steps t > 0 of an ABC SMC algorithm  (Toni et al. 2009)

    Args:
        current_accepted (dict): Accepted simulations for the current step. Dictionary of Polar DataFrames or dictionaries with parameters.
        prev_step_accepted (dict): Dictionary of accepted simulations from the previous step. Dictionary of Polar DataFrames or dictionaries with parameters.
        prev_weights (dict): Dictionary of weights for each simulation from the previous step.
        prior_distributions (dict): Dictionary containing prior distribution objects for each parameter.
        perturbation_kernels (dict): Dictionary containing scipy.stats distributions used as perturbation kernels for each parameter.
        normalize (bool): If True, normalize the weights so they sum to 1.

    Returns:
        dict: Dictionary of calculated weights for each simulation in current_accepted.
    """

    # Initialize dictionary to store new weights
    new_weights = {}

    # Loop over all accepted simulations in current step
    for sim_number, params in current_accepted.items():
        # Convert Polars DataFrame to dictionary if necessary
        if isinstance(params, pl.DataFrame):
            params = params.to_dicts()[0]

        # Calculate numerator
        numerator = 1.0
        for param_name, param_value in params.items():
            prior_dist = prior_distributions[param_name]
            numerator *= prior_dist.pdf(param_value)

        # Calculate denominator: weighted sum over all previous particles' contribution
        denominator = 0.0
        for prev_sim_number, prev_params in prev_step_accepted.items():
            if isinstance(prev_params, pl.DataFrame):
                prev_params = prev_params.to_dicts()[0]

            kernel_product = 1.0
            for param_name in params.keys():
                if param_name in perturbation_kernels:
                    kernel_dist = perturbation_kernels[param_name]
                    kernel_product *= kernel_dist.pdf(
                        params[param_name] - prev_params[param_name]
                    )

            denominator += prev_weights[prev_sim_number] * kernel_product

        # Avoid division by zero; if denominator is zero set weight to zero directly
        weight = numerator / denominator if denominator != 0 else 0

        # Store calculated weight
        new_weights[sim_number] = weight

    if normalize:
        # Normalize weights so they sum up to 1
        total_new_weight = sum(new_weights.values())

        if total_new_weight == 0:
            raise ValueError(
                "Total weight is zero after normalization. Check input data and distributions."
            )

        new_weights = {
            sim_number: w / total_new_weight
            for sim_number, w in new_weights.items()
        }

    return new_weights

def get_truncated_normal(mean, sd, low=0, upp=1):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
