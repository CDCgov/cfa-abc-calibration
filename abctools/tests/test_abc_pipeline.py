import math
import os
import unittest

import matplotlib.pyplot as plt
import polars as pl
from scipy.stats import uniform

from abctools import SimulationBundle, abc_methods, plot_utils, toy_model

# Set random seed
random_seed = 12345


def run_toy_model(params: dict):
    """Runs toy stochastic SIR model"""
    # Configure compartments
    N = params["population"]
    I0 = int(params["population"] * params["initialPrevalence"])
    S0 = N - I0
    R0 = 0
    initial_state = (S0, I0, R0)

    # Configure time
    tmax = params["totalDays"]

    # Configure rates
    beta = 1 / params["averageContactsPerDay"]
    gamma = 1 / params["infectiousPeriod"]
    rates = (beta, gamma)

    # Configure random seed
    if "randomSeed" in params:
        random_seed = params["randomSeed"]
    else:
        random_seed = None

    # Run model
    # Note: I am not using 'susceptible' for anything in this example
    time_points, _, infected, recovered = toy_model.ctmc_gillespie_model(
        initial_state, rates, tmax, t=0, random_seed=random_seed
    )

    # Format as dataframe
    results_df = pl.DataFrame(
        {"time": time_points, "infected": infected, "recovered": recovered},
        schema=[
            ("time", pl.Float32),
            ("infected", pl.Int64),
            ("recovered", pl.Int64),
        ],
    )
    return results_df


def run_experiment_sequence(
    simulations_df: pl.DataFrame, summary_function=None
):
    """Takes in Polars Dataframe of simulation (full) parameter data and outputs complete or summarized results from a stochastic SIR model"""
    results_dict = {}
    for simulation_params in simulations_df.rows(named=True):
        results = run_toy_model(simulation_params)

        if summary_function:
            summarized_results = summary_function(results)
            results_dict[simulation_params["simulation"]] = summarized_results
        else:
            results_dict[simulation_params["simulation"]] = results

    return results_dict


def calculate_infection_metrics(df):
    """User-defined function to calculate infection metrics, in this case time to peak infection and total infected"""
    # Find the time of peak infection (first instance, if multiple)
    time_to_peak_infection = (
        df.sort("infected", descending=True).select(pl.first("time"))
    ).item()

    # Calculate total infected by taking the maximum value from the 'recovered' column
    total_infected = df.get_column("recovered").max()

    metrics = (time_to_peak_infection, total_infected)
    return metrics


def calculate_distance(results_data: tuple, target_data: tuple) -> float:
    """User-defined function to measure Euclidean distance from target"""

    # Unpack data from tuples
    time_to_peak_infection_results, total_infected_results = results_data
    time_to_peak_infection_target, total_infected_target = target_data

    # Calculate differences
    time_diff = time_to_peak_infection_results - time_to_peak_infection_target
    infected_diff = total_infected_results - total_infected_target

    # Compute Euclidean distance
    distance = math.sqrt(time_diff**2 + infected_diff**2)

    return distance


class TestABCPipeline(unittest.TestCase):
    def setUp(self):
        # Number of simulations (per step/iteration/generation)
        self.n_init = 100  # Number of simulations to initialize each step
        self.n_required = (
            30  # Number of accepted simulations required to proceed
        )

        # Number of steps/iterations/generations
        self.n_steps = 6

        # Set stochasticity
        self.stochastic = True

        # Baseline parameters
        self.baseline_params = {
            "population": 10000,
            "totalDays": 100,
            "initialPrevalence": 0.03,
        }

        # Define prior distributions for the experiment parameters
        self.experiment_params_prior_dist = {
            "averageContactsPerDay": abc_methods.get_truncated_normal(
                mean=2, sd=1, low=0.01, upp=25
            ),
            "infectiousPeriod": abc_methods.get_truncated_normal(
                mean=4, sd=2.5, low=0.01, upp=25
            ),
        }

        # Generate target_data
        self.target_params = {
            "averageContactsPerDay": 1.7,
            "infectiousPeriod": 5,
        }
        target_params_dict = (
            self.baseline_params | self.target_params | {"randomSeed": 142}
        )
        self.target_data = run_toy_model(target_params_dict)
        self.target_metrics = calculate_infection_metrics(self.target_data)
        print(self.target_data.head(10))

        # Set tolerance level
        # Note, the final is high because it's just a draw from the approximated posterior distribution
        self.tolerance = [
            500,
            250,
            100,
            50,
            20,
            10000,
        ]  # todo: turn into percentages

        # Define perturbation kernels
        self.perturbation_kernels = {
            "averageContactsPerDay": uniform(-0.05, 0.1),
            "infectiousPeriod": uniform(-0.1, 0.2),
        }

    def test_pipeline(self):
        # Initialize empty dictionary to keep track of SimulationBundle objectes for each ABC step (iteration/generation)
        sim_bundles = {}

        for step_number in range(self.n_steps):
            # For the initialization step (step 0), sample from the priors
            # For steps 1+, the current sim_bundle is regenerated at the end of the previous loop
            if step_number == 0:
                with self.subTest("Initialize Samples"):
                    input_df = abc_methods.draw_simulation_parameters(
                        params_inputs=self.experiment_params_prior_dist,
                        n_simulations=self.n_init,
                        add_random_seed=self.stochastic,
                        seed=random_seed,
                    )
                    self.assertEqual(input_df.shape[0], self.n_init)
            else:
                # Only perturb if it's not the final step
                if step_number != (self.n_steps - 1):
                    with self.subTest(
                        f"Resample, step #{step_number} (includes perturbation and validation checks against prior distributions"
                    ):
                        input_df = abc_methods.resample(
                            sim_bundles[step_number - 1].accepted,
                            n_samples=self.n_init,
                            perturbation_kernels=self.perturbation_kernels,
                            prior_distributions=self.experiment_params_prior_dist,
                            weights=sim_bundles[step_number - 1].weights,
                            seed=random_seed,
                        )
                        self.assertEqual(input_df.shape[0], self.n_init)
                else:
                    with self.subTest(f"Resample, final step #{step_number}"):
                        input_df = abc_methods.resample(
                            sim_bundles[step_number - 1].accepted,
                            n_samples=self.n_init,
                            weights=sim_bundles[step_number - 1].weights,
                            seed=random_seed,
                        )
                        self.assertEqual(input_df.shape[0], self.n_init)

            print(f"Step {step_number}, trying {len(input_df)} samples")

            # Create the simulation bundle for the current step
            sim_bundle = SimulationBundle(
                inputs=input_df,
                step_number=step_number,
                baseline_params=self.baseline_params,
            )

            with self.subTest(f"Run Model, step #{step_number}"):
                sim_bundle.results = run_experiment_sequence(
                    sim_bundle.full_params_df
                )
                self.assertEqual(len(sim_bundle.results), self.n_init)
                self.assertIn("time", sim_bundle.results[0].columns)
                self.assertIn("infected", sim_bundle.results[0].columns)
                self.assertIn("recovered", sim_bundle.results[0].columns)

            with self.subTest(
                f"Calculate Summary Metrics, step #{step_number}"
            ):
                sim_bundle.calculate_summary_metrics(
                    calculate_infection_metrics
                )
                for _, value in sim_bundle.summary_metrics.items():
                    self.assertIsInstance(value, tuple)
                    # todo: could ensure time to peak infection is in reasonable range

            with self.subTest(f"Calculate Distances, step #{step_number}"):
                sim_bundle.calculate_distances(
                    self.target_metrics, calculate_distance
                )
                # Ensure all distances are >=0
                for distance in sim_bundle.distances.values():
                    self.assertGreaterEqual(distance, 0)

            with self.subTest(
                f"Accept or Reject Simulations, step #{step_number}"
            ):
                sim_bundle.accept_reject(self.tolerance[step_number])

                # Ensure at least one simulation is accepted
                self.assertGreaterEqual(len(sim_bundle.accepted), 1)

            with self.subTest(
                f"Check Accepted Simulations and Run/Check/Merge Additional if Needed, step #{step_number}"
            ):
                # Continue adding simulations until the required number is accepted
                fractional_acceptance = sim_bundle.n_accepted / self.n_init
                print(f"{fractional_acceptance*100:.1f}% of samples accepted")
                while sim_bundle.n_accepted < self.n_required:
                    # Calculate how many more simulations need to be initialized
                    n_additional = int(
                        (self.n_required - sim_bundle.n_accepted)
                        * 1.5
                        / fractional_acceptance
                    )

                    # Initialize more samples as a new SimulationBundle additional_sim_bundle
                    if step_number == 0:
                        additional_input_df = abc_methods.draw_simulation_parameters(
                            params_inputs=self.experiment_params_prior_dist,
                            n_simulations=n_additional,
                            add_random_seed=self.stochastic,
                            starting_simulation_number=sim_bundle.n_simulations,
                            seed=random_seed,
                        )
                    else:
                        additional_input_df = abc_methods.resample(
                            sim_bundles[step_number - 1].accepted,
                            n_samples=n_additional,
                            perturbation_kernels=self.perturbation_kernels,
                            prior_distributions=self.experiment_params_prior_dist,
                            weights=sim_bundles[step_number - 1].weights,
                            starting_simulation_number=sim_bundle.n_simulations,
                            seed=random_seed,
                        )

                    print(
                        f"Step {step_number}, trying {len(additional_input_df)} additional samples"
                    )

                    # Create additional SimulationBundle (will be merged into the current step's sim_bundle once evaluated)
                    additional_sim_bundle = SimulationBundle(
                        inputs=additional_input_df,
                        step_number=step_number,
                        baseline_params=self.baseline_params,
                    )

                    # Run model for additional simulations
                    additional_sim_bundle.results = run_experiment_sequence(
                        additional_sim_bundle.full_params_df
                    )

                    # Calculate summary metrics for the additional simulations
                    additional_sim_bundle.calculate_summary_metrics(
                        calculate_infection_metrics
                    )

                    # Calculate distances for the additional simulations
                    additional_sim_bundle.calculate_distances(
                        self.target_metrics,
                        calculate_distance,
                        use_summary_metrics=True,
                    )

                    # Accept or reject the additional simulations based on tolerance criteria
                    additional_sim_bundle.accept_reject(
                        self.tolerance[step_number]
                    )

                    # Merge the new SimulationBundle into current sim_bundle
                    sim_bundle.merge_with(
                        additional_sim_bundle
                    )  # Notes: -pay attention to simulation numbers, -keep track of number of additional runs/merges

                # Check if there are enough accepted simulations
                print(
                    f"Step {step_number}, {sim_bundle.n_accepted} samples accepted"
                )
                self.assertGreaterEqual(sim_bundle.n_accepted, self.n_required)

            with self.subTest(f"Calculate weights, step #{step_number}"):
                if step_number == 0:
                    # uniform weights on the initial step
                    weights = {
                        key: 1 / len(sim_bundle.accepted)
                        for key in sim_bundle.accepted
                    }
                else:
                    prev_step_accepted = sim_bundles[step_number - 1].accepted
                    prev_step_weights = sim_bundles[step_number - 1].weights
                    weights = abc_methods.calculate_weights_abcsmc(
                        sim_bundle.accepted,
                        prev_step_accepted,
                        prev_step_weights,
                        self.experiment_params_prior_dist,
                        self.perturbation_kernels,
                        normalize=True,
                    )

                sim_bundle.weights = weights
                total_weight = sum(weights.values())
                self.assertAlmostEqual(total_weight, 1.0, places=5)

            # Add current sim_bundle to sim_bundles dictionary
            sim_bundles[step_number] = sim_bundle
            del sim_bundle

        ### Plots
        with self.subTest("Make timeseries plots"):
            output_folder = "abctools/tests/figs"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for step_number, sim_bundle in sim_bundles.items():
                data_list = list(sim_bundle.results.values())
                x_col = "time"
                y_col = "infected"
                plot_args_list = [{"color": "blue", "alpha": 0.5}] * len(
                    data_list
                )

                # Add target data
                data_list.append(self.target_data)
                plot_args_list.append({"color": "red", "alpha": 0.8})

                # Label axes
                xlabel = "Time (days)"
                ylabel = "Infections"

                # Plot
                fig = plot_utils.plot_xy_data(
                    data_list, x_col, y_col, plot_args_list, xlabel, ylabel
                )
                file_out = os.path.join(
                    output_folder,
                    f"infection_timeseries_step_{step_number}.jpg",
                )
                fig.savefig(file_out)

        with self.subTest("Make results plots"):
            for step_number, sim_bundle in sim_bundles.items():
                data_list = []
                for summary_metrics in sim_bundle.summary_metrics.values():
                    data_list.append(
                        pl.DataFrame(
                            {
                                "time_to_peak_infections": [
                                    summary_metrics[0]
                                ],
                                "total_infected": [summary_metrics[1]],
                            }
                        )
                    )
                x_col = "time_to_peak_infections"
                y_col = "total_infected"
                plot_args_list = [
                    {"color": "blue", "alpha": 0.10, "marker": "o"}
                ] * len(data_list)

                # Add target data
                data_list.append(
                    pl.DataFrame(
                        {
                            "time_to_peak_infections": [
                                self.target_metrics[0]
                            ],
                            "total_infected": [self.target_metrics[1]],
                        }
                    )
                )
                plot_args_list.append(
                    {"color": "red", "alpha": 0.9, "marker": "o"}
                )
                # Label axes
                xlabel = "Time to peak infections"
                ylabel = "Total Infections"

                # Plot
                fig = plot_utils.plot_xy_data(
                    data_list, x_col, y_col, plot_args_list, xlabel, ylabel
                )
                file_out = os.path.join(
                    output_folder,
                    f"target_metrics_step_{step_number}.jpg",
                )
                fig.savefig(file_out)

            with self.subTest("Make parameter histograms"):
                output_folder = "abctools/tests/figs/parameters"
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                for step_number, sim_bundle in sim_bundles.items():
                    for experiment_param in sim_bundle.experiment_params:
                        df = sim_bundle.inputs
                        col = experiment_param
                        vline_value = self.target_params[experiment_param]

                        plt.hist(df[col])
                        plt.axvline(
                            x=vline_value,
                            color="red",
                        )
                        plt.xlabel(experiment_param)
                        plt.ylabel("frequency")

                        file_out = os.path.join(
                            output_folder,
                            f"{experiment_param}_hist_step_{step_number}.jpg",
                        )
                        plt.savefig(file_out)
                        plt.close()

        with self.subTest("Posterior predictive check"):
            # TODO: add posterior predictive check
            pass
            # Add assertion


if __name__ == "__main__":
    unittest.main()
