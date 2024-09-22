import random

import numpy as np


def ctmc_gillespie_model(initial_state, params, tmax, t=0, random_seed=None):
    """
    Simulate the stochastic dynamics of an SIR model using the Gillespie algorithm. This model uses a stochastic simulation algorithm that generates sample paths of a continuous-time Markov chain (CTMC).

    Parameters:
        initial_state (tuple): A tuple containing the initial counts of susceptible (S0), infected (I0), and recovered (R0) individuals in the population.
        params (tuple): A tuple containing the parameters of the model: infection rate (beta) and recovery rate (gamma).
        tmax (float): The maximum time until which to run the simulation.
        t (float, optional): The starting time of the simulation. Defaults to 0 if not specified.
        random_seed (int, optional): An optional seed for the random number generator to ensure reproducibility. If not provided, results will vary between runs.

    Returns:
        time_points (list): A list containing timestamps at which events occurred during the simulation.
        susceptible (list): A list containing the count of susceptible individuals corresponding to each timestamp in `time_points`.
        infected (list): A list containing the count of infected individuals corresponding to each timestamp in `time_points`.
        recovered (list): A list containing the count of recovered individuals corresponding to each timestamp in `time_points`.

    References:
        Allen LJS. A primer on stochastic epidemic models: Formulation, numerical simulation, and analysis. Infect Dis Model. 2017 Mar 11;2(2):128-142. doi: 10.1016/j.idm.2017.03.001. PMID: 29928733; PMCID: PMC6002090.

    Example usage:
        # Define initial conditions and parameters
        initial_state = (990, 10, 0)
        params = (0.3, 0.1)

        # Run simulation up to time tmax=100
        time_points, susceptible_counts, infected_counts, recovered_counts = ctmc_gillespie_model(
            initial_state=initial_state,
            params=params,
            tmax=100,
            random_seed=42
        )
    """

    if random_seed:
        random.seed(random_seed)

    S0, I0, R0 = initial_state
    N = S0 + I0 + R0
    beta, gamma = params
    event_count = 0

    # Initiate lists to store results
    time_points = [t]
    susceptible = [S0]
    infected = [I0]
    recovered = [R0]

    # Initialize the states
    S, Inf, R = S0, I0, R0

    while t < tmax:
        # Calculate transition rates
        rate_infection = beta * S * Inf / N
        rate_recovery = gamma * Inf
        rate_total = rate_infection + rate_recovery

        # Check if total rate is 0, which means no more events can occur
        if rate_total == 0:
            break

        # Generate two random numbers
        r1, r2 = random.random(), random.random()

        # Calculate time until next event and update time
        dt = float(-np.log(r1) / rate_total)
        t += dt

        # Determine which event occurs
        if r2 < rate_infection / rate_total:
            # Infection event
            S -= 1
            Inf += 1
        else:
            # Recovery Event
            Inf -= 1
            R += 1

        # Record the current state and time
        time_points.append(t)
        susceptible.append(S)
        infected.append(Inf)
        recovered.append(R)

        event_count += 1

    # After the while loop has finished, check if we need to fill in values up to tmax
    if t < tmax:
        # Fill in the time points up to tmax with the last known state
        while t < tmax:
            t = min(t + 1, tmax)

            # Append the current state to each list
            time_points.append(t)
            susceptible.append(S)
            infected.append(Inf)
            recovered.append(R)

    return time_points, susceptible, infected, recovered
