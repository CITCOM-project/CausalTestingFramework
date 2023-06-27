# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import covasim as cv  # Version used in our study is 3.07
import random
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.data_collection.data_collector import ExperimentalDataCollector
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import Positive, Negative, NoEffect
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase

import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")

ROOT = os.path.realpath(os.path.dirname(__file__))


def test_experimental_vaccinate_elderly(runs_per_test_per_config: int = 30, verbose: bool = False):
    """Run the causal test case for the effect of changing vaccine to prioritise elderly. This uses the experimental
        data collector.

    :param runs_per_test_per_config: Number of times to run each input configuration (control and treatment) per test.
                                     Hence, the total number of runs per test will be twice this value.
    :param verbose: Whether to print verbose details (causal test results).
    :return results_dict: A dictionary containing ATE, 95% CIs, and Test Pass/Fail
    """

    # 1. Read in the Causal DAG
    causal_dag = CausalDAG(f"{ROOT}/dag.dot")

    # 2. Create variables
    pop_size = Input("pop_size", int)
    pop_infected = Input("pop_infected", int)
    n_days = Input("n_days", int)
    vaccine = Input("vaccine", int)
    cum_infections = Output("cum_infections", int)
    cum_vaccinations = Output("cum_vaccinations", int)
    cum_vaccinated = Output("cum_vaccinated", int)
    max_doses = Output("max_doses", int)

    # 3. Create scenario by applying constraints over a subset of the input variables
    scenario = Scenario(
        variables={
            pop_size,
            pop_infected,
            n_days,
            cum_infections,
            vaccine,
            cum_vaccinated,
            cum_vaccinations,
            max_doses,
        },
        constraints={pop_size.z3 == 50000, pop_infected.z3 == 1000, n_days.z3 == 50},
    )

    # 4. Construct a causal specification from the scenario and causal DAG
    causal_specification = CausalSpecification(scenario, causal_dag)

    # 5. Instantiate the experimental data collector for Covasim
    covasim_parameters_dict = {"pop_size": 50000, "pop_type": "hybrid", "pop_infected": 1000, "n_days": 50}
    control_input_configuration = {"covasim_parameters_dict": covasim_parameters_dict, "target_elderly": False}
    treatment_input_configuration = {"covasim_parameters_dict": covasim_parameters_dict, "target_elderly": True}
    data_collector = CovasimVaccineDataCollector(
        scenario, control_input_configuration, treatment_input_configuration, runs_per_test_per_config
    )

    # 6. Express expected outcomes
    expected_outcome_effects = {
        cum_infections: Positive(),
        cum_vaccinations: Negative(),
        cum_vaccinated: Negative(),
        max_doses: NoEffect(),
    }
    results_dict = {"cum_infections": {}, "cum_vaccinations": {}, "cum_vaccinated": {}, "max_doses": {}}

    # 7. Create an instance of the causal test engine
    causal_test_engine = CausalTestEngine(causal_specification, data_collector, index_col=0)

    for outcome_variable, expected_effect in expected_outcome_effects.items():
        base_test_case = BaseTestCase(treatment_variable=vaccine, outcome_variable=outcome_variable)
        causal_test_case = CausalTestCase(
            base_test_case=base_test_case, expected_causal_effect=expected_effect, control_value=0, treatment_value=1
        )

        # 8. Obtain the minimal adjustment set for the causal test case from the causal DAG
        minimal_adjustment_set = causal_dag.identification(base_test_case)

        # 9. Build statistical model
        linear_regression_estimator = LinearRegressionEstimator(
            vaccine.name, 1, 0, minimal_adjustment_set, outcome_variable.name
        )

        # 10. Execute test and save results in dict
        causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, causal_test_case)
        if verbose:
            logging.info("Causation:\n%s", causal_test_result)
        results_dict[outcome_variable.name]["ate"] = causal_test_result.test_value.value
        results_dict[outcome_variable.name]["cis"] = causal_test_result.confidence_intervals
        results_dict[outcome_variable.name]["test_passes"] = causal_test_case.expected_causal_effect.apply(
            causal_test_result
        )
    return results_dict


class CovasimVaccineDataCollector(ExperimentalDataCollector):
    """A custom experimental data collector for the elderly vaccination Covasim case study.

    This experimental data collector runs covasim with a normal Pfizer vaccine and then again with the same vaccine but
    this time prioritising the elderly for vaccination.
    """

    def run_system_with_input_configuration(self, input_configuration: dict) -> pd.DataFrame:
        """Run the system with a given input configuration.

        :param input_configuration: A nested dictionary containing Covasim parameters, desired number of repeats, and
        a bool to determine whether elderly should be prioritised for vaccination.
        :return: A dataframe containing results for this input configuration.
        """
        results_df = self.simulate_vaccine(
            input_configuration["covasim_parameters_dict"], self.n_repeats, input_configuration["target_elderly"]
        )
        return results_df

    def simulate_vaccine(self, pars_dict: dict, n_simulations: int = 100, target_elderly: bool = False):
        """Simulate observational data that contains a vaccine that is optionally given preferentially to the elderly.

        :param pars_dict: A dictionary containing simulation parameters.
        :param n_simulations: Number of simulations to run.
        :param target_elderly: Whether to prioritise vaccination for the elderly.
        :return: A pandas dataframe containing results for each run.
        """
        simulations_results_dfs = []
        for sim_n in range(n_simulations):
            logging.info("Simulation %s/%s.", sim_n + 1, n_simulations)

            # Update simulation parameters with vaccine and optionally sub-target
            if target_elderly:
                logger.info("Prioritising the elderly for vaccination")
                vaccine = cv.vaccinate_prob(
                    vaccine="Pfizer",
                    label="prioritise_elderly",
                    subtarget=self.vaccinate_by_age,
                    days=list(range(7, pars_dict["n_days"])),
                )
            else:
                logger.info("Using standard vaccination protocol")
                vaccine = cv.vaccinate_prob(vaccine="Pfizer", label="regular", days=list(range(7, pars_dict["n_days"])))

            pars_dict["interventions"] = vaccine
            pars_dict["use_waning"] = True  # Must be set to true for vaccination
            sim_results_df = self.run_sim_with_pars(
                pars_dict=pars_dict,
                desired_outputs=[
                    "cum_infections",
                    "cum_deaths",
                    "cum_recoveries",
                    "cum_vaccinations",
                    "cum_vaccinated",
                ],
                n_runs=1,
            )

            sim_results_df["interventions"] = vaccine.label  # Store label in results instead of vaccine object
            sim_results_df["target_elderly"] = target_elderly
            sim_results_df["vaccine"] = int(target_elderly)  # 0 if standard vaccine, 1 if target elderly vaccine
            sim_results_df["max_doses"] = vaccine.p["doses"]  # Get max doses for the vaccine
            simulations_results_dfs.append(sim_results_df)

        # Create a single dataframe containing a row for every execution
        obs_df = pd.concat(simulations_results_dfs, ignore_index=True)
        obs_df.rename(columns={"interventions": "vaccine_type"}, inplace=True)
        return obs_df

    @staticmethod
    def run_sim_with_pars(pars_dict: dict, desired_outputs: [str], n_runs: int = 1, verbose: int = -1):
        """Runs a Covasim COVID-19 simulation with a given dict of parameters and collects the desired outputs,
            which are given as a list of output names.

        :param pars_dict: A dictionary containing the parameters and their values for the run.
        :param desired_outputs: A list of outputs which should be collected.
        :param n_runs: Number of times to run the simulation with a different seed.
        :param verbose: Covasim verbose setting (0 for no output, 1 for output).

        :return results_df: A pandas df containing the results for each run
        """
        results_dict = {k: [] for k in list(pars_dict.keys()) + desired_outputs + ["rand_seed"]}
        for _ in range(n_runs):
            # For every run, generate and use a new a random seed.
            # This is to avoid using Covasim's sequential random seeds.
            random.seed()
            rand_seed = random.randint(0, 10000)
            pars_dict["rand_seed"] = rand_seed
            logger.info("Rand Seed: %s", rand_seed)
            sim = cv.Sim(pars=pars_dict)
            m_sim = cv.MultiSim(sim)
            m_sim.run(n_runs=1, verbose=False, n_cpus=1)

            for run in m_sim.sims:
                results = run.results
                # Append inputs to results
                for param in pars_dict.keys():
                    results_dict[param].append(run.pars[param])

                # Append outputs to results
                for output in desired_outputs:
                    if output not in results:
                        raise IndexError(f"{output} is not in the Covasim outputs.")
                    results_dict[output].append(
                        results[output][-1]
                    )  # Append the final recorded value for each variable

        # Any parameters without results are assigned np.nan for each execution
        for param, results in results_dict.items():
            if not results:
                results_dict[param] = [np.nan] * len(results_dict["rand_seed"])
        return pd.DataFrame(results_dict)

    @staticmethod
    def vaccinate_by_age(simulation):
        """A custom method to prioritise vaccination of the elderly. This method is taken from Covasim Tutorial 5:
        https://github.com/InstituteforDiseaseModeling/covasim/blob/7bdf2ddf743f8798fcada28a61a03135d106f2ee/
        examples/t05_vaccine_subtargeting.py

        :param simulation: A covasim simulation for which the elderly will be prioritised for vaccination.
        :return output: A dictionary mapping individuals to vaccine probabilities.
        """
        young = cv.true(simulation.people.age < 50)
        middle = cv.true((simulation.people.age >= 50) * (simulation.people.age < 75))
        old = cv.true(simulation.people.age > 75)
        inds = simulation.people.uid
        vals = np.ones(len(simulation.people))
        vals[young] = 0.1
        vals[middle] = 0.5
        vals[old] = 0.9
        output = dict(inds=inds, vals=vals)
        return output


if __name__ == "__main__":
    test_results = test_experimental_vaccinate_elderly(runs_per_test_per_config=30, verbose=True)
    logging.info("%s", test_results)
