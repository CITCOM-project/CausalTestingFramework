import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import Positive
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator
from matplotlib.pyplot import rcParams

# Uncommenting the code below will make all graphs publication quality but requires a suitable latex installation

# plt.rcParams["figure.figsize"] = (8, 8)
# rc_fonts = {
#     "font.size": 8,
#     "figure.figsize": (10, 6),
#     "text.usetex": True,
#     "font.family": "serif",
#     "text.latex.preamble": r"\usepackage{libertine}",
# }
# rcParams.update(rc_fonts)

OBSERVATIONAL_DATA_PATH = "data/10k_observational_data.csv"


def doubling_beta_CATE_on_csv(observational_data_path: str, simulate_counterfactuals: bool = False,
                              verbose: bool = False):
    """ Compute the CATE of increasing beta from 0.016 to 0.032 on cum_infections using the dataframe
    loaded from the specified path. Additionally simulate the counterfactuals by repeating the analysis
    after removing rows with beta==0.032.

    :param observational_data_path: Path to csv containing observational data for analysis.
    :param simulate_counterfactuals: Whether to repeat analysis with counterfactuals.
    :param verbose: Whether to print verbose details (causal test results).
    :return results_dict: A nested dictionary containing results (ate and confidence intervals)
                          for association, causation, and counterfactual (if completed).
    """
    results_dict = {'association': {},
                    'causation': {}}

    # Read in the observational data and perform identification
    past_execution_df = pd.read_csv(observational_data_path)
    _, causal_test_engine = identification(observational_data_path)

    linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                            {'avg_age', 'contacts'},  # We use custom adjustment set
                                                            ('cum_infections',),
                                                            df=past_execution_df)

    # Add squared terms for beta, since it has a quadratic relationship with cumulative infections
    linear_regression_estimator.add_squared_term_to_df('beta')
    causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')

    # Repeat for association estimate (no adjustment)
    no_adjustment_linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                                          set(),
                                                                          ('cum_infections',),
                                                                          df=past_execution_df)
    no_adjustment_linear_regression_estimator.add_squared_term_to_df('beta')
    association_test_result = causal_test_engine.execute_test(no_adjustment_linear_regression_estimator, 'ate')

    # Store results for plotting
    results_dict['association'] = {'ate': association_test_result.ate,
                                   'cis': association_test_result.confidence_intervals,
                                   'df': past_execution_df}
    results_dict['causation'] = {'ate': causal_test_result.ate,
                                 'cis': causal_test_result.confidence_intervals,
                                 'df': past_execution_df}

    if verbose:
        print(f"Association:\n{association_test_result}")
        print(f"Causation:\n{causal_test_result}")

    # Repeat causal inference after deleting all rows with treatment value to obtain counterfactual inferences
    if simulate_counterfactuals:
        counterfactual_past_execution_df = past_execution_df[past_execution_df['beta'] != 0.032]
        counterfactual_linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                                               {'avg_age', 'contacts'},
                                                                               ('cum_infections',),
                                                                               df=counterfactual_past_execution_df)
        counterfactual_linear_regression_estimator.add_squared_term_to_df('beta')
        counterfactual_causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')
        results_dict['counterfactual'] = {'ate': counterfactual_causal_test_result.ate,
                                          'cis': counterfactual_causal_test_result.confidence_intervals,
                                          'df': counterfactual_past_execution_df}
        if verbose:
            print(f"Counterfactual:\n{counterfactual_causal_test_result}")

    return results_dict


def doubling_beta_CATEs(observational_data_path: str, simulate_counterfactual: bool = False, verbose: bool = False):
    """ Compute the CATE for the effect of doubling beta across simulations with different age demographics.
    To compute the CATE, this method splits the observational data into high and low age data and computes the
    ATE using this data and a linear regression model.

    Since this method already adjusts for age, adding age as an adjustment to the LR model will have no impact.
    However, adding contacts as an adjustment should reduce bias and reveal the average causal effect of doubling beta
    in simulations of a particular age demographic. """

    # Create separate subplots for each more specific causal question
    all_fig, all_axes = plt.subplots(1, 1, figsize=(4, 3), squeeze=False)
    age_fig, age_axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(7, 3), squeeze=False)
    age_contact_fig, age_contact_axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(7, 5))

    # Apply CT to get the ATE over all executions
    if verbose:
        print("Running causal tests for all data...")
    all_data_results_dict = doubling_beta_CATE_on_csv(observational_data_path, simulate_counterfactual, verbose=False)
    plot_doubling_beta_CATEs(all_data_results_dict, "All Data", all_fig, all_axes, row=0, col=0)

    # Split data into age-specific strata
    past_execution_df = pd.read_csv(observational_data_path)
    min_age = np.floor(past_execution_df['avg_age'].min())
    max_age = np.ceil(past_execution_df['avg_age'].max())
    mid_age = (min_age + max_age) / 2

    # Split df into two age ranges
    younger_population_df = past_execution_df.loc[past_execution_df['avg_age'] <= mid_age]
    younger_population_df.to_csv("./data/younger_population.csv")
    older_population_df = past_execution_df.loc[past_execution_df['avg_age'] > mid_age]
    older_population_df.to_csv("./data/older_population.csv")

    # Repeat analysis on age-specific strata
    separated_observational_data_paths = ["./data/younger_population.csv",
                                          "./data/older_population.csv"]

    for col, separated_observational_data_path in enumerate(separated_observational_data_paths):
        age_data_results_dict = doubling_beta_CATE_on_csv(separated_observational_data_path, simulate_counterfactual,
                                                          verbose=False)
        age_stratified_df = pd.read_csv(separated_observational_data_path)
        age_stratified_df_avg_age = round(age_stratified_df["avg_age"].mean(), 1)
        if verbose:
            print(f"Running causal tests for data with average age of {age_stratified_df_avg_age}")
        plot_doubling_beta_CATEs(age_data_results_dict, f"Age={age_stratified_df_avg_age}", age_fig, age_axes, row=0,
                                 col=col)

        # Split df into contact-specific strata
        min_contacts = np.floor(age_stratified_df['contacts'].min())
        max_contacts = np.ceil(age_stratified_df['contacts'].max())
        mid_contacts = (max_contacts + min_contacts) / 2

        # Save dfs to csv
        low_contacts_df = age_stratified_df.loc[age_stratified_df['contacts'] <= mid_contacts]
        low_contacts_df.to_csv(f"./data/low_contacts_avg_age_{age_stratified_df_avg_age}.csv")
        high_contacts_df = age_stratified_df.loc[age_stratified_df['contacts'] > mid_contacts]
        high_contacts_df.to_csv(f"./data/high_contacts_avg_age_{age_stratified_df_avg_age}.csv")

        contact_observational_data_paths = [f"./data/low_contacts_avg_age_"
                                            f"{age_stratified_df_avg_age}.csv",
                                            f"./data/high_contacts_avg_age_"
                                            f"{age_stratified_df_avg_age}.csv"]

        # Compute the CATE for each age-contact group
        for row, age_contact_data_path in enumerate(contact_observational_data_paths):
            age_contact_data_results_dict = doubling_beta_CATE_on_csv(age_contact_data_path, simulate_counterfactual,
                                                                      verbose=False)
            age_contact_stratified_df = pd.read_csv(age_contact_data_path)
            age_contact_stratified_df_avg_contacts = round(age_contact_stratified_df["contacts"].mean(), 1)
            if verbose:
                print(f"Running causal tests for data with average age of {age_stratified_df_avg_age} and "
                      f"{age_contact_stratified_df_avg_contacts} average household contacts.")
            plot_doubling_beta_CATEs(age_contact_data_results_dict,
                                     f"Age={age_stratified_df_avg_age} "
                                     f"Contacts={age_contact_stratified_df_avg_contacts}",
                                     age_contact_fig, age_contact_axes, row=row, col=col)

    # Save plots
    if simulate_counterfactual:
        outpath_base_str = './counterfactuals_'
    else:
        outpath_base_str = './'
    all_fig.savefig(outpath_base_str + "all_executions.pdf", format="pdf")
    age_fig.savefig(outpath_base_str + "age_executions.pdf", format="pdf")
    age_contact_fig.savefig(outpath_base_str + "age_contact_executions.pdf", format="pdf")


def identification(observational_data_path):
    # 1. Read in the Causal DAG
    causal_dag = CausalDAG('dag.dot')

    # 2. Create variables
    pop_size = Input('pop_size', int)
    pop_infected = Input('pop_infected', int)
    n_days = Input('n_days', int)
    cum_infections = Output('cum_infections', int)
    cum_deaths = Output('cum_deaths', int)
    location = Input('location', str)
    variants = Input('variants', str)
    avg_age = Input('avg_age', float)
    beta = Input('beta', float)
    contacts = Input('contacts', float)

    # 3. Create scenario by applying constraints over a subset of the input variables
    scenario = Scenario(variables={pop_size, pop_infected, n_days, cum_infections, cum_deaths,
                                   location, variants, avg_age, beta, contacts},
                        constraints={pop_size.z3 == 51633, pop_infected.z3 == 1000, n_days.z3 == 216})

    # 4. Construct a causal specification from the scenario and causal DAG
    causal_specification = CausalSpecification(scenario, causal_dag)

    # 5. Create a causal test case
    causal_test_case = CausalTestCase(control_input_configuration={beta: 0.016},
                                      expected_causal_effect=Positive,
                                      treatment_input_configuration={beta: 0.032},
                                      outcome_variables={cum_infections})

    # 6. Create a data collector
    data_collector = ObservationalDataCollector(scenario, observational_data_path)

    # 7. Create an instance of the causal test engine
    causal_test_engine = CausalTestEngine(causal_test_case, causal_specification, data_collector)

    # 8. Obtain the minimal adjustment set for the causal test case from the causal DAG
    minimal_adjustment_set = causal_test_engine.load_data(index_col=0)

    return minimal_adjustment_set, causal_test_engine


def plot_doubling_beta_CATEs(results_dict, title, figure=None, axes=None, row=None, col=None):
    # Get the CATE as a percentage for association and causation
    ate = results_dict['causation']['ate']
    association_ate = results_dict['association']['ate']

    causation_df = results_dict['causation']['df']
    association_df = results_dict['association']['df']

    percentage_ate = round((ate / causation_df['cum_infections'].mean()) * 100, 3)
    association_percentage_ate = round((association_ate / association_df['cum_infections'].mean()) * 100, 3)

    # Get 95% confidence intervals for association and causation
    ate_cis = results_dict['causation']['cis']
    association_ate_cis = results_dict['association']['cis']
    percentage_causal_ate_cis = [round(((ci / causation_df['cum_infections'].mean()) * 100), 3) for ci in ate_cis]
    percentage_association_ate_cis = [round(((ci / association_df['cum_infections'].mean()) * 100), 3) for ci in
                                      association_ate_cis]

    # Convert confidence intervals to errors for plotting
    percentage_causal_errs = [percentage_ate - percentage_causal_ate_cis[0],
                              percentage_causal_ate_cis[1] - percentage_ate]
    percentage_association_errs = [association_percentage_ate - percentage_association_ate_cis[0],
                                   percentage_association_ate_cis[1] - association_percentage_ate]

    xs = [1, 2]
    ys = [association_percentage_ate, percentage_ate]
    yerrs = [percentage_association_errs, percentage_causal_errs]
    xticks = ['Association', 'Causation']
    print(f"Association ATE: {association_percentage_ate} {percentage_association_ate_cis}")
    print(f"Association executions: {len(association_df)}")
    print(f"Causal ATE: {percentage_ate} {percentage_causal_ate_cis}")
    print(f"Causal executions: {len(causation_df)}")
    if 'counterfactual' in results_dict.keys():
        cf_ate = results_dict['counterfactual']['ate']
        cf_df = results_dict['counterfactual']['df']
        percentage_cf_ate = round((cf_ate / cf_df['cum_infections'].mean()) * 100, 3)
        cf_ate_cis = results_dict['counterfactual']['cis']
        percentage_cf_cis = [round(((ci / cf_df['cum_infections'].mean()) * 100), 3) for ci in cf_ate_cis]
        percentage_cf_errs = [percentage_cf_ate - percentage_cf_cis[0],
                              percentage_cf_cis[1] - percentage_cf_ate]
        xs = [0.5, 1.5, 2.5]
        ys = [association_percentage_ate, percentage_ate, percentage_cf_ate]
        yerrs = np.array([percentage_association_errs, percentage_causal_errs, percentage_cf_errs]).T
        xticks = ['Association', 'Causation', 'Counterfactual']
        print(f"Counterfactual ATE: {percentage_cf_ate} {percentage_cf_cis}")
        print(f"Counterfactual executions: {len(cf_df)}")
    axes[row, col].set_ylim(0, 30)
    axes[row, col].set_xlim(0, 3)
    axes[row, col].set_xticks(xs, xticks)
    axes[row, col].set_title(title)
    axes[row, col].errorbar(xs, ys, yerrs, fmt='o', markersize=3, capsize=3, markerfacecolor='red', color='black')
    figure.supylabel(r"\% Change in Cumulative Infections (ATE)", fontsize=10)


if __name__ == "__main__":
    doubling_beta_CATEs(OBSERVATIONAL_DATA_PATH, True, True)
