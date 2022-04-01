import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import matplotlib as mpl
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import Positive
from causal_testing.testing.intervention import Intervention
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator, CausalForestEstimator
from matplotlib.pyplot import rcParams

plt.rcParams["figure.figsize"] = (8, 8)

OBSERVATIONAL_DATA_PATH = "./data/10k_observational_data.csv"

rc_fonts = {
    "font.size": 8,
    "figure.figsize": (10, 6),
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{libertine}",
}
rcParams.update(rc_fonts)


def concatenate_csvs_in_directory(directory_path, output_path):
    """ Concatenate all csvs in a given directory, assuming all csvs share the same header. This will stack the csvs
    vertically and will not reset the index.
    """
    dfs = []
    for csv_name in glob.glob(directory_path):
        dfs.append(pd.read_csv(csv_name, index_col=0))
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(output_path)


def manual_CATE(observational_data_path: str, simulate_counterfactual: bool = False):
    """ Compute the CATE for the effect of doubling beta across simulations with different age demographics.
    To compute the CATE, this method splits the observational data into high and low age data and computes the
    ATE using this data and a linear regression model.

    Since this method already adjusts for age, adding age as
    an adjustment to the LR model will have no impact. However, adding contacts as an adjustment should reduce
    bias and reveal the average causal effect of doubling beta in simulations of a particular age demographic. """
    # Apply CT to get the ATE over all executions
    past_execution_df = pd.read_csv(observational_data_path)
    _, causal_test_engine = identification(observational_data_path)
    linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                            {'avg_age', 'contacts'},
                                                            ('cum_infections',),
                                                            df=past_execution_df)
    linear_regression_estimator.add_squared_term_to_df('beta')
    causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')

    no_adjustment_linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                                          set(),
                                                                          ('cum_infections',),
                                                                          df=past_execution_df)
    no_adjustment_linear_regression_estimator.add_squared_term_to_df('beta')
    association_test_result = causal_test_engine.execute_test(no_adjustment_linear_regression_estimator, 'ate')

    # Create a figure and axes to plot results for ATE
    all_fig, all_axes = plt.subplots(1, 1, figsize=(4, 3), squeeze=False)
    if simulate_counterfactual:
        counterfactual_past_execution_df = past_execution_df[past_execution_df['rel_beta'] != 2.05]
        counterfactual_linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                                               {'avg_age', 'contacts'},
                                                                               ('cum_infections',),
                                                                               df=past_execution_df)
        counterfactual_linear_regression_estimator.add_squared_term_to_df('beta')
        counterfactual_causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')
        plot_manual_CATE_result(causal_test_result, association_test_result, past_execution_df, title="All Executions",
                                figure=all_fig, axes=all_axes, row=0, col=0,
                                cf_CATE_result=counterfactual_causal_test_result,
                                cf_previous_data_df=counterfactual_past_execution_df)
    else:
        plot_manual_CATE_result(causal_test_result, association_test_result, past_execution_df, title="All Executions",
                                figure=all_fig, axes=all_axes, row=0, col=0)
    # Split the data into low and high age manually to get age-specific ATE (CATE by age)
    min_age = np.floor(past_execution_df['avg_age'].min())
    max_age = np.ceil(past_execution_df['avg_age'].max())
    mid_age = (min_age + max_age) / 2

    # Split df into two age ranges
    younger_population_df = past_execution_df.loc[past_execution_df['avg_age'] <= mid_age]
    younger_population_df.to_csv("./data/bessemer/younger_population.csv")
    older_population_df = past_execution_df.loc[past_execution_df['avg_age'] > mid_age]
    older_population_df.to_csv("./data/bessemer/older_population.csv")

    # Apply causal forest model to both datasets separately
    separated_observational_data_paths = ["./data/bessemer/younger_population.csv",
                                          "./data/bessemer/older_population.csv"]

    # Create a separate figure and axes for each CATE
    age_fig, age_axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(7, 3), squeeze=False)
    age_contact_fig, age_contact_axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(7, 5))

    # Compute the CATE for each age group
    for col, separated_observational_data_path in enumerate(separated_observational_data_paths):
        separated_observational_df = pd.read_csv(separated_observational_data_path)
        separated_observational_df_avg_age = round(separated_observational_df["avg_age"].mean(), 1)
        _, causal_test_engine = identification(separated_observational_data_path)

        linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                                {'avg_age', 'contacts'},
                                                                ('cum_infections',),
                                                                df=separated_observational_df)
        linear_regression_estimator.add_squared_term_to_df('beta')
        no_adjustment_linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                                              set(),
                                                                              ('cum_infections',),
                                                                              df=separated_observational_df)
        no_adjustment_linear_regression_estimator.add_squared_term_to_df('beta')
        causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')
        association_test_result = causal_test_engine.execute_test(no_adjustment_linear_regression_estimator, 'ate')
        if simulate_counterfactual:
            cf_separated_observational_df = separated_observational_df[separated_observational_df["rel_beta"] != 2.05]
            counterfactual_linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                                                   {'avg_age', 'contacts'},
                                                                                   ('cum_infections',),
                                                                                   df=cf_separated_observational_df)
            counterfactual_linear_regression_estimator.add_squared_term_to_df('beta')
            counterfactual_causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')
            plot_manual_CATE_result(causal_test_result, association_test_result, separated_observational_df,
                                    title=f"Average age = {separated_observational_df_avg_age}",
                                    figure=age_fig, axes=age_axes, row=0, col=col,
                                    cf_CATE_result=counterfactual_causal_test_result,
                                    cf_previous_data_df=cf_separated_observational_df)
        else:
            plot_manual_CATE_result(causal_test_result, association_test_result, separated_observational_df,
                                    title=f"Average age = {separated_observational_df_avg_age}",
                                    figure=age_fig, axes=age_axes, row=0, col=col)

        # Split the data into low and high household contacts to get age-contact-specific ATE (CATE by age and contacts)
        min_contacts = np.floor(separated_observational_df['contacts'].min())
        max_contacts = np.ceil(separated_observational_df['contacts'].max())
        mid_contacts = (max_contacts + min_contacts) / 2

        # Split df into two contact ranges
        low_contacts_df = separated_observational_df.loc[separated_observational_df['contacts'] <= mid_contacts]
        low_contacts_df.to_csv(f"./data/bessemer/low_contacts_avg_age_{separated_observational_df_avg_age}.csv")
        high_contacts_df = separated_observational_df.loc[separated_observational_df['contacts'] > mid_contacts]
        high_contacts_df.to_csv(f"./data/bessemer/high_contacts_avg_age_{separated_observational_df_avg_age}.csv")

        contact_observational_data_paths = [f"./data/bessemer/low_contacts_avg_age_"
                                            f"{separated_observational_df_avg_age}.csv",
                                            f"./data/bessemer/high_contacts_avg_age_"
                                            f"{separated_observational_df_avg_age}.csv"]

        # Compute the CATE for each age-contact group
        for row, contact_data_path in enumerate(contact_observational_data_paths):
            contact_df = pd.read_csv(contact_data_path)
            contact_df_avg_contacts = round(contact_df["contacts"].mean(), 1)
            _, causal_test_engine = identification(contact_data_path)

            linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                                    {'avg_age', 'contacts'},
                                                                    ('cum_infections',),
                                                                    df=contact_df)
            linear_regression_estimator.add_squared_term_to_df('beta')
            no_adjustment_linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                                                  set(),
                                                                                  ('cum_infections',),
                                                                                  df=contact_df)
            no_adjustment_linear_regression_estimator.add_squared_term_to_df('beta')
            causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')
            association_test_result = causal_test_engine.execute_test(no_adjustment_linear_regression_estimator, 'ate')
            if simulate_counterfactual:
                cf_contact_df = contact_df[contact_df["rel_beta"] != 2.05]
                counterfactual_linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032,
                                                                                       0.016,
                                                                                       {'avg_age', 'contacts'},
                                                                                       ('cum_infections',),
                                                                                       df=cf_contact_df)
                counterfactual_linear_regression_estimator.add_squared_term_to_df('beta')
                counterfactual_causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')
                plot_manual_CATE_result(causal_test_result, association_test_result, contact_df,
                                        title=f"Average age = {separated_observational_df_avg_age} and"
                                              f" contacts = {contact_df_avg_contacts}",
                                        figure=age_contact_fig, axes=age_contact_axes, row=row, col=col,
                                        cf_CATE_result=counterfactual_causal_test_result,
                                        cf_previous_data_df=cf_contact_df)
            else:
                plot_manual_CATE_result(causal_test_result, association_test_result, contact_df,
                                        title=f"Average age = {separated_observational_df_avg_age} and"
                                              f" contacts = {contact_df_avg_contacts}",
                                        figure=age_contact_fig, axes=age_contact_axes, row=row, col=col)
    # Save plots
    if simulate_counterfactual:
        outpath_base_str = './counterfactual_'
    else:
        outpath_base_str = './'
    all_fig.savefig(outpath_base_str + "all_executions.pdf", format="pdf")
    age_fig.savefig(outpath_base_str + "age_executions.pdf", format="pdf")
    age_contact_fig.savefig(outpath_base_str + "age_contact_executions.pdf", format="pdf")


def identification(observational_data_path):
    # 1. Read in the Causal DAG
    causal_dag = CausalDAG('./dag.dot')

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
                                      outcome_variables={cum_infections},
                                      intervention=Intervention((beta,), (0.032,), ), )

    # 6. Create a data collector
    data_collector = ObservationalDataCollector(scenario, observational_data_path)

    # 7. Create an instance of the causal test engine
    causal_test_engine = CausalTestEngine(causal_test_case, causal_specification, data_collector)

    # 8. Obtain the minimal adjustment set for the causal test case from the causal DAG
    minimal_adjustment_set = causal_test_engine.load_data(index_col=0)

    return minimal_adjustment_set, causal_test_engine


def causal_forest_CATE(observational_data_path):
    _, causal_test_engine = identification(observational_data_path)
    causal_forest_estimator = CausalForestEstimator(
        treatment=('beta',),
        treatment_values=0.032,
        control_values=0.016,
        adjustment_set={'avg_age', 'contacts'},
        outcome=('cum_infections',),
        effect_modifiers={causal_test_engine.scenario.variables['avg_age']})
    causal_forest_estimator_no_adjustment = CausalForestEstimator(
        treatment=('beta',),
        treatment_values=0.032,
        control_values=0.016,
        adjustment_set=set(),
        outcome=('cum_infections',),
        effect_modifiers={causal_test_engine.scenario.variables['avg_age']})

    # 10. Execute the test case and compare the results
    causal_test_result = causal_test_engine.execute_test(causal_forest_estimator, 'cate')
    association_test_result = causal_test_engine.execute_test(causal_forest_estimator_no_adjustment, 'cate')
    observational_data = pd.read_csv(observational_data_path)
    plot_causal_forest_result(causal_test_result, observational_data, "Causal Forest Adjusted for Age and Contacts.")
    plot_causal_forest_result(association_test_result, observational_data, "Causal Forest Without Adjustment.")


def plot_manual_CATE_result(causal_manual_CATE_result, association_manual_CATE_result, previous_data_df, title,
                            figure=None, axes=None, row=None, col=None, cf_CATE_result=None, cf_previous_data_df=None):
    # Get the CATE as a percentage for association and causation
    ate = causal_manual_CATE_result.ate
    association_ate = association_manual_CATE_result.ate
    percentage_ate = round((ate / previous_data_df['cum_infections'].mean()) * 100, 3)
    association_percentage_ate = round((association_ate / previous_data_df['cum_infections'].mean()) * 100, 3)

    # Get 95% confidence intervals for association and causation
    ate_cis = [causal_manual_CATE_result.ci_low(), causal_manual_CATE_result.ci_high()]
    association_ate_cis = [association_manual_CATE_result.ci_low(), association_manual_CATE_result.ci_high()]
    percentage_causal_ate_cis = [round(((ci / previous_data_df['cum_infections'].mean()) * 100), 3) for ci in ate_cis]
    percentage_association_ate_cis = [round(((ci / previous_data_df['cum_infections'].mean()) * 100), 3) for ci in
                                      association_ate_cis]

    # Convert confidence intervals to errors for plotting
    percentage_causal_errs = [percentage_ate - percentage_causal_ate_cis[0],
                              percentage_causal_ate_cis[1] - percentage_ate]
    percentage_association_errs = [association_percentage_ate - percentage_association_ate_cis[0],
                                   percentage_association_ate_cis[1] - association_percentage_ate]

    xs = [1, 2]
    ys = [association_percentage_ate, percentage_ate]
    yerrs = [percentage_association_errs, percentage_causal_errs]
    print(yerrs)
    xticks = ['Association', 'Causation']
    if cf_CATE_result:
        cf_ate = cf_CATE_result.ate
        percentage_cf_ate = round((cf_ate / cf_previous_data_df['cum_infections'].mean()) * 100, 3)
        cf_ate_cis = [cf_CATE_result.ci_low(), cf_CATE_result.ci_high()]
        percentage_cf_cis = [round(((ci / cf_previous_data_df['cum_infections'].mean()) * 100), 3) for ci in cf_ate_cis]
        percentage_cf_errs = [percentage_cf_ate - percentage_cf_cis[0],
                              percentage_cf_cis[1] - percentage_cf_ate]
        xs = [0.5, 1.5, 2.5]
        ys = [association_percentage_ate, percentage_ate, percentage_cf_ate]
        yerrs = np.array([percentage_association_errs, percentage_causal_errs, percentage_cf_errs]).T
        xticks = ['Association', 'Causation', 'Counterfactual']
        print(yerrs)
    # Plot the CATE and CIs for association and causation

    axes[row, col].set_ylim(0, 30)
    axes[row, col].set_xlim(0, 3)
    axes[row, col].set_xticks(xs, xticks)
    axes[row, col].set_title(title)
    axes[row, col].errorbar(xs, ys, yerrs, fmt='o', markersize=3, capsize=3, markerfacecolor='red', color='black')
    figure.supylabel(r"\% Change in Cumulative Infections (ATE)", fontsize=10)
    print(f"Causal ATE: {percentage_ate} {percentage_causal_ate_cis}")
    print(f"Association ATE: {association_percentage_ate} {percentage_association_ate_cis}")


def plot_causal_forest_result(causal_forest_test_result, previous_data_df, title=None, filter_data_by_variant=False):
    sorted_causal_forest_test_result = causal_forest_test_result.ate.sort_index()
    no_avg_age_causal_forest_test_result = sorted_causal_forest_test_result.drop(columns='avg_age')
    observational_df_with_results = previous_data_df.join(no_avg_age_causal_forest_test_result)
    observational_df_with_results['percentage_increase'] = \
        (observational_df_with_results['cate'] / observational_df_with_results['cum_infections']) * 100
    fig, ax = plt.subplots()
    if filter_data_by_variant:
        observational_df_with_results = observational_df_with_results.loc[observational_df_with_results['variants']
                                                                          == 'beta']
    for location in observational_df_with_results.location.unique():
        location_variant_df = observational_df_with_results.loc[observational_df_with_results['location'] == location]
        xs = location_variant_df['avg_age']
        ys = location_variant_df['percentage_increase']
        ax.scatter(xs, ys, s=1, alpha=.3, label=location)
        ax.set_ylabel("% change in cumulative infections")
        ax.set_xlabel("Average age")
        ax.set_title(title)
    ax.set_ylim(0, 40)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=4)
    plt.show()


if __name__ == "__main__":
    # concatenate_csvs_in_directory("./data/bessemer/custom_variants/thursday_31st_march/2k_executions/2k_data/*.csv",
    #                               "./data/10k_observational_data.csv")
    manual_CATE(OBSERVATIONAL_DATA_PATH, True)
    # causal_forest_CATE(OBSERVATIONAL_DATA_PATH)
