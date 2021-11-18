""" A module of causal inference methods and associated helpers. """
import pygraphviz
import numpy as np
from dowhy import CausalModel
from rpy2.robjects.packages import importr, isinstalled, STAP


def estimate_ate(df, treatment, outcome, control_val, treatment_val, dag, estimator="backdoor.linear_regression"):
    """
    Estimate the average treatment effect (ATE) of changing the treatment from control_val to treatment_val on the
    outcome in the data (df) using the causal assumptions outlined in the causal DAG and the specified estimator.
    :param df: Dataframe containing execution data from the system-under-test.
    :param treatment: The variable that is being intervened on.
    :param outcome: The outcome we expect the intervention to affect.
    :param control_val: The pre-intervention (control) value of the treatment variable.
    :param treatment_val: The post-intervention (treatment) value of the treatment variable.
    :param dag: The causal DAG depicting causal relationships amongst inputs and outputs in the scenario-under-test.
    :param estimator: The statistical estimator/model used to obtain the estimate.
    :return:
    """
    sufficient_adjustment_set = get_minimal_sufficient_adjustment_set(dag, treatment, outcome)
    causal_model = CausalModel(df, treatment, outcome, common_causes=sufficient_adjustment_set)
    causal_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
    causal_estimate = causal_model.estimate_effect(causal_estimand,
                                                   method_name=estimator,
                                                   control_value=control_val,
                                                   treatment_value=treatment_val,
                                                   confidence_intervals=True)
    return causal_estimate.target_estimand, causal_estimate.value, causal_estimate.get_confidence_intervals()


def get_minimal_sufficient_adjustment_set(dot_file_path, treatment, outcome):
    """
    Identify minimal adjustment set for the causal effect of treatment on outcome in the causal graph specified in the
    given dot file.
    :param dot_file_path: path to causal graph dot file as a string.
    :param treatment: name of treatment variable as a string.
    :param outcome: name of outcome variable as a string.
    :return: a list of adjustment variables as strings.
    """
    _install_r_packages(["devtools", "dagitty", "glue"])
    dagitty_dag_str = _dot_to_dagitty_dag(dot_file_path)
    r_identification_fn = """
       R_identification <- function(dag_str, treatment, outcome){
           library(devtools)
           library(dagitty)
           library(glue)
           dag <- dagitty(dag_str)
           adjustment_sets <- (adjustmentSets(dag, treatment, outcome, type="minimal", effect="total"))
           min_adjustment_set <- adjustment_sets[which.min(lengths(adjustment_sets))]
           return(min_adjustment_set)
       }
       """
    r_pkg = STAP(r_identification_fn, "r_pkg")
    min_adjustment_set = r_pkg.R_identification(dagitty_dag_str, treatment, outcome)
    min_adjustment_list = np.array(min_adjustment_set).tolist()
    if len(np.array(min_adjustment_set).tolist()) == 0:
        return []
    return min_adjustment_list[0]


def _dot_to_dagitty_dag(dot_file_path):
    """
    Convert a standard dot digraph to a dagitty-compatible dag.
    :param dot_file_path: path to causal graph dot file as a string.
    :return: string representation of dagitty-compatible dag.
    """
    dot_graph = pygraphviz.AGraph(dot_file_path)
    dot_string = (
        "dag {" + "\n".join([f"{s1} -> {s2};" for s1, s2 in dot_graph.edges()]) + "}"
    )
    dag_string = dot_string.replace("digraph", "dag")
    return dag_string


def _install_r_packages(package_names):
    """
    Download and install a given list of R packages from CRAN.
    :param package_names: a list of package names to install as strings.
    """
    utils = importr("utils")
    utils.chooseCRANmirror(ind=1)
    packages_to_install = [pkg for pkg in package_names if not isinstalled(pkg)]
    if len(packages_to_install) > 0:
        # The silent package install requires user input, which Behave captures, so it appears to hang
        # This makes the failure "noisy", so the user knows what's going on
        raise ValueError(
            f"Please install R packages f{packages_to_install} and try again"
        )
