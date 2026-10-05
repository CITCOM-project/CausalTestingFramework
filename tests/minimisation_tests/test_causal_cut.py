import random

import pandas as pd
import pytest

from causal_testing.minimisation.causal_cut import CausalCut
from causal_testing.specification.causal_dag import CausalDAG

TOTAL_TIME = 100


def dummy_tank(
    state: dict[str, bool], interventions: list[tuple[int, str, int]] = 0, tank_level=0, tank_size=10
) -> pd.DataFrame:
    """
    Simple tank system to generate test data.
    """
    data = []
    for t in range(TOTAL_TIME):
        if tank_level >= 0.7 * tank_size:
            state["pump_on"] = 0
        if tank_level >= 0.9 * tank_size:
            state["valve_open"] = 1
        if tank_level <= 0.3 * tank_size:
            state["valve_open"] = 0
        if tank_level <= 0.2 * tank_size:
            state["pump_on"] = 1
        current_interventions = [(time, var, val) for time, var, val in interventions if time == t]
        state |= {var: val for _, var, val in current_interventions}
        if tank_level > 0.1:
            if state.get("valve_open"):
                tank_level -= 1
            else:
                # Leaky!
                tank_level -= 0.1
        if state.get("pump_on"):
            tank_level += 3
        data.append(dict(state) | {"time": t, "tank_level": tank_level, "tank_size": tank_size})
        if tank_level > tank_size:
            break
    return pd.DataFrame(data)


def collect_data():
    """
    Use this to collect runs of the simple tank system.
    """
    random.seed(1)
    runs = []
    num_runs = 100
    for i in range(num_runs):
        runs.append(
            dummy_tank(
                {"pump_on": random.choice([0, 1]), "valve_open": random.choice([0, 1])},
                tank_level=random.choice([0, 7]),
                tank_size=random.randint(8, 12),
                interventions=[[t, "pump_on", random.choice([1, 0])] for t in range(random.randint(6, 11))],
            ).assign(id=i)
        )
    return pd.concat(runs)


@pytest.fixture(name="causal_dag")
def _causal_dag():
    dag = CausalDAG(ignore_cycles=True)
    dag.add_edges_from(
        [
            ("pump_on", "tank_level"),
            ("tank_level", "pump_on"),
            ("valve_open", "tank_level"),
            ("tank_level", "valve_open"),
        ]
    )
    return dag


@pytest.fixture(name="dummy_data")
def _dummy_data():
    return pd.read_csv("tests/resources/data/dummy_pump_data.csv", index_col=0)


def test_estimate_intervention_causality(causal_dag, dummy_data):
    """
    Test that interventions can be estimated.
    """
    causal_cut = CausalCut(
        dag=causal_dag,
        df=dummy_data,
        safe_ranges=pd.DataFrame({"tank_level": {"low": 0.2, "high": 9}}).T,
        reproduce_fault=dummy_tank,
    )
    tests = causal_cut.estimate_intervention_causality(
        interventions=[[t, "pump_on", 1] for t in range(1, 5)],
        outcome_variable="tank_level",
        total_time=TOTAL_TIME,
        background_confounders=["tank_size"],
    )
    assert {intervention: test.result.passed for intervention, test in tests.items()} == {
        (1, "pump_on", 1): True,
        (2, "pump_on", 1): False,
        (3, "pump_on", 1): False,
        (4, "pump_on", 1): False,
    }


def test_estimate_intervention_causality_missing_nodes(causal_dag, dummy_data):
    """
    Test that interventions can be estimated.
    """
    causal_cut = CausalCut(
        dag=causal_dag,
        df=dummy_data,
        safe_ranges=pd.DataFrame({"tank_level": {"low": 0.2, "high": 9}}).T,
        reproduce_fault=dummy_tank,
    )
    with pytest.raises(ValueError) as e:
        causal_cut.estimate_intervention_causality(
            interventions=[[t, "invalid", 1] for t in range(1, 5)],
            outcome_variable="tank_level",
            total_time=TOTAL_TIME,
            background_confounders=["tank_size"],
        )
        assert e.exception == "DAG missing nodes [invalid] for control_strategy."


def test_estimate_intervention_causality_missing_neighbours(causal_dag, dummy_data):
    """
    Test that interventions can be estimated.
    """
    causal_dag.add_node("invalid")
    causal_cut = CausalCut(
        dag=causal_dag,
        df=dummy_data,
        safe_ranges=pd.DataFrame({"tank_level": {"low": 0.2, "high": 9}}).T,
        reproduce_fault=dummy_tank,
    )
    with pytest.raises(ValueError) as e:
        causal_cut.estimate_intervention_causality(
            interventions=[[t, "invalid", 1] for t in range(1, 5)],
            outcome_variable="tank_level",
            total_time=TOTAL_TIME,
            background_confounders=["tank_size"],
        )
        assert e.exception == "No neighbours for node invalid."


def test_minimise_test_case(causal_dag, dummy_data):
    """
    Test that tests can be minimised.
    """
    tank_size = 10
    causal_cut = CausalCut(
        dag=causal_dag,
        df=dummy_data,
        safe_ranges=pd.DataFrame({"tank_level": {"low": 0.2, "high": 9}}).T,
        reproduce_fault=lambda interventions: (
            dummy_tank(state={"pump_on": 0, "valve_open": 0}, interventions=interventions, tank_size=tank_size)[
                "tank_level"
            ]
            > tank_size
        ).any(),
    )
    minimised_test = causal_cut.minimise_test(
        interventions=[[t, "pump_on", 1] for t in range(1, 5)],
        outcome_variable="tank_level",
        total_time=TOTAL_TIME,
        background_confounders=["tank_size"],
    )
    assert minimised_test == [(1, "pump_on", 1), (4, "pump_on", 1)]


def test_minimise_test_case_greedy_prune(causal_dag, dummy_data):
    """
    Test that tests can be minimised and pruned.
    """
    tank_size = 10
    causal_cut = CausalCut(
        dag=causal_dag,
        df=dummy_data,
        safe_ranges=pd.DataFrame({"tank_level": {"low": 0.2, "high": 9}}).T,
        reproduce_fault=lambda interventions: (
            dummy_tank(state={"pump_on": 0, "valve_open": 0}, interventions=interventions, tank_size=tank_size)[
                "tank_level"
            ]
            > tank_size
        ).any(),
    )
    minimised_test = causal_cut.minimise_test(
        interventions=[[t, "pump_on", 1] for t in range(1, 5)],
        outcome_variable="tank_level",
        total_time=TOTAL_TIME,
        background_confounders=["tank_size"],
        greedy_minimise=True,
    )
    assert minimised_test == [(4, "pump_on", 1)]


def test_minimise_test_case_fallback(causal_dag, dummy_data):
    """
    Test that tests can be minimised even when no tests can be estimated.
    """
    tank_size = 10
    causal_cut = CausalCut(
        dag=causal_dag,
        df=dummy_data.loc[dummy_data["id"] == 1],
        safe_ranges=pd.DataFrame({"tank_level": {"low": 0.2, "high": 9}}).T,
        reproduce_fault=lambda interventions: (
            dummy_tank(state={"pump_on": 0, "valve_open": 0}, interventions=interventions, tank_size=tank_size)[
                "tank_level"
            ]
            > tank_size
        ).any(),
    )
    minimised_test = causal_cut.minimise_test(
        interventions=[[t, "pump_on", 1] for t in range(1, 5)],
        outcome_variable="tank_level",
        total_time=TOTAL_TIME,
        background_confounders=["tank_size"],
        greedy_minimise=True,
    )
    assert minimised_test == [(4, "pump_on", 1)]
