from pathlib import Path

import holoviews as hv
import pandas as pd
import pytest

from causal_testing.causal_testing_framework import CausalTestingFramework
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_result import TestOutcome
from causal_testing.visualisation.visualisation_plotter import VisualisationPlotter


@pytest.fixture(name="dag")
def _dag() -> CausalTestingFramework:
    dag = CausalDAG()
    dag.add_edges_from(
        [
            ("variants", "beta"),
            ("beta", "cum_infections"),
            ("location", "variants"),
            ("location", "avg_age"),
            ("location", "contacts"),
            ("contacts", "cum_infections"),
            ("avg_age", "cum_infections"),
        ]
    )
    return dag


@pytest.fixture(name="plotter")
def _plotter(dag) -> VisualisationPlotter:
    return VisualisationPlotter(
        dag=dag, df=pd.read_csv(Path(__file__).parent.parent / "resources" / "data" / "doubling_beta_results.csv")
    )


def test_results_dag(dag, plotter):
    """
    The result DAG should have the same causal edges as the original, plus dashed edges for failing independence
    tests. Passing edges should be green. Failing edges should be red. Inestimable edges should be yellow.
    """
    results_dag = plotter.results_dag()

    for _, test in plotter.df.iterrows():
        edge_data = results_dag.get_edge_data(test["estimator.treatment_variable"], test["estimator.outcome_variable"])
        if test["result.outcome"] == "FAIL":
            assert edge_data is not None, "Expected edge for failing test."
            assert (
                edge_data.get("color") == plotter.colour_map[TestOutcome.FAIL]
            ), "Expected failing tests to map to red."
        elif test["result.outcome"] == "INESTIMABLE":
            assert edge_data is not None, "Expected edge for inestimable test."
            assert (
                edge_data.get("color") == plotter.colour_map[TestOutcome.INESTIMABLE]
            ), "Expected inestimable tests to map to yellow."
        elif (test["estimator.treatment_variable"], test["estimator.outcome_variable"]) in dag.edges:
            assert edge_data is not None, "Expected edge for passing causal test."
            assert (
                edge_data.get("color") == plotter.colour_map[TestOutcome.PASS]
            ), "Expected passing tests to map to green."
        else:
            assert edge_data is None, "Passing independence tests should not have edges in the result DAG."


def test_interactive_results_dag(dag, plotter):
    """
    The result DAG should have the same causal edges as the original, plus dashed edges for failing independence
    tests. Passing edges should be green. Failing edges should be red. Inestimable edges should be yellow.
    """
    interactive_dag = plotter.interactive_results_dag().Graph.I
    hv.renderer("bokeh").get_plot(interactive_dag)

    edges_df = interactive_dag.dframe()

    src_col, dst_col = interactive_dag.kdims[0].name, interactive_dag.kdims[1].name
    edge_map = {(row[src_col], row[dst_col]): row.to_dict() for _, row in edges_df.iterrows()}

    for _, test in plotter.df.iterrows():
        edge_data = edge_map.get((test["estimator.treatment_variable"], test["estimator.outcome_variable"]))

        if test["result.outcome"] == "FAIL":
            assert edge_data is not None, "Expected edge for failing test."
            assert (
                edge_data.get("color") == plotter.colour_map[TestOutcome.FAIL]
            ), "Expected failing tests to map to red."
        elif test["result.outcome"] == "INESTIMABLE":
            assert edge_data is not None, "Expected edge for inestimable test."
            assert (
                edge_data.get("color") == plotter.colour_map[TestOutcome.INESTIMABLE]
            ), "Expected inestimable tests to map to yellow."
        elif (test["estimator.treatment_variable"], test["estimator.outcome_variable"]) in dag.edges:
            assert edge_data is not None, "Expected edge for passing causal test."
            assert (
                edge_data.get("color") == plotter.colour_map[TestOutcome.PASS]
            ), "Expected passing tests to map to green."
        else:
            assert edge_data is None, "Passing independence tests should not have edges in the result DAG."


def test_outcome_adjacency(dag, plotter):
    heatmap = plotter.outcome_adjacency()
    # Check correct axes
    assert [kdim.name for kdim in heatmap.kdims] == [
        "treatment_variable_inx",
        "outcome_variable_inx",
    ], f"Unexpected kdims {heatmap.kdims}"
    assert "result.outcome" in [vdim.name for vdim in heatmap.vdims], f"'result.outcome' not found in {heatmap.vdims}"

    bokeh_fig = hv.render(heatmap, backend="bokeh")
    glyph_renderer = [r for r in bokeh_fig.renderers if hasattr(r, "glyph")][0]
    fill_color_transform = glyph_renderer.glyph.fill_color
    color_mapper = fill_color_transform.transform

    df = pd.DataFrame(glyph_renderer.data_source.data)
    category_color_map = dict(zip(color_mapper.factors, color_mapper.palette))
    df["rendered_color"] = df[fill_color_transform.field].apply(lambda c: category_color_map.get(c))

    test_df = plotter.df.copy()
    test_df["expected_color"] = [
        plotter.colour_map[getattr(TestOutcome, outcome)] for outcome in test_df["result.outcome"]
    ]

    merged_df = pd.merge(
        df,
        test_df[["estimator.treatment_variable", "estimator.outcome_variable", "expected_color"]],
        left_on=["estimator_full_stop_treatment_variable", "estimator_full_stop_outcome_variable"],
        right_on=["estimator.treatment_variable", "estimator.outcome_variable"],
        how="inner",  # or 'left', 'right', 'outer'
    )
    assert (merged_df["expected_color"] == merged_df["rendered_color"]).all()


@pytest.mark.parametrize(
    "method_name, expected_vdim",
    [
        ("dag_adequacy_heatmap", "result.adequacy.passing"),
        ("data_adequacy_heatmap", "result.adequacy.kurtosis"),
    ],
)
def test_adequacy_heatmap(plotter, method_name, expected_vdim):
    """
    Every test with an adequacy value should have a non-grey square.
    """
    # Call the plotter method dynamically
    heatmap = getattr(plotter, method_name)()
    hv.renderer("bokeh").get_plot(heatmap)

    # Check correct axes
    assert [kdim.name for kdim in heatmap.kdims] == [
        "treatment_variable_inx",
        "outcome_variable_inx",
    ], f"Unexpected kdims {heatmap.kdims}"
    assert expected_vdim in [
        vdim.name for vdim in heatmap.vdims
    ], f"'{expected_vdim}' not found in vdims {heatmap.vdims}"

    # Check all tests with results have a square
    # It's very difficult to test that the squares are the right colours or in the right place
    for _, test in plotter.df.iterrows():
        if test["result.outcome"] != "INESTIMABLE":
            assert not heatmap.data[
                (heatmap.data["estimator.treatment_variable"] == test["estimator.treatment_variable"])
                & (heatmap.data["estimator.outcome_variable"] == test["estimator.outcome_variable"])
            ].empty, (
                f"Test with treatment '{test['estimator.treatment_variable']}' and "
                "outcome='{test.estimator.outcome_variable}' should be included."
            )
