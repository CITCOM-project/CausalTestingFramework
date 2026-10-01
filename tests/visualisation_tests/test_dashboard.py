import json
import re
import time
from pathlib import Path

import pandas as pd
import panel as pn
import pytest
import requests
from bs4 import BeautifulSoup

from causal_testing.causal_testing_framework import CausalTestingFramework
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.visualisation.testing_dashboard import Dashboard, serve_dashboard


@pytest.fixture(name="dag_path")
def _dag_path():
    return Path(__file__).parent.parent.parent / "examples" / "poisson-line-process" / "dag.dot"


@pytest.fixture(name="data_path")
def _data_path():
    return (
        Path(__file__).parent.parent.parent
        / "examples"
        / "poisson-line-process"
        / "data"
        / "random"
        / "data_random_1000.csv"
    )


@pytest.fixture(name="test_path")
def _test_path():
    return Path(__file__).parent.parent.parent / "examples" / "poisson-line-process" / "causal_tests.json"


@pytest.fixture(name="completed_test_path")
def _completed_test_path():
    return Path(__file__).parent.parent / "resources" / "data" / "poisson_line_tests.json"


def test_dashboard_initialization():
    """Test that the dashboard initializes components with expected default states."""
    dashboard = Dashboard()
    assert dashboard.adequacy is False
    assert dashboard.run_tests.disabled is True
    assert dashboard.generate_tests.disabled is True


def test_load_dag_file(dag_path):
    """
    Test that the dashboard loads the DAG correctly.
    """
    dashboard = Dashboard()
    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    expected_dag = CausalDAG(str(dag_path))

    assert (
        dashboard.ctf.dag.nodes == expected_dag.nodes
    ), f"Mismatched nodes. Expected {CausalDAG(dag_path).nodes} but was {dashboard.ctf.dag.nodes}"
    assert (
        dashboard.ctf.dag.edges == expected_dag.edges
    ), f"Mismatched edges. Expected {CausalDAG(dag_path).edges} but was {dashboard.ctf.dag.edges}"


def test_load_data_file(data_path):
    """
    Test that the dashboard loads the data correctly.
    """
    dashboard = Dashboard()

    with open(data_path, encoding="utf-8") as f:
        dashboard.data_file_input.filename = str(data_path)
        dashboard.data_file_input.value = f.read().encode()

    pd.testing.assert_frame_equal(pd.read_csv(data_path), dashboard.ctf.df)


def test_load_test_file(dag_path, test_path):
    """
    Test that the dashboard loads tests the same as the native CTF.
    """

    ctf = CausalTestingFramework()
    ctf.load_dag(str(dag_path))
    ctf.load_test_cases_from_json(test_path)

    dashboard = Dashboard()

    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    with open(test_path, encoding="utf-8") as f:
        dashboard.test_file_input.value = f.read().encode()

    assert [test.to_dict() for test in ctf.test_cases] == [test.to_dict() for test in dashboard.ctf.test_cases]


def test_generate_tests(dag_path, data_path):
    """
    Test that the dashboard generates tests the same as the native CTF.
    """

    ctf = CausalTestingFramework()
    ctf.load_dag(str(dag_path))
    ctf.load_data([data_path])
    ctf.generate_causal_tests()

    dashboard = Dashboard()

    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    with open(data_path, encoding="utf-8") as f:
        dashboard.data_file_input.filename = str(data_path)
        dashboard.data_file_input.value = f.read().encode()

    dashboard.generate_tests.value = True

    assert [test.to_dict() for test in ctf.test_cases] == [test.to_dict() for test in dashboard.ctf.test_cases]


def test_generate_tests_no_datatypes(dag_path):
    """
    Test that the dashboard generates a suitable error message when we try to generate tests without datatype info.
    """

    dashboard = Dashboard()

    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    dashboard.generate_tests.value = True
    assert pn.state.notifications.notifications[0].message == "No datatype specified for num_lines_abs."


def test_run_tests(dag_path, data_path):
    """
    Test that the dashboard runs tests the same as the native CTF.
    """

    ctf = CausalTestingFramework()
    ctf.load_dag(str(dag_path))
    ctf.load_data([data_path])
    ctf.generate_causal_tests()
    ctf.run_tests()

    dashboard = Dashboard()

    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    with open(data_path, encoding="utf-8") as f:
        dashboard.data_file_input.filename = str(data_path)
        dashboard.data_file_input.value = f.read().encode()

    dashboard.generate_tests.value = True
    dashboard.run_tests.value = True

    assert [test.to_dict() for test in ctf.test_cases] == [test.to_dict() for test in dashboard.ctf.test_cases]


def test_completed_test_suite_stats_text(dag_path, completed_test_path):
    """
    Test that the executed tests lead to the correct summary.
    """
    dashboard = Dashboard()

    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    with open(completed_test_path, encoding="utf-8") as f:
        dashboard.test_file_input.value = f.read().encode()

    extracted_texts = [
        BeautifulSoup(pane.object, "html.parser").get_text(separator=" ", strip=True)
        for pane in dashboard.test_suite_stats().objects
    ]

    expected = [
        "Test Cases 26",
        "Passing tests 18 (69.2%)",
        "Failing tests 7 (26.9%)",
        "Inestimable tests 1 (3.8%)",
    ]

    assert extracted_texts == expected


def test_suite_stats_colour_only_failing(dag_path, completed_test_path):
    """
    Test that the executed tests lead to the correct summary.
    """
    dashboard = Dashboard()

    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    with open(completed_test_path, encoding="utf-8") as f:
        tests = json.load(f)
    only_failing = [test for test in tests if test["result"]["outcome"] == "FAIL"]
    dashboard.test_file_input.value = json.dumps(only_failing).encode()

    colours = [re.search("color: (\w+)", pane.object).group(1) for pane in dashboard.test_suite_stats().objects[1:]]

    expected_colours = ["red", "red", "green"]

    assert colours == expected_colours


def test_suite_stats_colour_only_passing(dag_path, completed_test_path):
    """
    Test that the executed tests lead to the correct summary.
    """
    dashboard = Dashboard()

    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    with open(completed_test_path, encoding="utf-8") as f:
        tests = json.load(f)
    only_passing = [test for test in tests if test["result"]["outcome"] == "PASS"]
    dashboard.test_file_input.value = json.dumps(only_passing).encode()

    colours = [re.search("color: (\w+)", pane.object).group(1) for pane in dashboard.test_suite_stats().objects[1:]]

    expected_colours = ["green", "green", "green"]

    assert colours == expected_colours


def test_unexecuted_test_suite_stats_text(dag_path, test_path):
    """
    Test that the non-executed tests lead to the correct summary (i.e. just the number of tests).
    """
    dashboard = Dashboard()

    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    with open(test_path, encoding="utf-8") as f:
        dashboard.test_file_input.value = f.read().encode()

    extracted_texts = [
        BeautifulSoup(pane.object, "html.parser").get_text(separator=" ", strip=True)
        for pane in dashboard.test_suite_stats().objects
    ]

    expected = [
        "Test Cases 26",
        "Passing tests -",
        "Failing tests -",
        "Inestimable tests -",
    ]

    assert extracted_texts == expected


def test_update_tests(dag_path, test_path):
    """
    Test that we can update test cases.
    """
    dashboard = Dashboard()

    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    with open(test_path, encoding="utf-8") as f:
        tests = json.load(f)
    dashboard.test_file_input.value = json.dumps(tests).encode()

    dashboard.test_editor.value = json.dumps([tests[1]])
    dashboard.update_tests.value = True

    assert [test.to_dict() for test in dashboard.ctf.test_cases] == [tests[1]]


def test_download_tests(dag_path, test_path):
    """
    Test that we can download test cases.
    """
    dashboard = Dashboard()

    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    with open(test_path, encoding="utf-8") as f:
        tests_string = f.read()

    dashboard.test_file_input.value = tests_string.encode()
    dashboard.test_editor_panel()
    downloaded_tests = dashboard.download_tests.callback().getvalue().decode("utf-8")

    assert json.loads(downloaded_tests) == json.loads(tests_string)


def test_build_template_html_structure(dag_path, completed_test_path):
    """
    Test that we have test outcomes and test adequacy on the main tab and a code editor in the test editor tab.

    NOTE: This is a very basic test.
          Ideally we'd be using playwright, but that seems like overkill for such a basic dashboard.
    """
    dashboard = Dashboard()

    with open(dag_path, encoding="utf-8") as f:
        dashboard.dag_file_input.value = f.read().encode()

    with open(completed_test_path, encoding="utf-8") as f:
        dashboard.test_file_input.value = f.read().encode()

    template = dashboard.build_template()

    template = dashboard.build_template()
    [main_tab, test_editor_tab] = template.main[0]  # pn.Tabs layout

    main_content = main_tab.object()
    markdown_panes = main_content.select(pn.pane.Markdown)

    assert any("Test Outcomes" in pane.object for pane in markdown_panes), "Expected to see a section for test outcomes"
    assert any("Test Adequacy" in pane.object for pane in markdown_panes), "Expected to see a section for test adequacy"

    test_editor_content = test_editor_tab.object()
    assert len(test_editor_content.select(pn.widgets.CodeEditor)) == 1, "Expected to see a code editor object"


@pytest.fixture(name="dashboard_server")
def _dashboard_server():
    """
    Configurable dashboard server.
    """
    servers = []

    def _start_server(port=5006, timeout=5, **kwargs):
        server = serve_dashboard(port=port, threaded=True, **kwargs)
        servers.append(server)

        url = f"http://localhost:{port}/"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                requests.get(url, timeout=timeout)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)
        else:
            server.stop()
            pytest.fail(f"Server failed to start on {url} within {timeout} seconds")

        return server

    yield _start_server

    for server in servers:
        server.stop()


def test_dashboard_spins_up_and_loads(dashboard_server):
    """
    Test that we can connect to an active dashboard instance.
    """
    port = 5006
    dashboard_server(port)
    response = requests.get(f"http://localhost:{port}/", timeout=5)

    assert response.status_code == 200
