import time
from pathlib import Path

import pandas as pd
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
    Test that the dashboard loads the data correctly.
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
    Test that the dashboard loads the data correctly.
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


def test_run_tests(dag_path, data_path):
    """
    Test that the dashboard loads the data correctly.
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


def test_unexecuted_test_suite_stats_text(dag_path, test_path):
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


@pytest.fixture(name="dashboard_server")
def _dashboard_server():
    server = serve_dashboard(threaded=True)

    # Wait up to 5 seconds for the server to be ready
    url = "http://localhost:5006/"
    timeout = 5
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # Quick ping to check if the port is listening
            requests.get(url, timeout=1)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
    else:
        server.stop()
        pytest.fail(f"Server failed to start on {url} within {timeout} seconds")

    yield server
    server.stop()


def test_dashboard_spins_up_and_loads():
    url = "http://localhost:5006/"
    response = requests.get(url, timeout=5)

    assert response.status_code == 200
