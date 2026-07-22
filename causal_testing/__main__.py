"""This module contains the main entrypoint functionality to the Causal Testing Framework."""

import argparse
import json
import logging
from enum import Enum
from importlib.metadata import entry_points
from typing import Optional, Sequence

import networkx as nx
import pandas as pd

from causal_testing.causal_testing_framework import CausalTestingFramework, read_dataframe
from causal_testing.specification.causal_dag import CausalDAG

logger = logging.getLogger(__name__)


class Command(Enum):
    """
    Enum for supported CTF commands.
    """

    TEST = "test"
    GENERATE = "generate"
    DISCOVER = "discover"
    EVALUATE = "evaluate"


def setup_logging(level: str) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    main_parser = argparse.ArgumentParser(
        add_help=True,
        description="Causal Testing Framework - "
        "A causal inference-driven framework for functional black-box testing of complex software.",
    )

    subparsers = main_parser.add_subparsers(
        help="The action you want to run - call `causal_testing {action} -h` for further details", dest="command"
    )

    # Generation
    parser_generate = subparsers.add_parser(Command.GENERATE.value, help="Generate causal tests from a DAG")
    parser_generate.add_argument("-D", "--dag-path", help="Path to the DAG file (.dot)", required=True)
    parser_generate.add_argument("-o", "--output", help="Path for output file (.json)", required=True)
    parser_generate.add_argument(
        "-i", "--ignore-cycles", help="Ignore cycles in DAG", action="store_true", default=False
    )
    parser_generate.add_argument(
        "--threads", "-t", type=int, help="The number of parallel threads to use.", required=False, default=0
    )

    # Testing
    parser_test = subparsers.add_parser(Command.TEST.value, help="Run causal tests")
    parser_test.add_argument("-D", "--dag-path", help="Path to the DAG file (.dot)", required=True)
    parser_test.add_argument("-o", "--output", help="Path for output file (.json)", required=True)
    parser_test.add_argument("-i", "--ignore-cycles", help="Ignore cycles in DAG", action="store_true", default=False)
    parser_test.add_argument("-t", "--test-config", help="Path to test configuration file (.json)", required=True)
    parser_test.add_argument("-q", "--query", help="Query string to filter data (e.g. 'age > 18')", type=str)
    parser_test.add_argument(
        "-A", "--adequacy", help="Calculate causal test adequacy for each test case", action="store_true", default=False
    )
    parser_test.add_argument(
        "-b",
        "--adequacy-bootstrap-size",
        dest="bootstrap_size",
        help="Number of bootstrap samples for causal test adequacy. Defaults to 100",
        type=int,
    )
    parser_test.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Do not crash on error. If set to true, errors are recorded as test results.",
        default=False,
    )

    # DAG evaluation
    parser_evaluate = subparsers.add_parser(
        Command.EVALUATE.value, help="Evaluate how well a causal DAG fits a dataset"
    )
    parser_evaluate.add_argument("-D", "--dag-path", help="Path to the DAG file (.dot)", required=True)
    parser_evaluate.add_argument("-o", "--output", help="Path for output file (.csv)", required=True)
    parser_evaluate.add_argument(
        "-i", "--ignore-cycles", help="Ignore cycles in DAG", action="store_true", default=False
    )
    parser_evaluate.add_argument("-q", "--query", help="Query string to filter data (e.g. 'age > 18')", type=str)
    parser_evaluate.add_argument(
        "-b",
        "--adequacy-bootstrap-size",
        dest="bootstrap_size",
        help="Number of bootstrap samples for causal test adequacy. Defaults to 100",
        type=int,
        default=100,
    )
    parser_evaluate.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Do not crash on error. If set to true, errors are recorded as test results.",
        default=False,
    )

    # Discovery
    parser_discover = subparsers.add_parser(Command.DISCOVER.value, help="Discover causal structures from data")
    parser_discover.add_argument(
        "-t",
        "--technique",
        help="The name of the technique to use. Currently supported are 'HillClimberDiscovery' and 'NSGADiscovery'",
        required=True,
    )
    parser_discover.add_argument(
        "-V",
        "--variables",
        help="The subset of variables from the data to consider. Defaults to all.",
        nargs="*",
        default=[],
    )
    parser_discover.add_argument("-o", "--output", help="Path for output DAG file (.dot)", required=True)
    parser_discover.add_argument(
        "-i", "--include-edges", help="Path to file containing edges to include", required=False
    )
    parser_discover.add_argument(
        "-e", "--exclude-edges", help="Path to file containing edges to exclude", required=False
    )
    parser_discover.add_argument(
        "--technique-kwargs",
        help="Keywords for the discovery technique. These should be specified as `arg1=value1 arg2=value2...`.",
        nargs="*",
        default=[],
    )

    for parser in [parser_generate, parser_discover, parser_test, parser_evaluate]:
        parser.add_argument(
            "-l",
            "--log_level",
            default="WARNING",
            type=str.upper,
            choices=["NONE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set the logging level (default: WARNING).",
        )
        parser.add_argument(
            "-a",
            "--alpha",
            help=(
                "The significance level of the confidence intervals used to determine causality. "
                "This should be a value between 0 and 1. Defaults to 0.05 for 95%% confidence intervals."
            ),
            default=0.05,
        )
        parser.add_argument("-d", "--data-paths", help="Paths to data files (.csv)", nargs="+", required=True)

    args = main_parser.parse_args(args)

    # Assume the user wants test adequacy if they're setting bootstrap_size
    if getattr(args, "bootstrap_size", None) is not None:
        args.adequacy = True
    if getattr(args, "adequacy", False) and getattr(args, "bootstrap_size", None) is None:
        # Need this here rather than a default value because otherwise the above always sets adequacy to True
        args.bootstrap_size = 100

    args.command = Command(args.command)
    return args


def main() -> None:
    """

    Main entry point for the Causal Testing Framework

    """

    # Parse arguments
    args = parse_args()
    # Setup logging
    setup_logging(args.log_level)

    match args.command:
        case Command.GENERATE:
            logging.info("Generating causal tests")
            df = pd.concat(read_dataframe(path) for path in args.data_paths)
            causal_dag = CausalDAG(args.dag_path, ignore_cycles=args.ignore_cycles, datatypes=df.dtypes)
            causal_tests = causal_dag.generate_causal_tests(
                threads=args.threads,
                skip=False,
            )
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({"tests": [test.to_dict() for test in causal_tests]}, f)
            logging.info("Causal test generation completed successfully.")

        case Command.DISCOVER:
            discover_map = {ff.name: ff for ff in entry_points(group="discovery")}
            if args.technique not in discover_map:
                raise ValueError(
                    f"Unsupported technique {args.technique}. Supported: {sorted(discover_map)}. "
                    "If you have implemented a custom technique, you will need to add this to your entrypoints via "
                    "your pyproject.toml file."
                )
            kwargs = {}
            for argument in args.technique_kwargs:
                split = argument.split("=")
                if len(split) != 2:
                    raise ValueError(f"Malformed argument {argument}. Should be specified as `arg_name=arg_value`")
                kwargs[split[0]] = split[1]

            logging.info("Discovering causal structure")
            # Need to reset index to allow for multiple files having the same index (i.e. starting at zero).
            # Otherwise you end up with duplicate indices, which causes problems further down the line
            df = pd.concat([read_dataframe(path) for path in args.data_paths]).reset_index()
            if args.variables:
                df = df[args.variables]
            # Drop unnamed columns
            unnamed_columns = [c for c in df.columns if c.startswith("Unnamed: ")]
            if unnamed_columns:
                logger.warning(f"Dropping unnamed columns: {unnamed_columns}")
            df = df.drop(unnamed_columns)

            discover_class = discover_map[args.technique].load()
            discover = discover_class(
                df=df,
                exclude_edges=(
                    list(nx.nx_pydot.read_dot(args.exclude_edges).edges()) if args.exclude_edges is not None else []
                ),
                include_edges=(
                    list(nx.nx_pydot.read_dot(args.include_edges).edges()) if args.include_edges is not None else []
                ),
                alpha=args.alpha,
                **kwargs,
            )
            evolved_dag = discover.discover()
            discover.write_dot(evolved_dag, args.output)
            logging.info("Causal structure discovery completed successfully.")
        case Command.TEST:
            # Create and setup framework
            framework = CausalTestingFramework()

            framework.setup(
                dag_path=args.dag_path,
                data_paths=args.data_paths,
                test_cases_path=args.test_config,
                query=args.query,
                ignore_cycles=args.ignore_cycles,
            )

            logging.info("Running tests")
            framework.run_tests(silent=args.silent, adequacy=args.adequacy, bootstrap_size=args.bootstrap_size)
            framework.save_results(args.output)

            logging.info("Causal testing completed successfully.")
        case Command.EVALUATE:
            # Create and setup framework
            framework = CausalTestingFramework()

            framework.setup(
                dag_path=args.dag_path,
                data_paths=args.data_paths,
                test_cases_path=args.test_config,
                query=args.query,
                ignore_cycles=args.ignore_cycles,
            )

            logging.info("Running tests on entire dataset")
            results = framework.evaluate_dag(alpha=args.alpha, bootstrap_size=args.bootstrap_size)
            logging.info("Causal testing completed successfully.")
            logging.info("Running tests on bootstrap samples")
            results.to_csv(args.output)


if __name__ == "__main__":
    main()
