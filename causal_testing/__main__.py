"""This module contains the main entrypoint functionality to the Causal Testing Framework."""

import json
import logging
import os
import tempfile
from importlib.metadata import entry_points
from pathlib import Path

import networkx as nx
import pandas as pd

from causal_testing.testing.metamorphic_relation import generate_causal_tests

from .main import CausalTestingFramework, CausalTestingPaths, Command, parse_args, setup_logging


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
            generate_causal_tests(
                args.dag_path,
                args.output,
                args.ignore_cycles,
                args.threads,
                effect_type=args.effect_type,
                estimate_type=args.estimate_type,
                estimator=args.estimator,
                skip=False,
            )
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
            df = pd.concat([pd.read_csv(path) for path in args.data_paths]).reset_index()
            if args.variables:
                df = df[args.variables]

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
            # Create paths object
            paths = CausalTestingPaths(
                dag_path=args.dag_path,
                data_paths=args.data_paths,
                test_config_path=args.test_config,
                output_path=args.output,
            )

            # Create and setup framework
            framework = CausalTestingFramework(paths, ignore_cycles=args.ignore_cycles, query=args.query)
            framework.setup()

            # Load and run tests
            framework.load_tests()

            logging.info("Running tests in regular mode")
            results = framework.run_tests(
                silent=args.silent, adequacy=args.adequacy, bootstrap_size=args.bootstrap_size
            )
            framework.save_results(results)

            logging.info("Causal testing completed successfully.")


if __name__ == "__main__":
    main()
