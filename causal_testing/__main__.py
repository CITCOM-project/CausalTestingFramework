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
            exclude_edges = list(nx.nx_pydot.read_dot(args.exclude_edges).edges()) if args.exclude_edges is not None else []
            
            if args.context:
                dfs = []
                for i, path in enumerate(args.data_paths):
                    temp_df = pd.read_csv(path)
                    temp_df['file_index'] = i
                    dfs.append(temp_df)

                df = pd.concat(dfs, ignore_index=True)
                
                if args.variables:
                    df = df[list(set(args.variables + ['file_index']))]

                exclude_edges.append('".*" -> file_index')
            else:
                df = pd.concat((pd.read_csv(path) for path in args.data_paths), ignore_index=True)
                
                if args.variables:
                    df = df[args.variables]

            discover_class = discover_map[args.technique].load()
            discover = discover_class(
                df=df,
                exclude_edges=exclude_edges,
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

            if args.batch_size > 0:
                logging.info(f"Running tests in batches of size {args.batch_size}")
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_files = []
                    for i, results in enumerate(
                        framework.run_tests_in_batches(
                            batch_size=args.batch_size,
                            silent=args.silent,
                            adequacy=args.adequacy,
                            bootstrap_size=args.bootstrap_size,
                        )
                    ):
                        temp_file_path = os.path.join(tmpdir, f"output_{i}.json")
                        framework.save_results(results, temp_file_path)
                        output_files.append(temp_file_path)
                        del results

                    # Now stitch the results together from the temporary files
                    all_results = []
                    for file_path in output_files:
                        with open(file_path, "r", encoding="utf-8") as f:
                            all_results.extend(json.load(f))

                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(args.output, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, indent=4)
            else:
                logging.info("Running tests in regular mode")
                results = framework.run_tests(
                    silent=args.silent, adequacy=args.adequacy, bootstrap_size=args.bootstrap_size
                )
                framework.save_results(results)

            logging.info("Causal testing completed successfully.")


if __name__ == "__main__":
    main()
