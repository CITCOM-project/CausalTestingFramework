"""This module contains the main entrypoint functionality to the Causal Testing Framework."""

import logging
import tempfile
import json
import os

from causal_testing.testing.metamorphic_relation import generate_causal_tests
from .main import setup_logging, parse_args, CausalTestingPaths, CausalTestingFramework


def main() -> None:
    """

    Main entry point for the Causal Testing Framework

    """

    # Parse arguments
    args = parse_args()

    if args.generate:
        logging.info("Generating causal tests")
        generate_causal_tests(args.dag_path, args.output, args.ignore_cycles, args.threads)
        logging.info("Causal test generation completed successfully")
        return

    # Setup logging
    setup_logging(args.verbose)

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
            for i, results in enumerate(framework.run_tests_in_batches(batch_size=args.batch_size, silent=args.silent)):
                temp_file_path = os.path.join(tmpdir, f"output_{i}.json")
                framework.save_results(results, temp_file_path)
                output_files.append(temp_file_path)
                del results

            # Now stitch the results together from the temporary files
            all_results = []
            for file_path in output_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))

            # Save the final stitched results to your desired location
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4)
    else:
        logging.info("Running tests in regular mode")
        results = framework.run_tests(silent=args.silent)
        framework.save_results(results)

    logging.info("Causal testing completed successfully.")


if __name__ == "__main__":
    main()
