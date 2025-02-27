"""This module contains the main entrypoint functionality to the Causal Testing Framework."""

import logging
from .main import setup_logging, parse_args, CausalTestingPaths, CausalTestingFramework


def main() -> None:
    """

    Main entry point for the Causal Testing Framework

    """

    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
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
        results = framework.run_tests(silent=args.silent)

        # Save results
        framework.save_results(results)

        logging.info("Causal testing completed successfully.")

    except Exception as e:
        logging.error("Error during causal testing: %s", str(e))
        raise


if __name__ == "__main__":
    main()
