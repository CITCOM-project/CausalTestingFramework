import logging

from causal_testing.causal_testing_framework import CausalTestingFramework
from causal_testing.estimation.ipcw_estimator import IPCWEstimator
from causal_testing.testing.causal_effect import SomeEffect
from causal_testing.testing.causal_test_case import CausalTestCase


def minimise_test(
    test: list[tuple[int, str, any]],
    ctf: CausalTestingFramework,
    safe_ranges: dict[str, dict[str:float]],
    total_time: int,
    start_time: int = 0,
    background_confounders: list[str] = None,
    timesteps_per_intervention=1,
    ci_alpha=0.05,
):
    background_confounders = background_confounders if background_confounders is not None else []

    outcome = test["outcome"]
    logging.debug(f"\nOUTCOME: {outcome}")

    test["test"] = list(filter(lambda x: start_time <= x[0] <= total_time, test["test"]))

    lo, hi = safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]

    test["safe_range"] = (lo, hi)
    control_strategy = test["test"]
    logging.debug(f"  CONTROL STRATEGY   {control_strategy}")
    test["control_strategy"] = control_strategy

    if not (~ctf.df[outcome].between(lo, hi)).any():
        raise ValueError(
            f"No faults with {outcome}. Cannot perform estimation.\n"
            f"Observed range [{ctf.df[outcome].min()}, {ctf.df[outcome].max()}].\n"
            f"Safe range {safe_ranges[outcome]}"
        )
    if ctf.df[outcome].between(lo, hi).all():
        raise ValueError(
            f"All faults with {outcome}. Cannot perform estimation.\n"
            f"Observed range [{ctf.df[outcome].min()}, {ctf.df[outcome].max()}].\n"
            f"Safe range {safe_ranges[outcome]}"
        )
    if any(var not in ctf.df for _, var, _ in control_strategy):
        raise ValueError("Missing data for control_strategy")
    if any(var not in ctf.dag.nodes for _, var, _ in control_strategy):
        missing = [var for _, var, _ in control_strategy if var not in ctf.dag.nodes]
        raise ValueError(f"Missing nodes {missing} for control_strategy. Valid nodes {ctf.dag.nodes}")

    indexed_control = list(enumerate(control_strategy))

    for i in range(0, len(control_strategy)):
        print(f"Event {i}/{len(control_strategy)}")
        if "treatment_strategies" not in test:
            test["treatment_strategies"] = []
        indexed_capabilities = indexed_control[i : i + 1]
        treatment_strategy = [x[:] for x in control_strategy]
        for i, capability in indexed_capabilities:
            _, variable, value = capability
            # Treatment strategy is the same, but with one capability negated
            # i.e. we examine the counterfactual "What if we had not done that?"
            treatment_strategy[i][2] = int(not value)
        result = {"treatment_strategy": treatment_strategy, "intervention_index": i}
        test["treatment_strategies"].append(result)

        logging.debug(f"  TREATMENT STRATEGY {treatment_strategy}")
        logging.debug(f"  OUTCOME {outcome}")
        logging.debug(f"  SAFE RANGE {lo} {hi}")

        neighbours = list(ctf.dag.predecessors(variable))
        neighbours += list(ctf.dag.successors(variable))

        if len(neighbours) == 0:
            raise ValueError(f"No neighbours for node {variable}.")

        if "time" not in background_confounders:
            background_confounders.append("time")
        fitBLswitch_formula = f"xo_t_do ~ {' + '.join(background_confounders)}"
        ctf.df["within_safe_range"] = ctf.df[outcome].between(lo, hi)

        try:
            causal_test_case = CausalTestCase(
                expected_causal_effect=SomeEffect(),
                # control_value=control_strategy,
                # treatment_value=treatment_strategy,
                effect_measure="hazard_ratio",
                # effect="temporal",
                estimator=IPCWEstimator(
                    timesteps_per_intervention,
                    control_strategy,
                    treatment_strategy,
                    outcome,
                    "within_safe_range",
                    fit_bl_switch_formula=fitBLswitch_formula,
                    fit_bltd_switch_formula=f"{fitBLswitch_formula} + {' + '.join(neighbours)}",
                    eligibility=None,
                    alpha=ci_alpha,
                    total_time=total_time,
                ),
            )
        except ValueError as e:
            logging.error(f"ValueError: {e}")
            result["error"] = f"ValueError: {e}"

        causal_test_case.execute_test(df=ctf.df, suppress_estimation_errors=True)
