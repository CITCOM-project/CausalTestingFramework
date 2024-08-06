import random
import warnings
import patsy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels
import sympy
import copy

from functools import partial
from deap import algorithms, base, creator, tools, gp

from numpy import negative, exp, power, log, sin, cos, tan, sinh, cosh, tanh
from inspect import isclass

from operator import add, mul


def root(x):
    return power(x, 0.5)


def square(x):
    return power(x, 2)


def cube(x):
    return power(x, 3)


def fourth_power(x):
    return power(x, 4)


def reciprocal(x):
    return power(x, -1)


def mutInsert(individual, pset):
    """
    Copied from gp.mutInsert, except that we import isclass from inspect, so we
    won't have the "isclass not defined" bug.

    Inserts a new branch at a random position in *individual*. The subtree
    at the chosen position is used as child node of the created subtree, in
    that way, it is really an insertion rather than a replacement. Note that
    the original subtree will become one of the children of the new primitive
    inserted, but not perforce the first (its position is randomly selected if
    the new primitive has more than one child).

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    choice = random.choice

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

    if len(primitives) == 0:
        return (individual,)

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)
    position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = choice(pset.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position : position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree
    return (individual,)


class GP:

    def __init__(
        self,
        df: pd.DataFrame,
        features: list,
        outcome: str,
        extra_operators: list = None,
        sympy_conversions: dict = None,
        seed=0,
    ):
        random.seed(seed)
        self.df = df
        self.features = features
        self.outcome = outcome
        self.seed = seed
        self.pset = gp.PrimitiveSet("MAIN", len(self.features))
        self.pset.renameArguments(**{f"ARG{i}": f for i, f in enumerate(self.features)})

        standard_operators = [(add, 2), (mul, 2), (reciprocal, 1)]
        if extra_operators is None:
            extra_operators = [(log, 1), (reciprocal, 1)]
        for operator, num_args in standard_operators + extra_operators:
            self.pset.addPrimitive(operator, num_args)
        if sympy_conversions is None:
            sympy_conversions = {}
        self.sympy_conversions = {
            "mul": lambda *args_: "Mul({},{})".format(*args_),
            "add": lambda *args_: "Add({},{})".format(*args_),
            "reciprocal": lambda *args_: "Pow({},-1)".format(*args_),
        } | sympy_conversions

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.evalSymbReg)
        self.toolbox.register("repair", self.repair)
        self.toolbox.register("select", tools.selBest)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", self.mutate, expr=self.toolbox.expr_mut)
        self.toolbox.decorate("mate", gp.staticLimit(key=lambda x: x.height + 1, max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=lambda x: x.height + 1, max_value=17))

    def split(self, individual):
        if len(individual) > 1:
            terms = []
            # Recurse over children if add/sub
            if individual[0].name in ["add", "sub"]:
                terms.extend(
                    self.split(
                        creator.Individual(
                            gp.PrimitiveTree(
                                individual[individual.searchSubtree(1).start : individual.searchSubtree(1).stop]
                            )
                        )
                    )
                )
                terms.extend(
                    self.split(creator.Individual(gp.PrimitiveTree(individual[individual.searchSubtree(1).stop :])))
                )
            else:
                terms.append(individual)
            return terms
        return [individual]

    def _convert_inverse_prim(self, prim, args):
        """
        Convert inverse prims according to:
        [Dd]iv(a,b) -> Mul[a, 1/b]
        [Ss]ub(a,b) -> Add[a, -b]
        We achieve this by overwriting the corresponding format method of the sub and div prim.
        """
        prim = copy.copy(prim)
        prim_formatter = self.sympy_conversions.get(prim.name, prim.format)

        return prim_formatter(*args)

    def _stringify_for_sympy(self, f):
        """Return the expression in a human readable string."""
        string = ""
        stack = []
        for node in f:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = self._convert_inverse_prim(prim, args)
                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)
        return string

    def simplify(self, individual):
        return sympy.simplify(self._stringify_for_sympy(individual))

    def repair(self, individual):
        eq = f"{self.outcome} ~ {' + '.join(str(x) for x in self.split(individual))}"
        try:
            # Create model, fit (run) it, give estimates from it]
            model = smf.ols(eq, self.df)
            res = model.fit()
            y_estimates = res.predict(self.df)

            eqn = f"{res.params['Intercept']}"
            for term, coefficient in res.params.items():
                if term != "Intercept":
                    eqn = f"add({eqn}, mul({coefficient}, {term}))"
            repaired = type(individual)(gp.PrimitiveTree.from_string(eqn, self.pset))
            return repaired
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
            statsmodels.tools.sm_exceptions.MissingDataError,
            patsy.PatsyError,
        ) as e:
            return individual

    def evalSymbReg(self, individual):
        old_settings = np.seterr(all="raise")
        try:
            # Create model, fit (run) it, give estimates from it]
            func = gp.compile(individual, self.pset)
            y_estimates = pd.Series([func(**x) for _, x in self.df[self.features].iterrows()])

            # Calc errors using an improved normalised mean squared
            sqerrors = (self.df[self.outcome] - y_estimates) ** 2
            mean_squared = sqerrors.sum() / len(self.df)
            nmse = mean_squared / (self.df[self.outcome].sum() / len(self.df))

            return (nmse,)

            # Fitness value of infinite if error - not return 1
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
            statsmodels.tools.sm_exceptions.MissingDataError,
            patsy.PatsyError,
            RuntimeWarning,
            FloatingPointError,
        ) as e:
            return (float("inf"),)
        finally:
            np.seterr(**old_settings)  # Restore original settings

    def make_offspring(self, population, lambda_):
        offspring = []
        for i in range(lambda_):
            parent1, parent2 = tools.selTournament(population, 2, 2)
            child, _ = self.toolbox.mate(self.toolbox.clone(parent1), self.toolbox.clone(parent2))
            del child.fitness.values
            (child,) = self.toolbox.mutate(child)
            offspring.append(child)
        return offspring

    def eaMuPlusLambda(self, ngen, mu=20, lambda_=10, stats=None, verbose=False, seeds=None):
        population = [self.toolbox.repair(ind) for ind in self.toolbox.population(n=mu)]
        if seeds is not None:
            for seed in seeds:
                ind = creator.Individual(gp.PrimitiveTree.from_string(seed, self.pset))
                ind.fitness.values = self.toolbox.evaluate(ind)
                population.append(ind)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        for ind in population:
            ind.fitness.values = self.toolbox.evaluate(ind)
        population.sort(key=lambda x: (x.fitness.values, x.height))

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Vary the population
            offspring = self.make_offspring(population, lambda_)
            offspring = [self.toolbox.repair(ind) for ind in offspring]

            # Evaluate the individuals with an invalid fitness
            for ind in offspring:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Select the next generation population
            population[:] = self.toolbox.select(population + offspring, mu)

            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(offspring), **record)
            if verbose:
                print(logbook.stream)
            population.sort(key=lambda x: (x.fitness.values, x.height))

        return population[0]

    def mutate(self, individual, expr):
        choice = random.randint(1, 3)
        if choice == 1:
            mutated = gp.mutNodeReplacement(self.toolbox.clone(individual), self.pset)
        elif choice == 2:
            mutated = mutInsert(self.toolbox.clone(individual), self.pset)
        elif choice == 3:
            mutated = gp.mutShrink(self.toolbox.clone(individual))
        else:
            raise ValueError("Invalid mutation choice")
        return mutated


if __name__ == "__main__":
    df = pd.DataFrame()
    df["X"] = np.arange(10)
    df["Y"] = 1 / (df.X + 1)

    gp1 = GP(df.astype(float), ["X"], "Y", seed=1)
    best = gp1.eaMuPlusLambda(ngen=100)
    print(best, best.fitness.values[0])
    simplified = gp1.simplify(best)
    print(simplified)
