"""
This module contains a genetic programming implementation to infer the functional
form between the adjustment set and the outcome.
"""

import copy
from inspect import isclass
from operator import add, mul
import random

import patsy
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels
import sympy

from deap import base, creator, tools, gp

from numpy import power, log


def reciprocal(x: float) -> float:
    """
    Return the reciprocal of the input.
    :param x: Float to reciprocate.
    :return: 1/x
    """
    return power(x, -1)


def mut_insert(expression: gp.PrimitiveTree, pset: gp.PrimitiveSet):
    """
    NOTE: This is a temporary workaround. This method is copied verbatim from
    gp.mutInsert. It seems they forgot to import isclass from inspect, so their
    method throws an error, saying that "isclass is not defined". A couple of
    lines are not covered by tests, but since this is 1. a temporary workaround
    until they release a new version of DEAP, and 2. not our code, I don't think
    that matters.

    Inserts a new branch at a random position in *expression*. The subtree
    at the chosen position is used as child node of the created subtree, in
    that way, it is really an insertion rather than a replacement. Note that
    the original subtree will become one of the children of the new primitive
    inserted, but not perforce the first (its position is randomly selected if
    the new primitive has more than one child).

    :param expression: The normal or typed tree to be mutated.
    :param pset: The pset object defining the variables and constants.

    :return: A tuple of one tree.
    """
    index = random.randrange(len(expression))
    node = expression[index]
    expr_slice = expression.searchSubtree(index)
    choice = random.choice

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

    if len(primitives) == 0:
        return (expression,)

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)
    position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = choice(pset.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position : position + 1] = expression[expr_slice]
    new_subtree.insert(0, new_node)
    expression[expr_slice] = new_subtree
    return (expression,)


def create_power_function(order: int):
    """
    Creates a power operator and its corresponding sympy conversion.

    :param order: The order of the power, e.g. `order=2` will give x^2.

    :return: A pair consisting of the power function and the sympy conversion
    """

    def power_func(x):
        return power(x, order)

    def sympy_conversion(x):
        return f"Pow({x},{order})"

    return power_func, sympy_conversion


class GP:
    """
    Object to perform genetic programming.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        df: pd.DataFrame,
        features: list,
        outcome: str,
        max_order: int = 0,
        extra_operators: list = None,
        sympy_conversions: dict = None,
        seed=0,
    ):
        # pylint: disable=too-many-arguments
        random.seed(seed)
        self.df = df
        self.features = features
        self.outcome = outcome
        self.max_order = max_order
        self.seed = seed
        self.pset = gp.PrimitiveSet("MAIN", len(self.features))
        self.pset.renameArguments(**{f"ARG{i}": f for i, f in enumerate(self.features)})

        standard_operators = [(add, 2), (mul, 2)]
        if extra_operators is None:
            extra_operators = [(log, 1), (reciprocal, 1)]
        if sympy_conversions is None:
            sympy_conversions = {}
        for operator, num_args in standard_operators + extra_operators:
            self.pset.addPrimitive(operator, num_args)

        self.sympy_conversions = {
            "mul": lambda x1, x2: f"Mul({x1},{x2})",
            "add": lambda x1, x2: f"Add({x1},{x2})",
            "reciprocal": lambda x1: f"Pow({x1},-1)",
        } | sympy_conversions

        for i in range(self.max_order + 1):
            name = f"power_{i}"
            func, conversion = create_power_function(i)
            self.pset.addPrimitive(func, 1, name=name)
            if name in self.sympy_conversions:
                raise ValueError(
                    f"You have provided a function called {name}, which is reserved for raising to power"
                    f"{i}. Please choose a different name for your function."
                )
            self.sympy_conversions[name] = conversion

        print(self.pset.mapping)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.fitness)
        self.toolbox.register("repair", self.repair)
        self.toolbox.register("select", tools.selBest)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.decorate("mate", gp.staticLimit(key=lambda x: x.height + 1, max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=lambda x: x.height + 1, max_value=17))

    def split(self, individual: gp.PrimitiveTree) -> list:
        """
        Split an expression into its components, e.g. 2x + 4y - xy -> [2x, 4y, xy].

        :param individual: The expression to be split.
        :return: A list of the equations components that are linearly combined into the full equation.
        """
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

    def _convert_prim(self, prim: gp.Primitive, args: list) -> str:
        """
        Convert primitives to sympy format.

        :param prim: A GP primitive, e.g. add
        :param args: The list of arguments

        :return: A sympy compatible string representing the function, e.g. add(x, y) -> Add(x, y).
        """
        prim = copy.copy(prim)
        prim_formatter = self.sympy_conversions.get(prim.name, prim.format)
        return prim_formatter(*args)

    def _stringify_for_sympy(self, expression: gp.PrimitiveTree) -> str:
        """
        Return the expression in a sympy compatible string.

        :param expression: The expression to be simplified.

        :return: A sympy compatible string representing the equation.
        """
        string = ""
        stack = []
        for node in expression:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = self._convert_prim(prim, args)
                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)
        return string

    def simplify(self, expression: gp.PrimitiveTree) -> sympy.core.Expr:
        """
        Simplify an expression by appling mathematical equivalences.

        :param expression: The expression to simplify.

        :return: The simplified expression as a sympy Expr object.
        """
        return sympy.simplify(self._stringify_for_sympy(expression))

    def repair(self, expression: gp.PrimitiveTree) -> gp.PrimitiveTree:
        """
        Use linear regression to infer the coefficients of the linear components of the expression.
        Named "repair" since a "repair operator" is quite common in GP.

        :param expression: The expression to process.

        :return: The expression with constant coefficients, or the original expression if that fails.
        """
        eq = f"{self.outcome} ~ {' + '.join(str(x) for x in self.split(expression))}"
        try:
            # Create model, fit (run) it, give estimates from it]
            model = smf.ols(eq, self.df)
            res = model.fit()

            eqn = f"{res.params['Intercept']}"
            for term, coefficient in res.params.items():
                if term != "Intercept":
                    eqn = f"add({eqn}, mul({coefficient}, {term}))"
            repaired = type(expression)(gp.PrimitiveTree.from_string(eqn, self.pset))
            return repaired
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
            statsmodels.tools.sm_exceptions.MissingDataError,
            patsy.PatsyError,
        ):
            return expression

    def fitness(self, expression: gp.PrimitiveTree) -> float:
        """
        Evaluate the fitness of an candidate expression according to the error between the estimated and observed
        values. Low values are better.

        :param expression: The candidate expression to evaluate.

        :return: The fitness of the individual.
        """
        old_settings = np.seterr(all="raise")
        try:
            # Create model, fit (run) it, give estimates from it]
            func = gp.compile(expression, self.pset)
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
        ):
            return (float("inf"),)
        finally:
            np.seterr(**old_settings)  # Restore original settings

    def make_offspring(self, population: list, num_offspring: int) -> list:
        """
        Create the next generation of individuals.

        :param population: The current population.
        :param num_offspring: The number of new individuals to generate.

        :return: A list of num_offspring new individuals generated through crossover and mutation.
        """
        offspring = []
        for _ in range(num_offspring):
            parent1, parent2 = tools.selTournament(population, 2, 2)
            child, _ = self.toolbox.mate(self.toolbox.clone(parent1), self.toolbox.clone(parent2))
            del child.fitness.values
            (child,) = self.toolbox.mutate(child)
            offspring.append(child)
        return offspring

    def run_gp(self, ngen: int, pop_size: int = 20, num_offspring: int = 10, seeds: list = None) -> gp.PrimitiveTree:
        """
        Execute Genetic Programming to find the best expression using a mu+lambda algorithm.

        :param ngen: The maximum number of generations.
        :param pop_size: The population size.
        :param num_offspring: The number of new individuals per generation.
        :param seeds: Seed individuals for the initial population.

        :return: The best candididate expression.
        """
        population = [self.toolbox.repair(ind) for ind in self.toolbox.population(n=pop_size)]
        if seeds is not None:
            for seed in seeds:
                ind = creator.Individual(gp.PrimitiveTree.from_string(seed, self.pset))
                ind.fitness.values = self.toolbox.evaluate(ind)
                population.append(ind)

        # Evaluate the individuals with an invalid fitness
        for ind in population:
            ind.fitness.values = self.toolbox.evaluate(ind)
        population.sort(key=lambda x: (x.fitness.values, x.height))

        # Begin the generational process
        for _ in range(1, ngen + 1):
            # Vary the population
            offspring = self.make_offspring(population, num_offspring)
            offspring = [self.toolbox.repair(ind) for ind in offspring]

            # Evaluate the individuals with an invalid fitness
            for ind in offspring:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Select the best pop_size individuals to continue to the next generation
            population[:] = self.toolbox.select(population + offspring, pop_size)

            # Update the statistics with the new population
            population.sort(key=lambda x: (x.fitness.values, x.height))

        return population[0]

    def mutate(self, expression: gp.PrimitiveTree) -> gp.PrimitiveTree:
        """
        mutate individuals to replicate the small changes in DNA that occur in natural reproduction.
        A node will randomly be inserted, removed, or replaced.

        :param expression: The expression to mutate.

        :return: The mutated expression.
        """
        choice = random.randint(1, 3)
        if choice == 1:
            mutated = gp.mutNodeReplacement(self.toolbox.clone(expression), self.pset)
        elif choice == 2:
            mutated = mut_insert(self.toolbox.clone(expression), self.pset)
        elif choice == 3:
            mutated = gp.mutShrink(self.toolbox.clone(expression))
        return mutated
