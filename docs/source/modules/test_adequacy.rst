Causal Test Adequacy
====================

As with all testing techniques, the question of "How do I know when I can stop testing?" is a tricky one to answer.
In once sense, this is actually *easier* to answer for causal testing than in the traditional context, since the causal DAG provides an exhaustive list of all of the causal relationships (and independences) that should be checked.
However, determining whether your data leads to *accurate* causal effect estimates and *trustworthy* test outcomes is another matter.

Intuitively, we might say that the :term:`confidence intervals` associated with a given causal effect estimate should be able to give us some insight into this since wider intervals indicate a higher level of uncertainty associated with the estimate.
However, many systems are extremely stochastic and can produce outputs that vary by orders of magnitude, even for repeated runs of the same input configuration.
For such systems, the confidence intervals will always be wide, no matter how much data is collected.
What we really need to know to determine test adequacy is whether collecting additional data would change the causal effect estimate or, critically, the outcome of the test case.

To tackle this problem, we have produced a dedicated causal test adequacy metric.
Because of the complex nature of the test adequacy question in this context, the metric itself has a slightly more nuanced interpretation than a simple percentage.
The full technical details can be found in `this paper <https://ieeexplore.ieee.org/document/10638595>`_.
An informal explanation is as follows.

The fundamental question we want to answer is "will collecting additional data change the causal effect estimate?" or, in other words "is our causal effect estimate stable?".
To investigate this, we repeatedly resample the data and re-evaluate the test case, first calculating the causal effect estimate and then checking that it is as expected.
From the resampled causal effect estimates, we calculate the `kurtosis <https://en.wikipedia.org/wiki/Kurtosis>`_, which measures the "tailedness" of the distribution of estimates.
The basic idea here is that adequate test data will lead to a normal distribution of estimates.

Interpretation
--------------

Kurtosis values close to zero represent adequate test data.
This represents a *stable* causal effect estimate and a trustworthy test outcome.
Kurtosis values less than zero also represent stable causal effect estimates, but these can be thought of as "too stable".
In other words, we don't have enough data to have fully observed the stochasticity of the system under test.
Kurtosis values larger than zero represent unstable causal effect estimates.
That is, the estimates are highly dependent on individual data points, meaning that the causal test outcomes are unlikely to be trustworthy.
While, the acceptable kurtosis values will vary between systems and applications, a general rule would be that kurtosis values between 0 and 1 are mostly acceptable, with negative values being generally undesirable.

Configuration Options
---------------------

Since causal test adequacy involves repeatedly resampling the data and re-executing the test case, the main configuration option here is the number of times the data is resampled.
To obtain the most accurate estimate, more samples is always better, but this can become extremely computationally expensive, since the process effectively involves repeatedly executing the test suite, so for *n* bootstraps, the causal test adequacy calculation will take around *n* times the runtime of the test suite.
The default value is 100 samples, which we do not recommend going below.
If your test suite is very fast to run, we recommend running 1000 samples or even more.
