# Examples of how to use the framework

## Overview
The examples in this folder all illustrate ways to leverage the framework in actual situations. Emphasis is on 
end-to-end frameworks, including "all the glue stuff". 

### Basic examples
#### Univariate systems
1. [Closed-loop optimization of _known_ function](#Example-1:-Closed-loop-optimization-of-_known_-function)
2. [Iterative optimization of a function, which can be sampled](#Example-2:-Iterative-optimization-of-a-function,-which-can-be-sampled)
#### Multivariate systems
3. [Maximize a multivariate function using either closed-loop or iterative approach](#Example-3:-Maximize-a-multivariate-function-using-either-closed-loop-or-iterative-approach)
4. [Optimize a noisy, multivariate system](#Example-4:-Optimize-a-noisy,-multivariate-system)
#### Advanced examples
5. [Stop optimization early when convergence seems to be reached based on relative improvement in best response](#Example-5:-Stop-optimization-early-when-convergence-seems-to-be-reached-based-on-relative-improvement-in-best-response)
6. [Functions with integer covariates](#Example-6:-Functions-with-integer-covariates)
7. [Make the best brownie](#Example-7:-Make-the-best-brownie)
  

## Details

### Example 1: Closed-loop optimization of _known_ function

In this example, we use the `.auto`-method to perform closed-loop maximization of a known univariate function.

File: `Example 1 - Closed-loop optimization of known function.ipynb`

### Example 2: Iterative optimization of a function, which can be sampled
Here, the iterative solution approach accessible via the `.ask` and `.tell` methods is used to find the maximum of a univariate function, which can be sampled. The example makes use of a well-defined function, but any samplable function could be used (also e.g. the readout from a physical system).

File: `Example 2 - Iterative optimization of function, which can be sampled.ipynb`

### Example 3: Maximize a multivariate function using either closed-loop or iterative approach

In this example, a variant of the multivariate [Easom function](https://www.sfu.ca/~ssurjano/easom.html) is maximized using both the closed-loop approach of the `.auto`-method as well as the iterative approach of `.ask` and `.tell` methods. Similar to [Example 2](#Example-2:-Iterative-optimization-of-a-function,-which-can-be-sampled), the response function need not be explicitly defined for the iterative `.ask`-`.tell` methods to work, as long as the function can be sampled, the framework can be applied.

Towards the end, one of the two covariates ($x_1$) is changed to being an integer only, and it is demonstrated how the framework is used to solve under these different circumstances than for continuous variables only.

File: `Example 3 - Maximum of a multivariate function.ipynb`

### Example 4: Optimize a noisy, multivariate system

Here a noisy multivariate response function is optimized. This illustrates how to apply Bayesian optimization for noisy systems such as those found in many real-world applications. The noise could either be inherent in the system itself or be a measurement uncertainty.

File: `Example 4 - Optimize a noisy multivariate system.ipynb`

### Example 5: Stop optimization early when convergence seems to be reached based on relative improvement in best response

In this example illustrate how to use `rel_tol` and `rel_tol_steps`-parameters in `.auto`-method to stop iterations before the specified number of iterations if the solution is deemed converged. This is assessed by looking at the relative improvement in best response between consecutive iterations. `rel_tol` defines the relative improvement threshold required for convergence; also setting `rel_tol_steps` requires that this threshold is reached for `rel_tol_steps` consecutive iterations.

The example discusses best practise for using these functionalities to obtain good convergence.

File: `Example 5 - Optimization stopping criteria based on relative improvements.ipynb`

### Example 6: Functions with integer covariates

In this example it is shown how to define integer covariates and illustrates how the framework handles cases of mixed covariate data types (some continuous, some integer) better than pure integer covariates. 

File: `Example 6 - Functions with integer covariates.ipynb`

### Example 7: Make the best brownie

This example illustrates how to combine covariates of all the different data types (continuous, integer, categorical) to solve a complex and somewhat relevant real-life problem: finding the best brownie recipe!

File: `Example 7 - Make the best brownie.ipynb`