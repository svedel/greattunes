# Examples of how to use the framework

## Overview
The examples in this folder all illustrate ways to leverage the framework in actual situations. Emphasis is on 
end-to-end frameworks, including ``all the glue stuff´´. 

### Basic examples
* Univariate systems
  1. [Closed-loop optimization of _known_ function](#Example-1:-Closed-loop-optimization-of-_known_-function)
  2. [Iterative optimization of a function, which can be sampled](#Example-2:-Iterative-optimization-of-a-function,-which-can-be-sampled)
* Multivariate systems
  3. [Maximize a multivariate function using either closed-loop or iterative approach](#Example-3:-Maximize-a-multivariate-function-using-either-closed-loop-or-iterative-approach)
  4. [Optimize a noisy, multivariate system](#Example-4:-Optimize-a-noisy,-multivariate-system)
  

## Details

### Example 1: Closed-loop optimization of _known_ function

In this example, we use the `.auto`-method to perform closed-loop maximization of a known univariate function.

File: `Example 1 - Closed-loop optimization of known function.ipynb`

### Example 2: Iterative optimization of a function, which can be sampled
Here, the iterative solution approach accessible via the `.ask` and `.tell` methods is used to find the maximum of a univariate function, which can be sampled. The example makes use of a well-defined function, but any samplable function could be used (also e.g. the readout from a physical system).

File: `Example 2 - Iterative optimization of function, which can be sampled.ipynb`

### Example 3: Maximize a multivariate function using either closed-loop or iterative approach

In this example, a variant of the multivariate [Easom function](https://www.sfu.ca/~ssurjano/easom.html) is maximized using both the closed-loop approach of the `.auto`-method as well as the iterative approach of `.ask` and `.tell` methods. Similar to [Example 2](#Example-2:-Iterative-optimization-of-a-function,-which-can-be-sampled), the response function need not be explicitly defined for the iterative `.ask`-`.tell` methods to work, as long as the function can be sampled, the framework can be applied.

File: `Example 3 - Maximum of a multivariate function.ipynb`

### Example 4: Optimize a noisy, multivariate system

Here a noisy multivariate response function is optimized. This illustrates how to apply Bayesian optimization for noisy systems such as those found in many real-world applications. The noise could either be inherent in the system itself or be a measurement uncertainty.

File: `Example 4 - Optimize a noisy multivariate system.ipynb`