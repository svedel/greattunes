# Examples of how to use the framework

## Overview
The examples in this folder all illustrate ways to leverage the framework in actual situations. Emphasis is on 
end-to-end frameworks, including ``all the glue stuff´´. 

Basic examples
1. [Closed-loop optimization of _known_ function](#Example-1:-Closed-loop-optimization-of-_known_-function)
2. [Iterative optimization of a function, which can be sampled](#Example-2:-Iterative-optimization-of-a-function,-which-can-be-sampled)

## Details

### Example 1: Closed-loop optimization of _known_ function
In this example, we use the `.auto`-method to perform closed-loop maximization of a known function.

File: [Example 1 - Closed-loop optimization of known function.ipynb](Example-1---Closed-loop-optimization-of-known-function.ipynb)

### Example 2: Iterative optimization of a function, which can be sampled
Here, the iterative solution approach accessible via the `.ask` and `.tell` methods is used to find the maximum of a function, which can be sampled. The example makes use of a well-defined function, but any samplable function could be used (also e.g. the readout from a physical system).

File: [Example 2 - Iterative optimization of function, which can be sampled.ipynb](examples/Example-2---Iterative-optimization-of-function,-which-can-be-sampled.ipynb)