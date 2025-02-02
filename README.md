

# ``BrainUnit``: physical units and unit-aware mathematical system for brain dynamics and AI4Science

<p align="center">
  	<img alt="Header image of brainunit." src="https://github.com/chaobrain/brainunit/blob/main/docs/_static/brainunit.png" width=50%>
</p> 



<p align="center">
	<a href="https://pypi.org/project/brainunit/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brainunit"></a>
	<a href="https://github.com/chaobrain/brainunit/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
    <a href='https://brainunit.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/brainunit/badge/?version=latest' alt='Documentation Status' />
    </a>  	
    <a href="https://badge.fury.io/py/brainunit"><img alt="PyPI version" src="https://badge.fury.io/py/brainunit.svg"></a>
    <a href="https://github.com/chaobrain/brainunit/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/brainunit/actions/workflows/CI.yml/badge.svg"></a>
    <a href="https://pepy.tech/projects/brainunit"><img src="https://static.pepy.tech/badge/brainunit" alt="PyPI Downloads"></a>
</p>


## Motivation


[``brainunit``](https://github.com/chaobrain/brainunit) provides physical units and unit-aware mathematical system in JAX for brain dynamics and AI4Science. 

It is initially designed to enable unit-aware computations in brain dynamics modeling (see our [ecosystem](https://ecosystem-for-brain-dynamics.readthedocs.io/)).

However, its features and capacities can be applied to general domains in scientific computing and AI for science. 
We also provide ample examples and tutorials to help users integrate ``brainunit`` into their projects 
(see [Unit-aware computation ecosystem](#unit-aware-computation-ecosystem) in the below).


## Features


The uniqueness of ``Brainunit`` lies in that it brings physical units handling and AI-driven computation together in a seamless way:

- It provides over 2,000 commonly used physical units and constants.
- It implements over 500 unit-aware mathematical functions.
- Its physical units and unit-aware functions are fully compatible with JAX, including autograd, JIT, vecterization, parallelization, and others.


```mermaid
graph TD
    A[BrainUnit] --> B[Physical Units]
    A --> C[Mathematical Functions]
    A --> D[JAX Integration]
    B --> B1[2000+ Units]
    B --> B2[Physical Constants]
    C --> C1[500+ Unit-aware Functions]
    D --> D1[Autograd]
    D --> D2[JIT Compilation]
    D --> D3[Vectorization]
    D --> D4[Parallelization]
```

A quick example:

```python

import brainunit as u

# Define a physical quantity
x = 3.0 * u.meter
x
# [out] 3. * meter

# autograd
f = lambda x: x ** 3
u.autograd.grad(f)(x)
# [out] 27. * meter2 


# JIT
import jax
jax.jit(f)(x)
# [out] 27. * klitre

# vmap
jax.vmap(f)(u.math.arange(0. * u.mV, 10. * u.mV, 1. * u.mV))
# [out]  ArrayImpl([  0.,   1.,   8.,  27.,  64., 125., 216., 343., 512., 729.],
#                  dtype=float32) * mvolt3
```



## Installation

You can install ``brainunit`` via pip:

```bash
pip install brainunit --upgrade
```

## Documentation

The official documentation is hosted on Read the Docs: [https://brainunit.readthedocs.io](https://brainunit.readthedocs.io)


## Unit-aware computation ecosystem


`brainunit` has been deeply integrated into following diverse projects, such as:

- [``brainstate``](https://github.com/chaobrain/brainstate): A State-based Transformation System for Program Compilation and Augmentation
- [``braintaichi``](https://github.com/chaobrain/braintaichi): Leveraging Taichi Lang to customize brain dynamics operators
- [``braintools``](https://github.com/chaobrain/braintools): The Common Toolbox for Brain Dynamics Programming.
- [``dendritex``](https://github.com/chaobrain/dendritex): Dendritic Modeling in JAX
- [``pinnx``](https://github.com/chaobrain/pinnx): Physics-Informed Neural Networks for Scientific Machine Learning in JAX.


Other unofficial projects include:

- [``diffrax``](https://github.com/chaoming0625/diffrax): Numerical differential equation solvers in JAX.
- [``jax-md``](https://github.com/Routhleck/jax-md): Differentiable Molecular Dynamics in JAX
- [``Catalax``](https://github.com/Routhleck/Catalax): JAX-based framework to model biological systems
- ...



## See also the BDP ecosystem

We are building the [brain dynamics programming ecosystem](https://ecosystem-for-brain-dynamics.readthedocs.io/). 
[``brainunit``](https://github.com/chaobrain/brainunit) has been deeply integrated into our BDP ecosystem.
