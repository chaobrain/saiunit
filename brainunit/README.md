<h1 align='center'>BrainUnit</h1>
<h2 align='center'>Physical units and unit-aware math system for general-purpose brain dynamics modeling</h2>

<p align="center">
  	<img alt="Header image of brainunit." src="https://raw.githubusercontent.com/chaobrain/saiunit/main/docs/_static/brainunit.png" width=30%>
</p> 



<p align="center">
	<a href="https://pypi.org/project/brainunit/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brainunit"></a>
	<a href="https://github.com/chaobrain/brainunit/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
    <a href='https://brainunit.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/brainunit/badge/?version=latest' alt='Documentation Status' />
    </a>  	
    <a href="https://badge.fury.io/py/brainunit"><img alt="PyPI version" src="https://badge.fury.io/py/brainunit.svg"></a>
    <a href="https://pepy.tech/projects/brainunit"><img src="https://static.pepy.tech/badge/brainunit" alt="PyPI Downloads"></a>
</p>




[BrainUnit](https://github.com/chaobrain/brainunit) provides physical units and unit-aware mathematical system in JAX for brain dynamics modeling. It introduces rigoirous physical units into high-performance AI-driven abstract numerical computing. 

BrainUnit is initially designed to enable unit-aware computations in brain dynamics modeling (see our [BDP ecosystem](https://ecosystem-for-brain-dynamics.readthedocs.io/)). However, its features and capacities can be applied to general domains in scientific computing and AI4Science. Starting in 2025/02, BrainUnit has been fully integrated into [SAIUnit](https://github.com/chaobrain/saiunit) (the **Unit** system for **S**cientific **AI**). 

Functionalities are the same for both ``brainunit`` and ``saiunit``, and their functions and data structures are interoperable, sharing the same set of APIs, and eliminating any potential conflicts. This meas that 

```python
import brainunit as u
```

equals to 

```python
import saiunit as u
```

For users primarily engaged in general scientific computing, `saiunit` is likely the preferred choice. However, for those focused on brain modeling, we recommend `brainunit`, as it is more closely aligned with our specialized brain dynamics programming ecosystem.



## Documentation

The official documentation of BrainUnit is hosted on Read the Docs: [https://brainunit.readthedocs.io](https://brainunit.readthedocs.io)



## Features

`brainunit` can be seamlessly integrated into every aspect of our [brain dynamics programming ecosystem](https://ecosystem-for-brain-dynamics.readthedocs.io/), such as, the checkpointing of [braintools](https://github.com/chaobrain/braintools), the event-driven operators in [brainevent](https://github.com/chaobrain/brainevent), the state-based JIT compilation in [brainstate](https://github.com/chaobrain/brainstate), online learning rules in [brainscale](https://github.com/chaobrain/brainscale), or event more. 

A quick example for this kind of integration:

```python

import braintools
import brainevent.nn
import brainstate
import brainunit as u


class EINet(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_exc = 3200
        self.n_inh = 800
        self.num = self.n_exc + self.n_inh
        self.N = brainstate.nn.LIFRef(
            self.num, V_rest=-60. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=brainstate.init.Normal(-55., 2., unit=u.mV)
        )
        self.E = brainstate.nn.AlignPostProj(
            comm=brainevent.nn.FixedProb(self.n_exc, self.num, 0.02, 0.6 * u.mS),
            syn=brainstate.nn.Expon.desc(self.num, tau=5. * u.ms),
            out=brainstate.nn.COBA.desc(E=0. * u.mV),
            post=self.N
        )
        self.I = brainstate.nn.AlignPostProj(
            comm=brainevent.nn.FixedProb(self.n_inh, self.num, 0.02, 6.7 * u.mS),
            syn=brainstate.nn.Expon.desc(self.num, tau=10. * u.ms),
            out=brainstate.nn.COBA.desc(E=-80. * u.mV),
            post=self.N
        )

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike() != 0.
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            self.N(inp)
            return self.N.get_spike()
    
    def save_checkpoint(self):
        braintools.file.msgpack_save('states.msgpack', self.states())
    
```



## Installation

You can install ``brainunit`` via pip:

```bash
pip install brainunit --upgrade
```



## See also the BDP ecosystem

We are building the [brain dynamics programming (BDP) ecosystem](https://ecosystem-for-brain-dynamics.readthedocs.io/). [brainunit](https://github.com/chaobrain/brainunit) has been deeply integrated into our BDP ecosystem.
