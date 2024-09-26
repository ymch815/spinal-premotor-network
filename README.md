# Code for paper "The spinal premotor network driving high-frequency flexor-extensor alternation"
Simulation of a neuromechanical model that captures the activity of spinal premotor network driving scratching behavior
The code for joint model are build on Fink et al. (2014), see also
https://github.com/anagamori/Fink2014-model/blob/main/biomechanical_model.ipynb

## Dependencies
- numpy 1.23.5
- pandas 2.0.3
- scipy 1.11.1

## Quick Start
### Run simulations on V1 ablation experiment when V1 fraction varies from 0% to 100% (ablation efficiency from 100% to 0%)
```
python SC_sim.py V1 ablation 0.0 1.0 0 1 21 irflag_ccrate_jointfreq_phasediff
```
### Generate traces of different dynamics types
See generate_trace.ipynb
