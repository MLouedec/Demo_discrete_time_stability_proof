# Demo_discrete_time_stability_proof
A demonstration of a numerical method to prove the exponential stability of nonlinear high-dimensional discrete-time systems
and to find positive invariant ellipsoids. It is related to the paper [Add a reference]
There are two executable main files
Dependencies: numpy, scipy, sympy, matplotlib and codac

## main_guaranteed_computation.py

The main file of the demo - all operations are guaranteed with intervals

The first part of the script defines the discrete-time systems. some parameters can be tuned.
The second part of the script proves the exponential stability of the system and finds a positive invariant ellipsoid

This second part uses the ellipsoidal propagation method presented in [Add the link]

The method can be too pessimistic and raise an error. 

The ellipsoids are plotted at the end 

## main_fast_computation.py

an optional file. the matrix inversion and the matrix square root are not guaranteed. The algorithm is much faster but not guaranteed

## lib.py

A toolbox for the main files
