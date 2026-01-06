# Theoretical Analysis: Inference Procedure Stress Tests

## Overview

This document provides theoretical analysis of two simulation-based inference problems designed to stress test different inference procedures. Each model exploits specific weaknesses of standard inference algorithms.

---

## Model 1: Projectile/Artillery Trajectory Inference

### Problem Description

**Simulator:** 2D ballistic trajectory with gravity (optionally with air resistance)

**Latent Variables:**
- `v0`: Initial velocity (m/s)
- `theta`: Launch angle (radians)
- (Optional) `drag`: Air resistance coefficient

**Observations:** Noisy measurements of impact position (x-coordinate where projectile lands)

**Goal:** Infer initial velocity and launch angle from observed impact positions.

### Why This Model is Interesting

#### 1. Multimodal Posterior

The fundamental challenge is that **multiple (v0, θ) combinations can hit the same target**. From basic projectile physics:

```
Range = (v0² * sin(2θ)) / g
```

For a given range R, there exist two solutions:
- **Low trajectory:** Lower angle, higher velocity
- **High trajectory:** Higher angle, different velocity

This creates a **bimodal posterior distribution** that is challenging for many inference methods.

#### 2. Parameter Correlation

Initial velocity and launch angle are **strongly correlated** in the posterior. If you increase the angle, you must adjust the velocity to maintain the same range. This correlation structure:
- Creates elongated "banana-shaped" posterior regions
- Makes random walk proposals inefficient
- Requires proposal distributions that respect the correlation structure

### Inference Procedure Analysis

| Method | Expected Performance | Reasoning |
|--------|---------------------|-----------|
| **Importance Sampling** | POOR | Cannot efficiently explore multimodal distributions. Most samples will miss one or both modes, leading to high variance estimates. Weight degeneracy is likely. |
| **Metropolis-Hastings** | MODERATE | Can eventually explore both modes but: (1) Random walk struggles with correlation, (2) May get stuck in one mode for long periods, (3) Mixing between modes is slow. |
| **HMC** | GOOD | Smooth, differentiable likelihood allows gradient-based exploration. HMC can follow the curved posterior geometry. However, may still struggle to jump between modes. |
| **Particle Filtering** | N/A | Not applicable - this is not a sequential inference problem. |

### Key Challenges for Each Method

**Importance Sampling:**
- Prior samples rarely hit the narrow "good" region
- Multimodality means weights concentrate on few particles
- Effective Sample Size (ESS) will be very low

**Metropolis-Hastings:**
- Standard proposal (independent on v0 and θ) ignores correlation
- Block proposals needed but hard to tune
- Mode switching requires proposals that can "jump" in parameter space

**HMC:**
- Works well within each mode (smooth gradients)
- Energy conservation helps explore curved posteriors
- Challenge: May not discover second mode if initialized in first

### Modifications to Increase Difficulty

1. **Add air resistance:** Introduces third parameter, more complex dynamics
2. **Multiple targets:** Observe impacts at different distances
3. **Noisy wind:** Add stochastic wind component during flight
4. **3D trajectories:** Azimuth angle adds another degree of freedom

---

## Model 2: Double Pendulum Parameter Inference

### Problem Description

**Simulator:** Double pendulum ODE solver using Lagrangian mechanics

**Latent Variables:**
- `L1, L2`: Pendulum lengths
- `m2`: Mass ratio (m1 fixed for identifiability)
- `θ1_init, θ2_init`: Initial angles
- `ω1_init, ω2_init`: Initial angular velocities

**Observations:** Noisy (x, y) positions of the second bob at discrete times

**Goal:** Infer physical parameters and/or initial conditions from observed trajectory.

### Why This Model is Interesting

#### 1. Chaotic Dynamics

The double pendulum is a **canonical example of deterministic chaos**. Key properties:

- **Sensitive dependence on initial conditions:** Trajectories that start arbitrarily close diverge exponentially over time
- **Lyapunov exponent > 0:** Small perturbations grow as e^(λt)
- **Apparent randomness:** Long-term behavior appears stochastic despite deterministic dynamics

**Quantitative Example:**
A perturbation of 0.001 radians (~0.06°) in initial angle leads to completely different trajectories after just 3-5 seconds. The divergence amplification can exceed 1000x.

#### 2. Pathological Likelihood Landscape

Due to chaos, the likelihood function p(observations | parameters) has:
- **Extreme roughness:** Tiny parameter changes cause likelihood to jump discontinuously
- **No useful gradient information:** Finite differences don't converge
- **Fractal-like structure:** The "good" parameter regions form complex, disconnected sets

#### 3. Gradient Catastrophe

For gradient-based methods (HMC, variational inference):

```
∂L/∂θ = ∂(simulator output)/∂θ × ∂(likelihood)/∂(simulator output)
```

The term ∂(simulator output)/∂θ **explodes** due to chaos:
- Gradients grow exponentially with simulation time
- Sign of gradient flips unpredictably
- Automatic differentiation produces numerically meaningless values

### Inference Procedure Analysis

| Method | Expected Performance | Reasoning |
|--------|---------------------|-----------|
| **Importance Sampling** | VERY POOR | Prior samples almost never produce trajectories matching observations. Essentially all weight on negligible fraction of samples. |
| **Metropolis-Hastings** | POOR | Likelihood landscape is so rough that most proposed moves are rejected. Cannot find "good" regions through random walk. |
| **HMC** | FAILS | Gradients are meaningless in chaotic regime. Hamiltonian dynamics don't preserve energy because computed gradients are wrong. Very low acceptance rate. |
| **Particle Filtering** | POOR-MODERATE | Can track short trajectories but particles diverge rapidly. Requires massive number of particles or frequent resampling. |

### Key Challenges for Each Method

**Importance Sampling:**
- Prior probability of matching observations is essentially zero
- No "nearby" samples to provide useful weights
- Would need astronomically many samples

**Metropolis-Hastings:**
- Acceptance rate near zero for any reasonable step size
- Large steps: Always rejected (likelihood changes dramatically)
- Small steps: Still rejected (chaos means even tiny moves change trajectory)

**HMC:**
- Leapfrog integrator relies on accurate gradients
- Chaotic gradients cause energy to explode
- Proposals are always rejected
- Even with very small step sizes, gradients point in wrong direction

**Particle Filtering:**
- Useful for sequential state estimation
- But particles representing parameter hypotheses diverge
- Resampling leads to particle collapse onto single hypothesis
- Doesn't help with parameter inference

### Why This is an "Adversarial" Test Case

The double pendulum represents a **fundamental limitation** of standard inference procedures:

1. **No method can efficiently sample from chaotic posteriors** without problem-specific structure
2. **Gradients provide no useful information** in chaotic regime
3. **The problem is well-posed** (likelihood is defined) but **inference is computationally intractable**

This model demonstrates that simply having a valid probabilistic model doesn't guarantee tractable inference.

### Possible Approaches (Beyond Standard Methods)

For Part 2 of the project, potential custom inference strategies:

1. **ABC (Approximate Bayesian Computation):** Use summary statistics instead of full trajectory match
2. **Multiple shooting:** Break trajectory into segments, match at boundaries
3. **Shorter observation windows:** Infer from data before chaos dominates
4. **Hierarchical inference:** First estimate rough parameters, then refine
5. **Learned proposals:** Train neural network to propose good parameters

---

## Summary Table

| Aspect | Projectile Model | Double Pendulum Model |
|--------|------------------|----------------------|
| **Primary Challenge** | Multimodality, parameter correlation | Chaos, gradient catastrophe |
| **Likelihood Surface** | Smooth but multimodal | Extremely rough, fractal-like |
| **Gradients Useful?** | Yes | No |
| **Best Standard Method** | HMC | None work well |
| **Worst Method** | Importance Sampling | HMC (gradients misleading) |
| **Particle Filtering** | N/A (not sequential) | POOR (chaotic divergence) |
| **Difficulty Level** | Moderate | Extreme |

## All Four Inference Procedures Summary

| Procedure | Projectile Model | Double Pendulum Model |
|-----------|------------------|----------------------|
| **Importance Sampling** | POOR (multimodal, weight degeneracy) | VERY POOR (chaos, no matching samples) |
| **Metropolis-Hastings** | MODERATE (correlation slows mixing) | POOR (rough likelihood, low acceptance) |
| **HMC** | GOOD (smooth gradients work) | FAILS (gradients meaningless) |
| **Particle Filtering** | N/A (not sequential) | POOR (particle collapse from chaos) |

---

## Conclusion

These two models provide complementary stress tests:

1. **Projectile Model:** Tests ability to handle multimodality and parameter correlation. A "solvable but challenging" problem where good proposals and HMC should help.

2. **Double Pendulum Model:** Tests fundamental limits of inference in chaotic systems. Demonstrates that some problems require specialized techniques beyond standard algorithms.

Together, they illustrate the key lesson: **there is no universal inference algorithm**, and understanding model structure is essential for designing effective inference procedures.
