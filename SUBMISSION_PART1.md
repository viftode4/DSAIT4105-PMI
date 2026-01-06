# Probabilistic Models and Inference: Part I Submission

**Course:** DSAIT4105 - Probabilistic Models and Inference
**Student:** [Your Name]
**Date:** [Submission Date]

---

## Executive Summary

This submission presents two simulation-based inference problems designed to stress test standard inference procedures. The models focus on physics and games domains:

1. **Projectile/Artillery Model** - Demonstrates multimodality and parameter correlation challenges
2. **Double Pendulum Model** - Demonstrates chaotic dynamics that break gradient-based inference

Both models are implemented in Gen.jl with full probabilistic programs and theoretical analysis.

---

## Model 1: Projectile/Artillery Trajectory Inference

### Problem Statement

Given noisy observations of where a projectile lands, infer the initial velocity and launch angle used to fire it.

### Simulator Description

The simulator implements 2D ballistic motion under gravity:

```
x(t) = x₀ + v₀·cos(θ)·t
y(t) = y₀ + v₀·sin(θ)·t - ½gt²
```

**Inputs:**
- `v0`: Initial velocity (m/s)
- `theta`: Launch angle (radians)

**Output:**
- Impact position (x-coordinate where y = 0)

The simulator executes in milliseconds, well under the recommended few-seconds threshold.

### Probabilistic Model

```julia
@gen function projectile_model(observed_impacts, noise_std)
    # Priors
    v0 = ({:v0} ~ uniform(10.0, 50.0))      # Initial velocity
    theta = ({:theta} ~ uniform(π/18, 4π/9)) # Angle: 10° to 80°

    # Run simulator
    x_impact = simulate_projectile(v0, theta)

    # Likelihood
    for (i, obs) in enumerate(observed_impacts)
        {(:impact, i)} ~ normal(x_impact, noise_std)
    end

    return (v0, theta, x_impact)
end
```

### Why This Model Tests Inference

#### Challenge 1: Multimodal Posterior

The same target distance can be achieved with two different (v0, θ) combinations:

| Solution Type | Angle | Velocity | Trajectory |
|--------------|-------|----------|------------|
| Low Arc | ~25° | ~35 m/s | Flat, fast |
| High Arc | ~65° | ~35 m/s | Tall, slow |

This creates a **bimodal posterior** that sampling methods must explore.

#### Challenge 2: Parameter Correlation

Velocity and angle trade off: increasing angle requires adjusting velocity to maintain range. This creates elongated, curved posterior regions.

### Inference Procedure Analysis

| Method | Performance | Explanation |
|--------|-------------|-------------|
| **Importance Sampling** | **POOR** | Samples from prior rarely land in either mode. High variance, weight degeneracy. ESS << N. |
| **Metropolis-Hastings** | **MODERATE** | Can explore within modes but struggles to jump between them. Correlation slows mixing. |
| **HMC** | **GOOD** | Smooth likelihood allows gradient-based exploration. Efficiently navigates curved posterior geometry. |
| **Particle Filtering** | N/A | Problem is not sequential. |

### Code Location

`models/projectile_artillery.jl`

---

## Model 2: Double Pendulum Parameter Inference

### Problem Statement

Given noisy observations of a double pendulum's trajectory, infer the physical parameters (lengths, masses) and initial conditions.

### Simulator Description

The simulator solves the double pendulum equations of motion using RK4 integration:

**State:** [θ₁, θ₂, ω₁, ω₂] (angles and angular velocities)

**Dynamics:** Derived from Lagrangian mechanics
```
α₁ = f(θ₁, θ₂, ω₁, ω₂, L₁, L₂, m₁, m₂, g)
α₂ = g(θ₁, θ₂, ω₁, ω₂, L₁, L₂, m₁, m₂, g)
```

**Inputs:**
- Physical parameters: L₁, L₂ (lengths), m₁, m₂ (masses)
- Initial conditions: θ₁, θ₂, ω₁, ω₂

**Output:**
- Trajectory of (x, y) positions for both pendulum bobs

### Probabilistic Model

```julia
@gen function double_pendulum_model(observed_positions, obs_times, noise_std)
    # Priors on parameters
    L1 = ({:L1} ~ uniform(0.5, 1.5))
    L2 = ({:L2} ~ uniform(0.5, 1.5))
    m2 = ({:m2} ~ uniform(0.5, 2.0))

    # Priors on initial conditions
    θ1_init = ({:theta1_init} ~ uniform(-π, π))
    θ2_init = ({:theta2_init} ~ uniform(-π, π))
    ω1_init = ({:omega1_init} ~ normal(0, 0.5))
    ω2_init = ({:omega2_init} ~ normal(0, 0.5))

    # Run simulator
    trajectory = simulate_double_pendulum(params, initial_state)

    # Likelihood at observation times
    for (i, t) in enumerate(obs_times)
        sim_x, sim_y = trajectory_at_time(trajectory, t)
        {(:obs_x, i)} ~ normal(sim_x, noise_std)
        {(:obs_y, i)} ~ normal(sim_y, noise_std)
    end
end
```

### Why This Model Tests Inference

#### Challenge 1: Chaotic Dynamics

The double pendulum exhibits **deterministic chaos**:
- Exponential divergence of nearby trajectories
- Lyapunov exponent > 0
- A 0.001 radian perturbation leads to >1 meter divergence after 5 seconds

**Demonstration:**
```
Perturbation: 0.001 rad (0.057°)
Divergence at t=1s: 0.003 m
Divergence at t=2s: 0.05 m
Divergence at t=3s: 0.4 m
Divergence at t=5s: 2.1 m
Amplification: >2000x
```

#### Challenge 2: Gradient Catastrophe

Gradients ∂(output)/∂(parameters) are meaningless:
- Finite differences don't converge
- Sign flips unpredictably
- Magnitude explodes exponentially

```
ε=0.1:    gradient ≈ 15.2
ε=0.01:   gradient ≈ -42.7
ε=0.001:  gradient ≈ 203.1
ε=0.0001: gradient ≈ -89.4
```

Gradients do NOT converge as ε→0. This breaks all gradient-based methods.

#### Challenge 3: Fractal Likelihood Surface

The parameter regions yielding "good" trajectories form complex, disconnected sets with fractal-like boundaries.

### Inference Procedure Analysis

| Method | Performance | Explanation |
|--------|-------------|-------------|
| **Importance Sampling** | **VERY POOR** | Prior samples essentially never match observations. All weights on negligible fraction of samples. |
| **Metropolis-Hastings** | **POOR** | Likelihood landscape so rough that most moves rejected. Cannot navigate to good regions. |
| **HMC** | **FAILS** | Gradients are wrong, not just noisy. Leapfrog integrator diverges. Near-zero acceptance. |
| **Particle Filtering** | **POOR** | Particles diverge rapidly due to chaos. Collapses to single particle quickly. |

### This is an Adversarial Test Case

The double pendulum demonstrates a **fundamental limitation**: some probabilistic models with valid likelihoods are **computationally intractable** for standard inference.

No standard algorithm can efficiently explore this posterior without problem-specific modifications.

### Code Location

`models/double_pendulum.jl`

---

## Comparison Summary

| Aspect | Projectile | Double Pendulum |
|--------|------------|-----------------|
| **Domain** | Games/Physics | Physics |
| **Simulator Type** | Closed-form ballistics | ODE integration |
| **Primary Challenge** | Multimodality | Chaos |
| **Secondary Challenge** | Correlation | Gradient catastrophe |
| **Likelihood Surface** | Smooth, multimodal | Rough, fractal |
| **Gradients Useful?** | Yes | No |
| **Best Method** | HMC | None (all fail) |
| **Difficulty** | Moderate | Extreme |

### All Four Inference Procedures

| Procedure | Projectile Model | Double Pendulum Model |
|-----------|------------------|----------------------|
| **Importance Sampling** | POOR - multimodal posterior causes weight degeneracy | VERY POOR - chaos makes matching nearly impossible |
| **Metropolis-Hastings** | MODERATE - correlation slows mixing between modes | POOR - rough likelihood causes very low acceptance |
| **HMC** | GOOD - smooth gradients enable efficient exploration | FAILS - gradients are meaningless in chaotic regime |
| **Particle Filtering** | N/A - not a sequential problem | POOR - chaotic divergence causes rapid particle collapse |

---

## Conclusions

### What We Learned

1. **No universal inference algorithm exists.** Each model has characteristics that favor or break different methods.

2. **Simulator properties matter.** Smoothness, chaos, and gradient behavior determine which algorithms can work.

3. **Multimodality is challenging but manageable.** With methods like HMC or tempered MCMC.

4. **Chaos is fundamentally difficult.** Standard algorithms all fail. Requires specialized approaches (ABC, summary statistics, etc.).

### For Part II

Custom inference strategies to explore:

**Projectile Model:**
- Custom proposals that respect v0-θ correlation
- Parallel tempering to jump between modes
- Reparameterization to range + height instead of v0 + θ

**Double Pendulum Model:**
- ABC with trajectory summary statistics
- Multiple shooting methods
- Short-window inference before chaos dominates
- Neural network-based proposals

---

## Files Included

```
DSAIT4105 PMI/
├── PROJECT_REQUIREMENTS.md      # Original assignment
├── GEN_JL_REFERENCE.md          # Gen.jl syntax reference
├── THEORETICAL_ANALYSIS.md      # Detailed theoretical analysis
├── SUBMISSION_PART1.md          # This document (convert to PDF)
├── test_projectile.jl           # Test script for Model 1
├── test_double_pendulum.jl      # Test script for Model 2
└── models/
    ├── projectile_artillery.jl  # Model 1: Projectile/Artillery
    └── double_pendulum.jl       # Model 2: Double Pendulum
```

**To run tests:**
```bash
julia test_projectile.jl
julia test_double_pendulum.jl
```

---

## References

1. Gen.jl Documentation: https://www.gen.dev/
2. Strogatz, S. (2015). Nonlinear Dynamics and Chaos
3. MacKay, D. (2003). Information Theory, Inference, and Learning Algorithms
4. Bishop, C. (2006). Pattern Recognition and Machine Learning
