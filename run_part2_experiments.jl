# =============================================================================
# Part 2: Custom Inference Experiments Runner
# =============================================================================
#
# This script runs all inference experiments for both models and
# generates comparison results.
#
# Usage: julia run_part2_experiments.jl
# =============================================================================

using Random

println("=" ^ 70)
println("PART 2: CUSTOM INFERENCE EXPERIMENTS")
println("=" ^ 70)

# Set random seed for reproducibility
Random.seed!(42)

# =============================================================================
# Model 1: Projectile/Artillery
# =============================================================================

println("\n")
println("=" ^ 70)
println("MODEL 1: PROJECTILE/ARTILLERY")
println("=" ^ 70)

include("models/projectile_custom_inference.jl")

println("\nRunning projectile model comparison...")
projectile_results = run_full_comparison(
    v0_true=30.0,
    theta_true=deg2rad(45.0),
    n_observations=5,
    noise_std=2.0,
    num_samples=500
)

# =============================================================================
# Model 2: Double Pendulum
# =============================================================================

println("\n\n")
println("=" ^ 70)
println("MODEL 2: DOUBLE PENDULUM (CHAOTIC)")
println("=" ^ 70)

include("models/double_pendulum_custom_inference.jl")

println("\nRunning double pendulum model comparison...")
dp_results, dp_obs, dp_times = run_full_comparison(
    theta1_true=pi/6,
    theta2_true=pi/4,
    L1=1.0,
    L2=1.0,
    obs_duration=2.0,
    noise_std=0.05,
    num_samples=300
)

# =============================================================================
# Summary
# =============================================================================

println("\n\n")
println("=" ^ 70)
println("EXPERIMENT SUMMARY")
println("=" ^ 70)

println("""

PROJECTILE MODEL:
-----------------
Custom inference strategies that work well:
1. Correlated MH Proposals - Proposes along iso-range curves
2. Parallel Tempering - Enables jumping between high/low arc modes
3. Reparameterized Model - Uses (range, height) instead of (v0, theta)
4. Combined Strategy - Best overall performance

Key insight: The multimodal posterior is smooth, so gradient-based
methods (HMC) work well. Custom proposals that respect the parameter
correlation further improve efficiency.

DOUBLE PENDULUM MODEL:
----------------------
Custom inference strategies for chaotic systems:
1. ABC-MCMC - Uses summary statistics instead of exact likelihood
2. Short-Window Inference - Only matches early observations (pre-chaos)
3. Tempered Inference - Flattens rough likelihood for exploration
4. Hierarchical Matching - Multi-scale trajectory matching

Key insight: Chaos makes exact posterior inference intractable.
We must use approximate methods that focus on robust features
(summary statistics, early trajectory) rather than exact matching.

CONCLUSIONS:
------------
1. For smooth, multimodal problems: Use parallel tempering + custom proposals
2. For chaotic problems: Use ABC or short-window inference
3. No single method works for all problems - tailor to the specific challenge!
""")

println("\nExperiments complete. See SUBMISSION_PART2.md for full writeup.")
