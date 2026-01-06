# =============================================================================
# Model 2: Double Pendulum Parameter Inference
# =============================================================================
#
# PROBLEM: Given observed trajectory of a double pendulum, infer physical
# parameters (lengths, masses) and initial conditions.
#
# SIMULATOR: Double pendulum ODE solver
#
# WHY THIS MODEL?
# - CHAOTIC DYNAMICS: Extreme sensitivity to initial conditions
# - Small parameter changes → dramatically different trajectories
# - Breaks gradient-based methods (HMC): gradients explode/vanish
# - Breaks all standard inference due to chaotic likelihood landscape
# - This is an "adversarial" test case - no method works well!
# =============================================================================

using Gen
using LinearAlgebra
using Statistics
using Random

# =============================================================================
# SIMULATOR: Double Pendulum Physics
# =============================================================================

"""
Double pendulum state: [θ1, θ2, ω1, ω2]
where θ = angle, ω = angular velocity

Parameters:
- L1, L2: pendulum lengths
- m1, m2: pendulum masses
- g: gravitational acceleration
"""
struct DoublePendulumParams
    L1::Float64  # Length of first pendulum
    L2::Float64  # Length of second pendulum
    m1::Float64  # Mass of first pendulum
    m2::Float64  # Mass of second pendulum
    g::Float64   # Gravitational acceleration
end

"""
Compute derivatives for double pendulum dynamics.
Uses Lagrangian mechanics formulation.

State: [θ1, θ2, ω1, ω2]
"""
function double_pendulum_derivatives(state::Vector{Float64}, p::DoublePendulumParams)
    θ1, θ2, ω1, ω2 = state
    L1, L2, m1, m2, g = p.L1, p.L2, p.m1, p.m2, p.g

    Δθ = θ1 - θ2

    # Denominators
    denom1 = L1 * (2*m1 + m2 - m2*cos(2*Δθ))
    denom2 = L2 * (2*m1 + m2 - m2*cos(2*Δθ))

    # Angular accelerations (from Lagrangian equations of motion)
    α1 = (-g*(2*m1 + m2)*sin(θ1) - m2*g*sin(θ1 - 2*θ2)
          - 2*sin(Δθ)*m2*(ω2^2*L2 + ω1^2*L1*cos(Δθ))) / denom1

    α2 = (2*sin(Δθ)*(ω1^2*L1*(m1 + m2) + g*(m1 + m2)*cos(θ1)
          + ω2^2*L2*m2*cos(Δθ))) / denom2

    return [ω1, ω2, α1, α2]
end

"""
Simulate double pendulum using RK4 integration.

Returns array of (x1, y1, x2, y2) positions over time.
"""
function simulate_double_pendulum(params::DoublePendulumParams,
                                  initial_state::Vector{Float64};
                                  dt::Float64=0.01,
                                  duration::Float64=5.0)
    n_steps = Int(ceil(duration / dt))
    state = copy(initial_state)

    # Store trajectory of bob positions
    trajectory = Vector{NTuple{4, Float64}}(undef, n_steps + 1)

    for i in 1:(n_steps + 1)
        θ1, θ2 = state[1], state[2]

        # Convert angles to Cartesian coordinates
        x1 = params.L1 * sin(θ1)
        y1 = -params.L1 * cos(θ1)
        x2 = x1 + params.L2 * sin(θ2)
        y2 = y1 - params.L2 * cos(θ2)

        trajectory[i] = (x1, y1, x2, y2)

        if i <= n_steps
            # RK4 integration step
            k1 = double_pendulum_derivatives(state, params)
            k2 = double_pendulum_derivatives(state .+ 0.5*dt.*k1, params)
            k3 = double_pendulum_derivatives(state .+ 0.5*dt.*k2, params)
            k4 = double_pendulum_derivatives(state .+ dt.*k3, params)

            state .+= (dt/6.0) .* (k1 .+ 2.0.*k2 .+ 2.0.*k3 .+ k4)
        end
    end

    return trajectory
end

"""
Extract just the second bob (x2, y2) positions from trajectory.
This is typically what we observe.
"""
function extract_bob2_trajectory(trajectory::Vector{NTuple{4, Float64}})
    return [(t[3], t[4]) for t in trajectory]
end

"""
Subsample trajectory at observation times.
"""
function subsample_trajectory(trajectory, obs_indices::Vector{Int})
    return [trajectory[i] for i in obs_indices]
end

# =============================================================================
# PROBABILISTIC MODEL
# =============================================================================

"""
Probabilistic model for double pendulum parameter inference.

Given observed positions of the second bob, infer:
- Physical parameters (lengths, masses)
- Initial conditions (angles, angular velocities)

This model is INTENTIONALLY DIFFICULT because:
1. Chaotic dynamics → tiny parameter changes cause huge trajectory changes
2. Likelihood landscape is extremely rough/discontinuous
3. Gradients are essentially meaningless due to chaos
"""
@gen function double_pendulum_model(observed_positions::Vector{Tuple{Float64, Float64}},
                                    obs_times::Vector{Float64},
                                    noise_std::Float64=0.1)
    # === Priors on physical parameters ===

    # Lengths (meters) - narrow prior to make problem slightly easier
    L1 = ({:L1} ~ uniform(0.5, 1.5))
    L2 = ({:L2} ~ uniform(0.5, 1.5))

    # Masses (kg) - fix m1=1 for identifiability, infer ratio
    m1 = 1.0
    m2 = ({:m2} ~ uniform(0.5, 2.0))

    # === Priors on initial conditions ===
    # Initial angles (radians)
    θ1_init = ({:theta1_init} ~ uniform(-π, π))
    θ2_init = ({:theta2_init} ~ uniform(-π, π))

    # Initial angular velocities (rad/s) - start near rest
    ω1_init = ({:omega1_init} ~ normal(0.0, 0.5))
    ω2_init = ({:omega2_init} ~ normal(0.0, 0.5))

    # === Run simulator ===
    params = DoublePendulumParams(L1, L2, m1, m2, 9.81)
    initial_state = [θ1_init, θ2_init, ω1_init, ω2_init]

    duration = maximum(obs_times)
    dt = 0.01
    trajectory = simulate_double_pendulum(params, initial_state;
                                          dt=dt, duration=duration)

    # Get positions at observation times
    bob2_traj = extract_bob2_trajectory(trajectory)

    # === Likelihood ===
    for (i, t) in enumerate(obs_times)
        # Find closest simulation time index
        idx = min(Int(round(t / dt)) + 1, length(bob2_traj))
        sim_x, sim_y = bob2_traj[idx]

        # Observed positions are noisy
        {(:obs_x, i)} ~ normal(sim_x, noise_std)
        {(:obs_y, i)} ~ normal(sim_y, noise_std)
    end

    return (L1, L2, m2, θ1_init, θ2_init)
end

"""
Simplified model: only infer initial angles, fix physical parameters.
Still chaotic, but fewer parameters.
"""
@gen function double_pendulum_simple(observed_positions::Vector{Tuple{Float64, Float64}},
                                     obs_times::Vector{Float64},
                                     L1::Float64, L2::Float64,
                                     noise_std::Float64=0.1)
    # Fixed physical parameters
    m1, m2, g = 1.0, 1.0, 9.81

    # Only infer initial angles
    θ1_init = ({:theta1_init} ~ uniform(-π/2, π/2))
    θ2_init = ({:theta2_init} ~ uniform(-π/2, π/2))

    # Start at rest
    ω1_init = 0.0
    ω2_init = 0.0

    # Run simulator
    params = DoublePendulumParams(L1, L2, m1, m2, g)
    initial_state = [θ1_init, θ2_init, ω1_init, ω2_init]

    duration = maximum(obs_times)
    trajectory = simulate_double_pendulum(params, initial_state;
                                          dt=0.01, duration=duration)

    bob2_traj = extract_bob2_trajectory(trajectory)

    # Likelihood
    dt = 0.01
    for (i, t) in enumerate(obs_times)
        idx = min(Int(round(t / dt)) + 1, length(bob2_traj))
        sim_x, sim_y = bob2_traj[idx]

        {(:obs_x, i)} ~ normal(sim_x, noise_std)
        {(:obs_y, i)} ~ normal(sim_y, noise_std)
    end

    return (θ1_init, θ2_init)
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
Generate synthetic observations from known parameters.
"""
function generate_observations(params::DoublePendulumParams,
                               initial_state::Vector{Float64},
                               obs_times::Vector{Float64},
                               noise_std::Float64)
    trajectory = simulate_double_pendulum(params, initial_state;
                                          duration=maximum(obs_times))
    bob2_traj = extract_bob2_trajectory(trajectory)

    dt = 0.01
    observations = Tuple{Float64, Float64}[]

    for t in obs_times
        idx = min(Int(round(t / dt)) + 1, length(bob2_traj))
        true_x, true_y = bob2_traj[idx]
        obs_x = true_x + randn() * noise_std
        obs_y = true_y + randn() * noise_std
        push!(observations, (obs_x, obs_y))
    end

    return observations, bob2_traj
end

"""
Create observation choicemap for inference.
"""
function make_observations(obs_positions::Vector{Tuple{Float64, Float64}})
    obs = choicemap()
    for (i, (x, y)) in enumerate(obs_positions)
        obs[(:obs_x, i)] = x
        obs[(:obs_y, i)] = y
    end
    return obs
end

"""
Run inference and return samples.
"""
function run_inference_comparison(observed_positions::Vector{Tuple{Float64, Float64}},
                                  obs_times::Vector{Float64};
                                  num_samples::Int=500,
                                  noise_std::Float64=0.1,
                                  L1::Float64=1.0,
                                  L2::Float64=1.0)
    observations = make_observations(observed_positions)

    results = Dict()

    # 1. Importance Resampling
    println("Running Importance Resampling...")
    is_samples = []
    for _ in 1:num_samples
        try
            (trace, _) = importance_resampling(
                double_pendulum_simple,
                (observed_positions, obs_times, L1, L2, noise_std),
                observations,
                50
            )
            push!(is_samples, (trace[:theta1_init], trace[:theta2_init]))
        catch
            # Skip failed samples
        end
    end
    results["importance_sampling"] = is_samples
    println("  Got $(length(is_samples)) valid samples")

    # 2. Metropolis-Hastings
    println("Running Metropolis-Hastings...")
    try
        (trace, _) = generate(double_pendulum_simple,
                              (observed_positions, obs_times, L1, L2, noise_std),
                              observations)
        mh_samples = []
        mh_accepted = 0
        for i in 1:num_samples
            (trace, accepted) = mh(trace, select(:theta1_init, :theta2_init))
            mh_accepted += accepted
            push!(mh_samples, (trace[:theta1_init], trace[:theta2_init]))
        end
        results["metropolis_hastings"] = mh_samples
        println("  MH acceptance rate: $(mh_accepted/num_samples)")
    catch e
        println("  MH failed: $e")
        results["metropolis_hastings"] = []
    end

    # 3. HMC - Expected to struggle due to chaos
    println("Running HMC...")
    try
        (trace, _) = generate(double_pendulum_simple,
                              (observed_positions, obs_times, L1, L2, noise_std),
                              observations)
        hmc_samples = []
        hmc_accepted = 0
        for i in 1:num_samples
            (trace, accepted) = hmc(trace, select(:theta1_init, :theta2_init);
                                    L=5, eps=0.01)
            hmc_accepted += accepted
            push!(hmc_samples, (trace[:theta1_init], trace[:theta2_init]))
        end
        results["hmc"] = hmc_samples
        println("  HMC acceptance rate: $(hmc_accepted/num_samples)")
    catch e
        println("  HMC failed: $e")
        results["hmc"] = []
    end

    return results
end

# =============================================================================
# DEMONSTRATION OF CHAOS
# =============================================================================

"""
Demonstrate sensitivity to initial conditions (chaos).
This shows why inference is fundamentally difficult.
"""
function demonstrate_chaos(;perturbation::Float64=0.001)
    println("\n=== Demonstrating Chaotic Sensitivity ===")

    params = DoublePendulumParams(1.0, 1.0, 1.0, 1.0, 9.81)

    # Base initial condition
    θ1, θ2 = π/4, π/4  # 45 degrees each
    ω1, ω2 = 0.0, 0.0

    base_state = [θ1, θ2, ω1, ω2]
    perturbed_state = [θ1 + perturbation, θ2, ω1, ω2]

    println("Perturbation in θ1: $perturbation radians ($(rad2deg(perturbation))°)")

    # Simulate both
    duration = 5.0
    traj_base = simulate_double_pendulum(params, base_state; duration=duration)
    traj_pert = simulate_double_pendulum(params, perturbed_state; duration=duration)

    bob2_base = extract_bob2_trajectory(traj_base)
    bob2_pert = extract_bob2_trajectory(traj_pert)

    # Compute divergence over time
    println("\nDivergence of second bob position over time:")
    times = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    dt = 0.01

    for t in times
        idx = min(Int(round(t / dt)) + 1, length(bob2_base))
        x1, y1 = bob2_base[idx]
        x2, y2 = bob2_pert[idx]
        dist = sqrt((x1-x2)^2 + (y1-y2)^2)
        println("  t=$(t)s: distance = $(round(dist, digits=4)) m")
    end

    # Final positions
    x1_final, y1_final = bob2_base[end]
    x2_final, y2_final = bob2_pert[end]
    final_dist = sqrt((x1_final-x2_final)^2 + (y1_final-y2_final)^2)

    println("\nFinal divergence after $(duration)s: $(round(final_dist, digits=4)) m")
    println("Amplification factor: $(round(final_dist/perturbation, digits=0))x")

    return traj_base, traj_pert
end

"""
Show how gradients behave in chaotic regime.
"""
function demonstrate_gradient_issues()
    println("\n=== Gradient Sensitivity in Chaotic System ===")

    params = DoublePendulumParams(1.0, 1.0, 1.0, 1.0, 9.81)

    # Target observation at t=3s
    target_state = [π/4, π/4, 0.0, 0.0]
    traj = simulate_double_pendulum(params, target_state; duration=3.0)
    bob2 = extract_bob2_trajectory(traj)
    target_pos = bob2[end]

    println("Target position at t=3s: $target_pos")

    # Compute finite difference "gradients" at different perturbation scales
    ε_values = [0.1, 0.01, 0.001, 0.0001]

    println("\nFinite difference gradients ∂(final_x)/∂(θ1_init):")

    for ε in ε_values
        state_plus = [π/4 + ε, π/4, 0.0, 0.0]
        state_minus = [π/4 - ε, π/4, 0.0, 0.0]

        traj_plus = simulate_double_pendulum(params, state_plus; duration=3.0)
        traj_minus = simulate_double_pendulum(params, state_minus; duration=3.0)

        bob2_plus = extract_bob2_trajectory(traj_plus)
        bob2_minus = extract_bob2_trajectory(traj_minus)

        grad = (bob2_plus[end][1] - bob2_minus[end][1]) / (2ε)

        println("  ε=$ε: gradient ≈ $(round(grad, digits=2))")
    end

    println("\nNote: Gradients do NOT converge! They fluctuate wildly.")
    println("This is why HMC fails for chaotic systems.")
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

function main()
    println("=" ^ 60)
    println("DOUBLE PENDULUM INFERENCE MODEL")
    println("=" ^ 60)

    # Demonstrate chaos first
    demonstrate_chaos()
    demonstrate_gradient_issues()

    # True parameters
    L1_true, L2_true = 1.0, 1.0
    m1_true, m2_true = 1.0, 1.0
    θ1_true, θ2_true = π/6, π/4  # 30° and 45°
    ω1_true, ω2_true = 0.0, 0.0

    params_true = DoublePendulumParams(L1_true, L2_true, m1_true, m2_true, 9.81)
    initial_state_true = [θ1_true, θ2_true, ω1_true, ω2_true]

    println("\n" * "=" ^ 60)
    println("TRUE PARAMETERS")
    println("=" ^ 60)
    println("L1=$L1_true, L2=$L2_true")
    println("θ1=$(rad2deg(θ1_true))°, θ2=$(rad2deg(θ2_true))°")

    # Generate observations
    obs_times = collect(0.0:0.2:2.0)  # Every 0.2s for 2 seconds
    noise_std = 0.05

    observed_positions, true_trajectory = generate_observations(
        params_true, initial_state_true, obs_times, noise_std
    )

    println("\nObservation times: $obs_times")
    println("Number of observations: $(length(observed_positions))")

    # Run inference
    println("\n" * "=" ^ 60)
    println("RUNNING INFERENCE (expect poor performance due to chaos)")
    println("=" ^ 60)

    results = run_inference_comparison(observed_positions, obs_times;
                                       num_samples=200,
                                       noise_std=noise_std,
                                       L1=L1_true, L2=L2_true)

    # Summary
    println("\n" * "=" ^ 60)
    println("RESULTS SUMMARY")
    println("=" ^ 60)
    println("True values: θ1=$(round(rad2deg(θ1_true), digits=1))°, θ2=$(round(rad2deg(θ2_true), digits=1))°")

    for (method, samples) in results
        if length(samples) > 0
            θ1s = [rad2deg(s[1]) for s in samples]
            θ2s = [rad2deg(s[2]) for s in samples]

            println("\n$method ($(length(samples)) samples):")
            println("  θ1: mean=$(round(mean(θ1s), digits=1))°, std=$(round(std(θ1s), digits=1))°")
            println("  θ2: mean=$(round(mean(θ2s), digits=1))°, std=$(round(std(θ2s), digits=1))°")
        else
            println("\n$method: No valid samples obtained")
        end
    end

    println("\n" * "=" ^ 60)
    println("CONCLUSION")
    println("=" ^ 60)
    println("The double pendulum's chaotic dynamics make posterior inference")
    println("extremely difficult for ALL standard inference methods:")
    println("- IS: Samples rarely match observations due to chaos")
    println("- MH: Random walk struggles in rough likelihood landscape")
    println("- HMC: Gradients are meaningless in chaotic regime")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
