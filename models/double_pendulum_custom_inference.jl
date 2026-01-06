# =============================================================================
# Custom Inference Procedures for Double Pendulum Model
# =============================================================================
#
# The double pendulum exhibits CHAOTIC DYNAMICS, which breaks standard inference:
# - Importance Sampling: Prior samples never match observations
# - MH: Rough likelihood causes very low acceptance
# - HMC: Gradients are meaningless due to chaos
#
# CUSTOM STRATEGIES:
# 1. ABC (Approximate Bayesian Computation) with summary statistics
# 2. Short-window inference (only match early observations before chaos)
# 3. Likelihood tempering (smoothed likelihood)
# 4. Hierarchical summary matching
#
# =============================================================================

using Gen
using LinearAlgebra
using Statistics
using Random

include("double_pendulum.jl")

# =============================================================================
# STRATEGY 1: ABC with Summary Statistics
# =============================================================================

"""
Compute summary statistics from a trajectory.
These are designed to be robust to chaotic variability.

Summary statistics:
1. Mean position (x, y) of second bob
2. Standard deviation of position
3. Total energy (conserved quantity)
4. Mean angular velocity
5. Position at early time points (before chaos kicks in)
"""
function compute_summary_statistics(trajectory::Vector{NTuple{4, Float64}},
                                    params::DoublePendulumParams)
    bob2_traj = extract_bob2_trajectory(trajectory)

    xs = [p[1] for p in bob2_traj]
    ys = [p[2] for p in bob2_traj]

    # Basic statistics
    mean_x = mean(xs)
    mean_y = mean(ys)
    std_x = std(xs)
    std_y = std(ys)

    # Range of motion
    range_x = maximum(xs) - minimum(xs)
    range_y = maximum(ys) - minimum(ys)

    # Early trajectory positions (first 10% of trajectory - before chaos)
    n_early = max(1, div(length(bob2_traj), 10))
    early_x = mean(xs[1:n_early])
    early_y = mean(ys[1:n_early])

    # Very early positions (first 5 points)
    very_early = min(5, length(bob2_traj))
    very_early_x = xs[very_early]
    very_early_y = ys[very_early]

    return [mean_x, mean_y, std_x, std_y, range_x, range_y,
            early_x, early_y, very_early_x, very_early_y]
end

"""
Compute summary statistics from a simulated trajectory,
sampled at the SAME times as observations.
This is CRITICAL for correct ABC: must compare same-length data.
"""
function compute_summaries_at_obs_times(trajectory::Vector{NTuple{4, Float64}},
                                        obs_times::Vector{Float64};
                                        dt::Float64=0.01)
    bob2_traj = extract_bob2_trajectory(trajectory)

    # Get positions at exact observation time indices
    subsampled_xs = Float64[]
    subsampled_ys = Float64[]

    for t in obs_times
        idx = min(Int(round(t / dt)) + 1, length(bob2_traj))
        x, y = bob2_traj[idx]
        push!(subsampled_xs, x)
        push!(subsampled_ys, y)
    end

    xs, ys = subsampled_xs, subsampled_ys
    n = length(xs)
    n_early = max(1, div(n, 10))
    very_early = min(2, n)

    return [mean(xs), mean(ys),
            n > 1 ? std(xs) : 0.0, n > 1 ? std(ys) : 0.0,
            maximum(xs) - minimum(xs), maximum(ys) - minimum(ys),
            mean(xs[1:n_early]), mean(ys[1:n_early]),
            xs[very_early], ys[very_early]]
end

"""
Compute summary statistics directly from observation positions.
Used for ABC comparison - ensures observations and simulations
are compared using same-length data structures.
"""
function compute_obs_summaries(observed_positions::Vector{Tuple{Float64, Float64}})
    obs_xs = [p[1] for p in observed_positions]
    obs_ys = [p[2] for p in observed_positions]
    n = length(obs_xs)
    n_early = max(1, div(n, 10))
    very_early = min(2, n)

    return [mean(obs_xs), mean(obs_ys),
            n > 1 ? std(obs_xs) : 0.0, n > 1 ? std(obs_ys) : 0.0,
            maximum(obs_xs) - minimum(obs_xs), maximum(obs_ys) - minimum(obs_ys),
            mean(obs_xs[1:n_early]), mean(obs_ys[1:n_early]),
            obs_xs[very_early], obs_ys[very_early]]
end

"""
ABC distance function: compares summary statistics.
Uses weighted Euclidean distance.
"""
function abc_distance(summary1::Vector{Float64}, summary2::Vector{Float64})
    # Weight early observations more heavily (they're more reliable)
    weights = [1.0, 1.0,    # mean position
               0.5, 0.5,    # std position
               0.3, 0.3,    # range
               3.0, 3.0,    # early mean (more important!)
               5.0, 5.0]    # very early positions (most important!)

    diff = summary1 .- summary2
    weighted_diff = diff .* weights

    return sqrt(sum(weighted_diff.^2))
end

"""
ABC rejection sampler.
Accepts samples if distance to observed summaries is below threshold.
"""
function run_abc_rejection(observed_positions::Vector{Tuple{Float64, Float64}},
                           obs_times::Vector{Float64};
                           num_samples::Int=1000,
                           epsilon::Float64=0.5,
                           L1::Float64=1.0, L2::Float64=1.0,
                           max_attempts::Int=100000)
    # Compute observed summary statistics
    # First, we need to create a "pseudo-trajectory" from observations
    # (just repeat obs positions to fill trajectory-like structure)
    observed_trajectory = [(0.0, 0.0, pos[1], pos[2]) for pos in observed_positions]
    params_ref = DoublePendulumParams(L1, L2, 1.0, 1.0, 9.81)
    observed_summaries = compute_summary_statistics(observed_trajectory, params_ref)

    samples = []
    distances = Float64[]
    attempts = 0

    while length(samples) < num_samples && attempts < max_attempts
        attempts += 1

        # Sample from prior
        theta1 = rand() * pi - pi/2  # uniform(-π/2, π/2)
        theta2 = rand() * pi - pi/2

        # Simulate
        params = DoublePendulumParams(L1, L2, 1.0, 1.0, 9.81)
        initial_state = [theta1, theta2, 0.0, 0.0]

        try
            duration = maximum(obs_times)
            trajectory = simulate_double_pendulum(params, initial_state;
                                                  dt=0.01, duration=duration)

            # Compute summaries
            sim_summaries = compute_summary_statistics(trajectory, params)

            # ABC acceptance
            dist = abc_distance(sim_summaries, observed_summaries)

            if dist < epsilon
                push!(samples, (theta1, theta2))
                push!(distances, dist)
            end
        catch
            # Skip failed simulations
        end
    end

    acceptance_rate = length(samples) / attempts

    return samples, distances, acceptance_rate
end

"""
ABC-MCMC: Use ABC within an MCMC framework.
More efficient than pure rejection.

FIXED VERSION: Uses compute_summaries_at_obs_times to ensure
simulations are subsampled at observation times before comparing.
"""
function run_abc_mcmc(observed_positions::Vector{Tuple{Float64, Float64}},
                      obs_times::Vector{Float64};
                      num_samples::Int=1000,
                      epsilon::Float64=0.5,
                      proposal_std::Float64=0.2,
                      L1::Float64=1.0, L2::Float64=1.0)
    # FIX: Compute observed summaries DIRECTLY from observations
    # This ensures both observed and simulated summaries have same length
    observed_summaries = compute_obs_summaries(observed_positions)

    # Initialize at random
    theta1_current = rand() * pi - pi/2
    theta2_current = rand() * pi - pi/2
    dist_current = Inf

    # Compute initial distance
    try
        params = DoublePendulumParams(L1, L2, 1.0, 1.0, 9.81)
        initial_state = [theta1_current, theta2_current, 0.0, 0.0]
        trajectory = simulate_double_pendulum(params, initial_state;
                                              duration=maximum(obs_times))
        # FIX: Use summaries at observation times (same length as observed)
        summaries = compute_summaries_at_obs_times(trajectory, obs_times)
        dist_current = abc_distance(summaries, observed_summaries)
    catch
        # Keep Inf if fails
    end

    samples = [(theta1_current, theta2_current)]
    distances = [dist_current]
    accepted = 0

    for i in 1:num_samples
        # Propose new values
        theta1_prop = theta1_current + randn() * proposal_std
        theta2_prop = theta2_current + randn() * proposal_std

        # Clamp to prior range
        theta1_prop = clamp(theta1_prop, -pi/2, pi/2)
        theta2_prop = clamp(theta2_prop, -pi/2, pi/2)

        # Simulate and compute distance
        dist_prop = Inf
        try
            params = DoublePendulumParams(L1, L2, 1.0, 1.0, 9.81)
            initial_state = [theta1_prop, theta2_prop, 0.0, 0.0]
            trajectory = simulate_double_pendulum(params, initial_state;
                                                  duration=maximum(obs_times))
            # FIX: Use summaries at observation times
            summaries = compute_summaries_at_obs_times(trajectory, obs_times)
            dist_prop = abc_distance(summaries, observed_summaries)
        catch
            # Keep Inf if fails
        end

        # ABC-MCMC acceptance: accept if both are below epsilon,
        # or if proposal is closer to observations
        accept = false
        if dist_prop < epsilon && dist_current < epsilon
            accept = true  # Both acceptable, always accept
        elseif dist_prop < dist_current
            accept = true  # Proposal is better
        elseif dist_prop < epsilon && dist_current >= epsilon
            accept = true  # Proposal is acceptable, current is not
        end

        if accept
            theta1_current = theta1_prop
            theta2_current = theta2_prop
            dist_current = dist_prop
            accepted += 1
        end

        push!(samples, (theta1_current, theta2_current))
        push!(distances, dist_current)
    end

    acceptance_rate = accepted / num_samples

    return samples, distances, acceptance_rate
end

# =============================================================================
# STRATEGY 2: Short-Window Inference
# =============================================================================

"""
Short-window double pendulum model.
Only matches observations in a short initial window before chaos dominates.
"""
@gen function double_pendulum_short_window(observed_positions::Vector{Tuple{Float64, Float64}},
                                            obs_times::Vector{Float64},
                                            L1::Float64, L2::Float64,
                                            noise_std::Float64,
                                            max_time::Float64)  # Only use observations up to max_time
    m1, m2, g = 1.0, 1.0, 9.81

    # Priors
    theta1_init = ({:theta1_init} ~ uniform(-pi/2, pi/2))
    theta2_init = ({:theta2_init} ~ uniform(-pi/2, pi/2))
    omega1_init = 0.0
    omega2_init = 0.0

    # Simulate
    params = DoublePendulumParams(L1, L2, m1, m2, g)
    initial_state = [theta1_init, theta2_init, omega1_init, omega2_init]

    trajectory = simulate_double_pendulum(params, initial_state;
                                          dt=0.01, duration=max_time)
    bob2_traj = extract_bob2_trajectory(trajectory)

    # Only match early observations
    dt = 0.01
    for (i, t) in enumerate(obs_times)
        if t <= max_time
            idx = min(Int(round(t / dt)) + 1, length(bob2_traj))
            sim_x, sim_y = bob2_traj[idx]

            {(:obs_x, i)} ~ normal(sim_x, noise_std)
            {(:obs_y, i)} ~ normal(sim_y, noise_std)
        end
    end

    return (theta1_init, theta2_init)
end

"""
Run short-window inference.
Key insight: Chaos takes time to develop. Early trajectory is more predictable.
"""
function run_short_window_inference(observed_positions::Vector{Tuple{Float64, Float64}},
                                    obs_times::Vector{Float64};
                                    num_samples::Int=500,
                                    noise_std::Float64=0.1,
                                    window::Float64=0.5,  # Only use first 0.5 seconds
                                    L1::Float64=1.0, L2::Float64=1.0)
    # Filter observations to short window
    short_obs = Tuple{Float64, Float64}[]
    short_times = Float64[]
    for (i, t) in enumerate(obs_times)
        if t <= window
            push!(short_obs, observed_positions[i])
            push!(short_times, t)
        end
    end

    if isempty(short_obs)
        println("  Warning: No observations in window. Using first observation.")
        short_obs = [observed_positions[1]]
        short_times = [obs_times[1]]
    end

    # Create observations choicemap
    obs = choicemap()
    for (i, (x, y)) in enumerate(short_obs)
        obs[(:obs_x, i)] = x
        obs[(:obs_y, i)] = y
    end

    # Run inference
    samples = []
    accepted = 0

    try
        (trace, _) = generate(
            double_pendulum_short_window,
            (short_obs, short_times, L1, L2, noise_std, window),
            obs
        )

        for i in 1:num_samples
            (trace, acc) = mh(trace, select(:theta1_init, :theta2_init))
            accepted += acc
            push!(samples, (trace[:theta1_init], trace[:theta2_init]))
        end
    catch e
        println("  Short-window inference failed: $e")
    end

    acceptance_rate = num_samples > 0 ? accepted / num_samples : 0.0

    return samples, acceptance_rate
end

# =============================================================================
# STRATEGY 2b: Weighted-Window Inference (IMPROVED)
# =============================================================================

"""
Weighted likelihood model for double pendulum.
Uses ALL observations but weights them by time - early observations
have full weight, late observations have reduced weight.

This is BETTER than short-window because:
1. Uses all available data (not just first 0.3s)
2. Still focuses on early reliable data
3. Late observations provide soft constraints

Weight decay: w(t) = exp(-decay_rate * t / characteristic_time)
Effective noise: σ_effective = base_noise / sqrt(w(t))
"""
@gen function double_pendulum_weighted(observed_positions::Vector{Tuple{Float64, Float64}},
                                       obs_times::Vector{Float64},
                                       L1::Float64, L2::Float64,
                                       base_noise_std::Float64,
                                       decay_rate::Float64,
                                       characteristic_time::Float64)
    m1, m2, g = 1.0, 1.0, 9.81

    # Priors
    theta1_init = ({:theta1_init} ~ uniform(-pi/2, pi/2))
    theta2_init = ({:theta2_init} ~ uniform(-pi/2, pi/2))

    # Simulate
    params = DoublePendulumParams(L1, L2, m1, m2, g)
    initial_state = [theta1_init, theta2_init, 0.0, 0.0]

    trajectory = simulate_double_pendulum(params, initial_state;
                                          dt=0.01, duration=maximum(obs_times))
    bob2_traj = extract_bob2_trajectory(trajectory)

    dt = 0.01
    for (i, t) in enumerate(obs_times)
        idx = min(Int(round(t / dt)) + 1, length(bob2_traj))
        sim_x, sim_y = bob2_traj[idx]

        # Time-decaying weight: early observations have weight≈1, late have lower
        weight = exp(-decay_rate * t / characteristic_time)
        # Ensure weight doesn't go too low (prevents infinite noise)
        effective_noise = base_noise_std / sqrt(max(weight, 0.01))

        {(:obs_x, i)} ~ normal(sim_x, effective_noise)
        {(:obs_y, i)} ~ normal(sim_y, effective_noise)
    end

    return (theta1_init, theta2_init)
end

"""
Run weighted-window inference.
Uses time-decaying weights instead of hard cutoff.

Key insight: Uses ALL data but trusts early observations more.
- characteristic_time: time at which weight = exp(-decay_rate)
- decay_rate: how fast to decay weights (larger = faster decay)
"""
function run_weighted_window_inference(observed_positions::Vector{Tuple{Float64, Float64}},
                                       obs_times::Vector{Float64};
                                       num_samples::Int=500,
                                       base_noise_std::Float64=0.1,
                                       characteristic_time::Float64=0.5,
                                       decay_rate::Float64=1.0,
                                       L1::Float64=1.0, L2::Float64=1.0)
    obs = choicemap()
    for (i, (x, y)) in enumerate(observed_positions)
        obs[(:obs_x, i)] = x
        obs[(:obs_y, i)] = y
    end

    samples = Tuple{Float64, Float64}[]
    accepted = 0

    try
        local trace
        (trace, _) = generate(
            double_pendulum_weighted,
            (observed_positions, obs_times, L1, L2, base_noise_std,
             decay_rate, characteristic_time),
            obs
        )

        for i in 1:num_samples
            local acc
            (trace, acc) = mh(trace, select(:theta1_init, :theta2_init))
            accepted += acc
            push!(samples, (trace[:theta1_init], trace[:theta2_init]))
        end
    catch e
        println("  Weighted-window inference failed: $e")
    end

    return samples, accepted / max(1, num_samples)
end

# =============================================================================
# STRATEGY 3: Likelihood Tempering
# =============================================================================

"""
Tempered likelihood model for double pendulum.
Higher temperature = flatter likelihood = easier to explore.
"""
@gen function double_pendulum_tempered(observed_positions::Vector{Tuple{Float64, Float64}},
                                        obs_times::Vector{Float64},
                                        L1::Float64, L2::Float64,
                                        noise_std::Float64,
                                        temperature::Float64)
    m1, m2, g = 1.0, 1.0, 9.81

    theta1_init = ({:theta1_init} ~ uniform(-pi/2, pi/2))
    theta2_init = ({:theta2_init} ~ uniform(-pi/2, pi/2))

    params = DoublePendulumParams(L1, L2, m1, m2, g)
    initial_state = [theta1_init, theta2_init, 0.0, 0.0]

    duration = maximum(obs_times)
    trajectory = simulate_double_pendulum(params, initial_state;
                                          dt=0.01, duration=duration)
    bob2_traj = extract_bob2_trajectory(trajectory)

    # Tempered noise (effectively reduces likelihood sharpness)
    tempered_noise = noise_std * sqrt(temperature)

    dt = 0.01
    for (i, t) in enumerate(obs_times)
        idx = min(Int(round(t / dt)) + 1, length(bob2_traj))
        sim_x, sim_y = bob2_traj[idx]

        {(:obs_x, i)} ~ normal(sim_x, tempered_noise)
        {(:obs_y, i)} ~ normal(sim_y, tempered_noise)
    end

    return (theta1_init, theta2_init)
end

"""
Run tempered inference with annealing.
Start hot (easy to explore), then cool down to target.

FIXED VERSION: Collects samples from all cold stages (temp <= 10),
not just the final stage, to get more samples.
"""
function run_tempered_inference(observed_positions::Vector{Tuple{Float64, Float64}},
                                obs_times::Vector{Float64};
                                num_samples::Int=500,
                                noise_std::Float64=0.1,
                                L1::Float64=1.0, L2::Float64=1.0,
                                initial_temp::Float64=100.0,
                                final_temp::Float64=1.0,
                                cold_threshold::Float64=10.0)  # Collect samples when T <= this
    obs = choicemap()
    for (i, (x, y)) in enumerate(observed_positions)
        obs[(:obs_x, i)] = x
        obs[(:obs_y, i)] = y
    end

    # Annealing schedule - more stages for better exploration
    num_stages = 15
    temps = exp.(range(log(initial_temp), log(final_temp), length=num_stages))

    # Count cold stages to allocate samples
    num_cold_stages = count(t -> t <= cold_threshold, temps)
    samples_per_stage = num_cold_stages > 0 ? max(20, div(num_samples, num_cold_stages)) : 50

    all_samples = Tuple{Float64, Float64}[]
    current_theta1 = rand() * pi - pi/2
    current_theta2 = rand() * pi - pi/2

    for (stage, temp) in enumerate(temps)
        # Generate trace at current temperature
        constraints = choicemap()
        constraints[:theta1_init] = current_theta1
        constraints[:theta2_init] = current_theta2
        for (i, (x, y)) in enumerate(observed_positions)
            constraints[(:obs_x, i)] = x
            constraints[(:obs_y, i)] = y
        end

        try
            local trace
            (trace, _) = generate(
                double_pendulum_tempered,
                (observed_positions, obs_times, L1, L2, noise_std, temp),
                constraints
            )

            # Determine how many MCMC steps at this stage
            # Hot stages: fewer steps (just exploring)
            # Cold stages: more steps (collecting samples)
            steps = temp <= cold_threshold ? samples_per_stage : max(10, div(samples_per_stage, 3))

            # Run MCMC at this temperature
            for i in 1:steps
                local acc
                (trace, acc) = mh(trace, select(:theta1_init, :theta2_init))

                # FIX: Collect samples from cold stages (temp <= cold_threshold)
                if temp <= cold_threshold
                    push!(all_samples, (trace[:theta1_init], trace[:theta2_init]))
                end
            end

            # Update current state for next stage
            current_theta1 = trace[:theta1_init]
            current_theta2 = trace[:theta2_init]
        catch e
            # println("  Stage $stage (T=$(round(temp, digits=1))) failed: $e")
        end
    end

    return all_samples
end

# =============================================================================
# STRATEGY 4: Hierarchical Summary Matching
# =============================================================================

"""
Multi-scale summary matching.
Match both coarse (global) and fine (local) trajectory features.
"""
function run_hierarchical_matching(observed_positions::Vector{Tuple{Float64, Float64}},
                                   obs_times::Vector{Float64};
                                   num_samples::Int=500,
                                   L1::Float64=1.0, L2::Float64=1.0,
                                   coarse_weight::Float64=0.3,
                                   fine_weight::Float64=0.7)
    # Coarse summary: overall trajectory statistics
    obs_xs = [p[1] for p in observed_positions]
    obs_ys = [p[2] for p in observed_positions]

    coarse_obs = [mean(obs_xs), mean(obs_ys), std(obs_xs), std(obs_ys)]

    # Fine summary: early trajectory (first 2 observations)
    n_fine = min(2, length(observed_positions))
    fine_obs = vcat([[p[1], p[2]] for p in observed_positions[1:n_fine]]...)

    samples = []
    scores = Float64[]

    # Sample from prior and score
    for i in 1:num_samples * 10  # Oversample
        theta1 = rand() * pi - pi/2
        theta2 = rand() * pi - pi/2

        try
            params = DoublePendulumParams(L1, L2, 1.0, 1.0, 9.81)
            initial_state = [theta1, theta2, 0.0, 0.0]

            trajectory = simulate_double_pendulum(params, initial_state;
                                                  duration=maximum(obs_times))
            bob2_traj = extract_bob2_trajectory(trajectory)

            # Coarse score
            sim_xs = [p[1] for p in bob2_traj]
            sim_ys = [p[2] for p in bob2_traj]
            coarse_sim = [mean(sim_xs), mean(sim_ys), std(sim_xs), std(sim_ys)]
            coarse_dist = norm(coarse_sim .- coarse_obs)

            # Fine score (match early observations)
            dt = 0.01
            fine_sim = Float64[]
            for t in obs_times[1:n_fine]
                idx = min(Int(round(t / dt)) + 1, length(bob2_traj))
                push!(fine_sim, bob2_traj[idx][1])
                push!(fine_sim, bob2_traj[idx][2])
            end
            fine_dist = norm(fine_sim .- fine_obs)

            # Combined score
            total_score = coarse_weight * coarse_dist + fine_weight * fine_dist

            if length(samples) < num_samples || total_score < maximum(scores)
                push!(samples, (theta1, theta2))
                push!(scores, total_score)

                # Keep only best samples
                if length(samples) > num_samples
                    worst_idx = argmax(scores)
                    deleteat!(samples, worst_idx)
                    deleteat!(scores, worst_idx)
                end
            end
        catch
            # Skip failed simulations
        end
    end

    return samples, scores
end

# =============================================================================
# COMBINED: Best Custom Strategy
# =============================================================================

"""
Combined strategy for chaotic systems:
1. ABC to get initial region
2. Short-window refinement
3. Return best samples
"""
function run_combined_custom_inference_dp(observed_positions::Vector{Tuple{Float64, Float64}},
                                          obs_times::Vector{Float64};
                                          num_samples::Int=500,
                                          noise_std::Float64=0.1,
                                          L1::Float64=1.0, L2::Float64=1.0)
    all_samples = []

    # Phase 1: ABC-MCMC for rough exploration
    println("  Phase 1: ABC-MCMC...")
    abc_samples, abc_distances, abc_rate = run_abc_mcmc(
        observed_positions, obs_times;
        num_samples=div(num_samples, 3),
        epsilon=1.0,
        L1=L1, L2=L2
    )
    println("    ABC acceptance rate: $(round(abc_rate, digits=3))")
    append!(all_samples, abc_samples)

    # Phase 2: Short-window inference
    println("  Phase 2: Short-window inference...")
    sw_samples, sw_rate = run_short_window_inference(
        observed_positions, obs_times;
        num_samples=div(num_samples, 3),
        noise_std=noise_std,
        window=0.3,  # Very short window
        L1=L1, L2=L2
    )
    println("    Short-window acceptance rate: $(round(sw_rate, digits=3))")
    append!(all_samples, sw_samples)

    # Phase 3: Hierarchical matching
    println("  Phase 3: Hierarchical matching...")
    hier_samples, _ = run_hierarchical_matching(
        observed_positions, obs_times;
        num_samples=div(num_samples, 3),
        L1=L1, L2=L2
    )
    append!(all_samples, hier_samples)

    return all_samples
end

# =============================================================================
# EVALUATION METRICS
# =============================================================================

"""
Compute trajectory prediction error at observation times.
"""
function compute_trajectory_error(samples::Vector{Tuple{Float64, Float64}},
                                  observed_positions::Vector{Tuple{Float64, Float64}},
                                  obs_times::Vector{Float64},
                                  L1::Float64, L2::Float64)
    errors = Float64[]

    for (theta1, theta2) in samples
        try
            params = DoublePendulumParams(L1, L2, 1.0, 1.0, 9.81)
            initial_state = [theta1, theta2, 0.0, 0.0]

            trajectory = simulate_double_pendulum(params, initial_state;
                                                  duration=maximum(obs_times))
            bob2_traj = extract_bob2_trajectory(trajectory)

            dt = 0.01
            sample_error = 0.0
            for (i, t) in enumerate(obs_times)
                idx = min(Int(round(t / dt)) + 1, length(bob2_traj))
                sim_x, sim_y = bob2_traj[idx]
                obs_x, obs_y = observed_positions[i]

                sample_error += sqrt((sim_x - obs_x)^2 + (sim_y - obs_y)^2)
            end
            sample_error /= length(obs_times)

            push!(errors, sample_error)
        catch
            push!(errors, Inf)
        end
    end

    return errors
end

"""
Compute RMSE to true parameters.
"""
function compute_param_rmse(samples::Vector{Tuple{Float64, Float64}},
                            theta1_true::Float64, theta2_true::Float64)
    if isempty(samples)
        return (theta1_rmse=Inf, theta2_rmse=Inf)
    end

    theta1s = [s[1] for s in samples]
    theta2s = [s[2] for s in samples]

    theta1_rmse = sqrt(mean((theta1s .- theta1_true).^2))
    theta2_rmse = sqrt(mean((theta2s .- theta2_true).^2))

    return (theta1_rmse=theta1_rmse, theta2_rmse=theta2_rmse)
end

# =============================================================================
# FULL COMPARISON
# =============================================================================

"""
Run full comparison of all inference methods.
"""
function run_full_comparison(;theta1_true::Float64=pi/6,
                              theta2_true::Float64=pi/4,
                              L1::Float64=1.0, L2::Float64=1.0,
                              obs_duration::Float64=2.0,
                              noise_std::Float64=0.05,
                              num_samples::Int=300)
    println("=" ^ 60)
    println("DOUBLE PENDULUM MODEL: INFERENCE COMPARISON")
    println("=" ^ 60)

    # Generate observations
    params_true = DoublePendulumParams(L1, L2, 1.0, 1.0, 9.81)
    initial_state_true = [theta1_true, theta2_true, 0.0, 0.0]
    obs_times = collect(0.0:0.2:obs_duration)

    observed_positions, true_traj = generate_observations(
        params_true, initial_state_true, obs_times, noise_std
    )

    println("\nTrue parameters: θ1=$(round(rad2deg(theta1_true), digits=1))°, θ2=$(round(rad2deg(theta2_true), digits=1))°")
    println("Observation duration: $(obs_duration) s")
    println("Number of observations: $(length(observed_positions))")

    results = Dict()

    # 1. Baseline: Importance Sampling
    println("\n--- Importance Sampling (baseline) ---")
    is_samples = []
    for _ in 1:num_samples
        try
            (trace, _) = importance_resampling(
                double_pendulum_simple,
                (observed_positions, obs_times, L1, L2, noise_std),
                make_observations(observed_positions),
                50
            )
            push!(is_samples, (trace[:theta1_init], trace[:theta2_init]))
        catch
        end
    end
    println("  Got $(length(is_samples)) valid samples")
    results["IS (baseline)"] = is_samples

    # 2. Baseline: Metropolis-Hastings
    println("\n--- Metropolis-Hastings (baseline) ---")
    mh_samples = []
    try
        (trace, _) = generate(
            double_pendulum_simple,
            (observed_positions, obs_times, L1, L2, noise_std),
            make_observations(observed_positions)
        )
        mh_acc = 0
        for i in 1:num_samples
            (trace, acc) = mh(trace, select(:theta1_init, :theta2_init))
            mh_acc += acc
            push!(mh_samples, (trace[:theta1_init], trace[:theta2_init]))
        end
        println("  Acceptance rate: $(round(mh_acc/num_samples, digits=3))")
    catch e
        println("  MH failed: $e")
    end
    results["MH (baseline)"] = mh_samples

    # 3. Baseline: HMC
    println("\n--- HMC (baseline) ---")
    hmc_samples = []
    try
        (trace, _) = generate(
            double_pendulum_simple,
            (observed_positions, obs_times, L1, L2, noise_std),
            make_observations(observed_positions)
        )
        hmc_acc = 0
        for i in 1:num_samples
            (trace, acc) = hmc(trace, select(:theta1_init, :theta2_init);
                               L=3, eps=0.01)
            hmc_acc += acc
            push!(hmc_samples, (trace[:theta1_init], trace[:theta2_init]))
        end
        println("  Acceptance rate: $(round(hmc_acc/num_samples, digits=3))")
    catch e
        println("  HMC failed: $e")
    end
    results["HMC (baseline)"] = hmc_samples

    # 4. Custom: ABC-MCMC
    println("\n--- Custom: ABC-MCMC ---")
    abc_samples, abc_dists, abc_rate = run_abc_mcmc(
        observed_positions, obs_times;
        num_samples=num_samples,
        epsilon=0.8,
        L1=L1, L2=L2
    )
    println("  Acceptance rate: $(round(abc_rate, digits=3))")
    println("  Mean ABC distance: $(round(mean(abc_dists), digits=3))")
    results["ABC-MCMC"] = abc_samples

    # 5. Custom: Short-Window Inference
    println("\n--- Custom: Short-Window Inference ---")
    sw_samples, sw_rate = run_short_window_inference(
        observed_positions, obs_times;
        num_samples=num_samples,
        noise_std=noise_std,
        window=0.4,
        L1=L1, L2=L2
    )
    println("  Acceptance rate: $(round(sw_rate, digits=3))")
    results["Short-Window"] = sw_samples

    # 6. Custom: Tempered Inference
    println("\n--- Custom: Tempered Inference ---")
    temp_samples = run_tempered_inference(
        observed_positions, obs_times;
        num_samples=num_samples,
        noise_std=noise_std,
        L1=L1, L2=L2
    )
    println("  Got $(length(temp_samples)) samples")
    results["Tempered"] = temp_samples

    # 7. Custom: Hierarchical Matching
    println("\n--- Custom: Hierarchical Matching ---")
    hier_samples, hier_scores = run_hierarchical_matching(
        observed_positions, obs_times;
        num_samples=num_samples,
        L1=L1, L2=L2
    )
    println("  Mean score: $(round(mean(hier_scores), digits=3))")
    results["Hierarchical"] = hier_samples

    # 8. Custom: Combined Strategy
    println("\n--- Custom: Combined Strategy ---")
    combined_samples = run_combined_custom_inference_dp(
        observed_positions, obs_times;
        num_samples=num_samples,
        noise_std=noise_std,
        L1=L1, L2=L2
    )
    results["Combined Custom"] = combined_samples

    # Evaluate all methods
    println("\n" * "=" ^ 60)
    println("EVALUATION RESULTS")
    println("=" ^ 60)

    println("\n| Method | N Samples | θ1 Mean | θ2 Mean | θ1 RMSE | θ2 RMSE | Traj Error |")
    println("|--------|-----------|---------|---------|---------|---------|------------|")

    for (method, samples) in results
        n = length(samples)
        if n > 0
            theta1s = [rad2deg(s[1]) for s in samples]
            theta2s = [rad2deg(s[2]) for s in samples]

            rmse = compute_param_rmse(samples, theta1_true, theta2_true)
            traj_errors = compute_trajectory_error(samples, observed_positions, obs_times, L1, L2)
            mean_traj_error = mean(filter(isfinite, traj_errors))

            println("| $(rpad(method, 16)) | $(lpad(n, 9)) | $(lpad(round(mean(theta1s), digits=1), 7))° | $(lpad(round(mean(theta2s), digits=1), 7))° | $(lpad(round(rad2deg(rmse.theta1_rmse), digits=1), 7))° | $(lpad(round(rad2deg(rmse.theta2_rmse), digits=1), 7))° | $(lpad(round(mean_traj_error, digits=3), 10)) |")
        else
            println("| $(rpad(method, 16)) | $(lpad(0, 9)) | N/A | N/A | N/A | N/A | N/A |")
        end
    end

    println("\nTrue values: θ1=$(round(rad2deg(theta1_true), digits=1))°, θ2=$(round(rad2deg(theta2_true), digits=1))°")

    return results, observed_positions, obs_times
end

# =============================================================================
# MAIN
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    run_full_comparison()
end
