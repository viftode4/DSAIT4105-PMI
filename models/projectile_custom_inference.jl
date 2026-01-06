# =============================================================================
# Custom Inference Procedures for Projectile Model
# =============================================================================
#
# This file implements custom inference procedures tailored to the
# projectile/artillery inference problem.
#
# STRATEGIES:
# 1. Custom MH proposals that respect v0-theta correlation
# 2. Parallel tempering to jump between modes (high arc vs low arc)
# 3. Reparameterized model using (range, max_height) instead of (v0, theta)
#
# =============================================================================

using Gen
using LinearAlgebra
using Statistics
using Random

include("projectile_artillery.jl")

# =============================================================================
# STRATEGY 1: Custom Correlated Proposals
# =============================================================================

"""
Custom proposal that proposes along iso-range curves.
When we change theta, we adjust v0 to maintain similar range.

The range formula is: R = v0^2 * sin(2*theta) / g
To keep R constant when changing theta: v0_new = sqrt(R * g / sin(2*theta_new))
"""
@gen function correlated_proposal(current_trace)
    current_v0 = current_trace[:v0]
    current_theta = current_trace[:theta]

    # Current range (approximately)
    g = 9.81
    current_range = current_v0^2 * sin(2*current_theta) / g

    # Propose new theta with small perturbation
    theta_new = ({:theta} ~ normal(current_theta, 0.1))

    # Clamp to valid range
    theta_new = clamp(theta_new, deg2rad(10.0), deg2rad(80.0))

    # Adjust v0 to maintain similar range (with some noise)
    sin_2theta = sin(2*theta_new)
    if sin_2theta > 0.1  # Avoid division by small numbers
        v0_target = sqrt(abs(current_range * g / sin_2theta))
        v0_target = clamp(v0_target, 10.0, 50.0)
    else
        v0_target = current_v0
    end

    # Propose v0 near the target with small noise
    {:v0} ~ normal(v0_target, 2.0)

    return nothing
end

"""
Mode-switching proposal: proposes complementary angle to jump between modes.
If current angle is low (< 45°), propose high angle and vice versa.
"""
@gen function mode_switch_proposal(current_trace)
    current_v0 = current_trace[:v0]
    current_theta = current_trace[:theta]
    g = 9.81

    # Complementary angle: if theta gives range R, so does (90° - theta)
    # with the same velocity
    complementary_theta = deg2rad(90.0) - current_theta

    # Propose near the complementary angle
    {:theta} ~ normal(complementary_theta, 0.15)

    # Velocity adjustment (range is same for complementary angles at same v0)
    {:v0} ~ normal(current_v0, 3.0)

    return nothing
end

"""
Run MH with custom correlated proposals.
Mixes correlated proposals with occasional mode-switching proposals.
"""
function run_custom_mh(observed_impacts::Vector{Float64};
                       num_samples::Int=1000,
                       noise_std::Float64=1.0,
                       mode_switch_prob::Float64=0.2)
    observations = make_observations(observed_impacts)

    # Initialize
    (trace, _) = generate(projectile_model, (observed_impacts, noise_std), observations)

    samples = Tuple{Float64, Float64}[]
    accepted_corr = 0
    accepted_switch = 0
    total_corr = 0
    total_switch = 0

    for i in 1:num_samples
        if rand() < mode_switch_prob
            # Try mode-switching proposal
            (trace, accepted) = mh(trace, mode_switch_proposal, ())
            accepted_switch += accepted
            total_switch += 1
        else
            # Try correlated proposal
            (trace, accepted) = mh(trace, correlated_proposal, ())
            accepted_corr += accepted
            total_corr += 1
        end

        push!(samples, (trace[:v0], trace[:theta]))
    end

    acceptance_corr = total_corr > 0 ? accepted_corr / total_corr : 0.0
    acceptance_switch = total_switch > 0 ? accepted_switch / total_switch : 0.0

    return samples, (correlated=acceptance_corr, switch=acceptance_switch)
end

# =============================================================================
# STRATEGY 2: Parallel Tempering / Replica Exchange
# =============================================================================

"""
Tempered version of the projectile model.
Temperature T > 1 flattens the likelihood, making it easier to explore.
"""
@gen function projectile_model_tempered(observed_impacts::Vector{Float64},
                                        noise_std::Float64,
                                        temperature::Float64)
    # Same priors
    v0 = ({:v0} ~ uniform(10.0, 50.0))
    theta = ({:theta} ~ uniform(deg2rad(10.0), deg2rad(80.0)))

    # Simulate
    x_impact, _, _ = simulate_projectile(v0, theta)

    # Tempered likelihood: effectively increases noise by sqrt(temperature)
    tempered_noise = noise_std * sqrt(temperature)

    for (i, obs) in enumerate(observed_impacts)
        {(:impact, i)} ~ normal(x_impact, tempered_noise)
    end

    return (v0, theta, x_impact)
end

"""
Run parallel tempering with multiple temperature chains.
Chains can exchange states, allowing the hot chain to explore freely
and pass good states down to the cold chain.
"""
function run_parallel_tempering(observed_impacts::Vector{Float64};
                                num_samples::Int=1000,
                                noise_std::Float64=1.0,
                                temperatures::Vector{Float64}=[1.0, 2.0, 4.0, 8.0],
                                swap_interval::Int=10)
    num_chains = length(temperatures)
    observations = make_observations(observed_impacts)

    # Initialize chains at different temperatures
    traces = []
    for T in temperatures
        (trace, _) = generate(
            projectile_model_tempered,
            (observed_impacts, noise_std, T),
            observations
        )
        push!(traces, trace)
    end

    samples = Tuple{Float64, Float64}[]
    swaps_attempted = 0
    swaps_accepted = 0

    for iter in 1:num_samples
        # Update each chain
        for (i, T) in enumerate(temperatures)
            # Use broader proposals for hotter chains
            proposal_scale = sqrt(T)

            # MH step on each chain
            (traces[i], _) = mh(traces[i], select(:v0, :theta))
        end

        # Attempt swap between adjacent chains
        if iter % swap_interval == 0
            for i in 1:(num_chains-1)
                swaps_attempted += 1

                # Metropolis criterion for exchange
                score_i = get_score(traces[i])
                score_j = get_score(traces[i+1])
                T_i = temperatures[i]
                T_j = temperatures[i+1]

                # Swap acceptance probability
                log_alpha = (1/T_i - 1/T_j) * (score_j - score_i)

                if log(rand()) < log_alpha
                    # Swap the traces (need to regenerate at different temps)
                    swaps_accepted += 1

                    # Extract current values
                    v0_i = traces[i][:v0]
                    theta_i = traces[i][:theta]
                    v0_j = traces[i+1][:v0]
                    theta_j = traces[i+1][:theta]

                    # Regenerate traces with swapped values
                    constraints_i = choicemap()
                    constraints_i[:v0] = v0_j
                    constraints_i[:theta] = theta_j
                    for (k, obs) in enumerate(observed_impacts)
                        constraints_i[(:impact, k)] = obs
                    end

                    constraints_j = choicemap()
                    constraints_j[:v0] = v0_i
                    constraints_j[:theta] = theta_i
                    for (k, obs) in enumerate(observed_impacts)
                        constraints_j[(:impact, k)] = obs
                    end

                    (traces[i], _) = generate(
                        projectile_model_tempered,
                        (observed_impacts, noise_std, T_i),
                        constraints_i
                    )
                    (traces[i+1], _) = generate(
                        projectile_model_tempered,
                        (observed_impacts, noise_std, T_j),
                        constraints_j
                    )
                end
            end
        end

        # Collect sample from cold chain (T=1)
        push!(samples, (traces[1][:v0], traces[1][:theta]))
    end

    swap_rate = swaps_attempted > 0 ? swaps_accepted / swaps_attempted : 0.0

    return samples, swap_rate
end

# =============================================================================
# STRATEGY 3: Reparameterized Model
# =============================================================================

"""
Reparameterized projectile model using (range, height) instead of (v0, theta).
This parameterization has less correlation and more direct physical meaning.

From range R and max height H:
- theta = atan(4*H / R)
- v0 = sqrt(R * g / sin(2*theta))
"""
@gen function projectile_model_reparam(observed_impacts::Vector{Float64},
                                        noise_std::Float64=1.0)
    g = 9.81

    # Prior on range (meters) - physical units
    target_range = ({:range} ~ uniform(20.0, 150.0))

    # Prior on max height (meters) - physical units
    max_height = ({:height} ~ uniform(5.0, 100.0))

    # Convert to v0, theta
    # theta = atan(4*H / R)  [approximation from parabolic trajectory]
    theta = atan(4 * max_height / target_range)

    # Clamp theta to valid range
    theta = clamp(theta, deg2rad(10.0), deg2rad(80.0))

    # v0 from range formula: R = v0^2 * sin(2*theta) / g
    sin_2theta = sin(2*theta)
    if sin_2theta > 0.01
        v0 = sqrt(abs(target_range * g / sin_2theta))
    else
        v0 = 30.0  # Fallback
    end
    v0 = clamp(v0, 10.0, 50.0)

    # Simulate
    x_impact, _, _ = simulate_projectile(v0, theta)

    # Likelihood
    for (i, obs) in enumerate(observed_impacts)
        {(:impact, i)} ~ normal(x_impact, noise_std)
    end

    return (v0, theta, target_range, max_height, x_impact)
end

"""
Run inference with reparameterized model.
"""
function run_reparam_inference(observed_impacts::Vector{Float64};
                               num_samples::Int=1000,
                               noise_std::Float64=1.0)
    observations = make_observations(observed_impacts)

    # Initialize
    (trace, _) = generate(projectile_model_reparam, (observed_impacts, noise_std), observations)

    samples = Tuple{Float64, Float64}[]
    accepted = 0

    for i in 1:num_samples
        # MH on the reparameterized space
        (trace, acc) = mh(trace, select(:range, :height))
        accepted += acc

        # Extract transformed parameters
        retval = get_retval(trace)
        v0, theta = retval[1], retval[2]
        push!(samples, (v0, theta))
    end

    acceptance_rate = accepted / num_samples

    return samples, acceptance_rate
end

# =============================================================================
# COMBINED: Best Custom Inference Strategy
# =============================================================================

"""
Combined strategy:
1. Start with parallel tempering to find both modes
2. Then use correlated proposals for efficient local exploration
3. Periodically try mode-switching proposals
"""
function run_combined_custom_inference(observed_impacts::Vector{Float64};
                                       num_samples::Int=1000,
                                       noise_std::Float64=1.0,
                                       warmup_samples::Int=200)
    # Phase 1: Warmup with parallel tempering to find modes
    println("  Phase 1: Parallel tempering warmup...")
    pt_samples, swap_rate = run_parallel_tempering(
        observed_impacts;
        num_samples=warmup_samples,
        noise_std=noise_std,
        temperatures=[1.0, 2.0, 4.0, 8.0],
        swap_interval=5
    )
    println("    Swap rate: $(round(swap_rate, digits=3))")

    # Phase 2: Main sampling with custom proposals
    println("  Phase 2: Custom proposal sampling...")
    custom_samples, acceptance = run_custom_mh(
        observed_impacts;
        num_samples=num_samples - warmup_samples,
        noise_std=noise_std,
        mode_switch_prob=0.15
    )
    println("    Correlated acceptance: $(round(acceptance.correlated, digits=3))")
    println("    Mode-switch acceptance: $(round(acceptance.switch, digits=3))")

    # Combine samples
    all_samples = vcat(pt_samples, custom_samples)

    return all_samples
end

# =============================================================================
# EVALUATION METRICS
# =============================================================================

"""
Compute mode detection: does the sampler find both modes?
Returns fraction of samples in each mode (low arc vs high arc).
"""
function evaluate_mode_detection(samples::Vector{Tuple{Float64, Float64}})
    n = length(samples)
    if n == 0
        return (low_arc=0.0, high_arc=0.0)
    end

    low_arc = 0
    high_arc = 0

    for (v0, theta) in samples
        if rad2deg(theta) < 45.0
            low_arc += 1
        else
            high_arc += 1
        end
    end

    return (low_arc=low_arc/n, high_arc=high_arc/n)
end

"""
Compute effective sample size (ESS) using autocorrelation.
"""
function effective_sample_size(samples::Vector{Tuple{Float64, Float64}})
    n = length(samples)
    if n < 10
        return Float64(n)  # Return as Float64 to avoid NaN issues
    end

    # Use v0 for ESS calculation
    v0s = [s[1] for s in samples]
    v0_mean = mean(v0s)
    v0s_centered = v0s .- v0_mean

    # Handle zero variance case
    var_sum = sum(v0s_centered.^2)
    if var_sum < 1e-10
        return Float64(n)
    end

    # Autocorrelation at lag 1
    autocorr = sum(v0s_centered[1:end-1] .* v0s_centered[2:end]) / var_sum
    autocorr = clamp(autocorr, 0.0, 0.99)

    # ESS approximation
    ess = n * (1 - autocorr) / (1 + autocorr)
    return ess
end

"""
Compute RMSE to true parameters (if known).
"""
function compute_rmse(samples::Vector{Tuple{Float64, Float64}},
                      v0_true::Float64, theta_true::Float64)
    if isempty(samples)
        return (v0_rmse=Inf, theta_rmse=Inf)
    end

    v0s = [s[1] for s in samples]
    thetas = [s[2] for s in samples]

    v0_rmse = sqrt(mean((v0s .- v0_true).^2))
    theta_rmse = sqrt(mean((thetas .- theta_true).^2))

    return (v0_rmse=v0_rmse, theta_rmse=theta_rmse)
end

# =============================================================================
# FULL COMPARISON
# =============================================================================

"""
Run full comparison of all inference methods.
"""
function run_full_comparison(;v0_true::Float64=30.0,
                              theta_true::Float64=deg2rad(45.0),
                              n_observations::Int=5,
                              noise_std::Float64=2.0,
                              num_samples::Int=1000)
    println("=" ^ 60)
    println("PROJECTILE MODEL: INFERENCE COMPARISON")
    println("=" ^ 60)

    # Generate observations
    observed_impacts, true_impact = generate_observations(
        v0_true, theta_true, n_observations, noise_std
    )
    observations = make_observations(observed_impacts)

    println("\nTrue parameters: v0=$(v0_true), theta=$(round(rad2deg(theta_true), digits=1))°")
    println("True impact: $(round(true_impact, digits=1)) m")
    println("Observations: $(round.(observed_impacts, digits=1))")

    results = Dict()

    # 1. Baseline: Importance Sampling
    println("\n--- Importance Sampling (baseline) ---")
    is_samples = []
    for _ in 1:num_samples
        (trace, _) = importance_resampling(
            projectile_model,
            (observed_impacts, noise_std),
            observations,
            100
        )
        push!(is_samples, (trace[:v0], trace[:theta]))
    end
    results["IS (baseline)"] = is_samples

    # 2. Baseline: Metropolis-Hastings
    println("--- Metropolis-Hastings (baseline) ---")
    (trace, _) = generate(projectile_model, (observed_impacts, noise_std), observations)
    mh_samples = []
    mh_acc = 0
    for i in 1:num_samples
        (trace, acc) = mh(trace, select(:v0, :theta))
        mh_acc += acc
        push!(mh_samples, (trace[:v0], trace[:theta]))
    end
    println("  Acceptance rate: $(round(mh_acc/num_samples, digits=3))")
    results["MH (baseline)"] = mh_samples

    # 3. Baseline: HMC
    println("--- HMC (baseline) ---")
    (trace, _) = generate(projectile_model, (observed_impacts, noise_std), observations)
    hmc_samples = []
    hmc_acc = 0
    for i in 1:num_samples
        (trace, acc) = hmc(trace, select(:v0, :theta))
        hmc_acc += acc
        push!(hmc_samples, (trace[:v0], trace[:theta]))
    end
    println("  Acceptance rate: $(round(hmc_acc/num_samples, digits=3))")
    results["HMC (baseline)"] = hmc_samples

    # 4. Custom: Correlated Proposals
    println("\n--- Custom: Correlated Proposals ---")
    custom_mh_samples, custom_acc = run_custom_mh(
        observed_impacts; num_samples=num_samples, noise_std=noise_std
    )
    println("  Correlated acceptance: $(round(custom_acc.correlated, digits=3))")
    println("  Mode-switch acceptance: $(round(custom_acc.switch, digits=3))")
    results["Custom MH"] = custom_mh_samples

    # 5. Custom: Parallel Tempering
    println("\n--- Custom: Parallel Tempering ---")
    pt_samples, swap_rate = run_parallel_tempering(
        observed_impacts; num_samples=num_samples, noise_std=noise_std
    )
    println("  Swap rate: $(round(swap_rate, digits=3))")
    results["Parallel Tempering"] = pt_samples

    # 6. Custom: Reparameterized
    println("\n--- Custom: Reparameterized Model ---")
    reparam_samples, reparam_acc = run_reparam_inference(
        observed_impacts; num_samples=num_samples, noise_std=noise_std
    )
    println("  Acceptance rate: $(round(reparam_acc, digits=3))")
    results["Reparameterized"] = reparam_samples

    # 7. Custom: Combined Strategy
    println("\n--- Custom: Combined Strategy ---")
    combined_samples = run_combined_custom_inference(
        observed_impacts; num_samples=num_samples, noise_std=noise_std
    )
    results["Combined Custom"] = combined_samples

    # Evaluate all methods
    println("\n" * "=" ^ 60)
    println("EVALUATION RESULTS")
    println("=" ^ 60)

    println("\n| Method | ESS | Low Arc | High Arc | v0 RMSE | θ RMSE |")
    println("|--------|-----|---------|----------|---------|--------|")

    for (method, samples) in results
        modes = evaluate_mode_detection(samples)
        ess = effective_sample_size(samples)
        rmse = compute_rmse(samples, v0_true, theta_true)

        println("| $(rpad(method, 18)) | $(round(Int, ess)) | $(round(modes.low_arc, digits=2)) | $(round(modes.high_arc, digits=2)) | $(round(rmse.v0_rmse, digits=1)) | $(round(rad2deg(rmse.theta_rmse), digits=1))° |")
    end

    return results
end

# =============================================================================
# MAIN
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    run_full_comparison()
end
