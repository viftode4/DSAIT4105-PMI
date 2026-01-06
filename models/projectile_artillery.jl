# =============================================================================
# Model 1: Projectile/Artillery Trajectory Inference
# =============================================================================
#
# PROBLEM: Given observed impact positions, infer the initial velocity and
# launch angle of a projectile.
#
# SIMULATOR: 2D ballistic trajectory with gravity and optional air resistance
#
# WHY THIS MODEL?
# - Creates MULTIMODAL posteriors: same target can be hit with different
#   angle/velocity combinations (high arc vs low arc)
# - Correlated parameters: velocity and angle trade off against each other
# - Smooth, differentiable simulator (good for HMC)
# - Importance Sampling struggles with multimodality
# - MH struggles with correlated parameters without good proposals
# =============================================================================

using Gen
using LinearAlgebra
using Statistics
using Random

# =============================================================================
# SIMULATOR: 2D Projectile Motion
# =============================================================================

"""
Simulate projectile trajectory with gravity.
Returns (x_impact, y_impact, trajectory) where trajectory is array of (x,y) positions.

Parameters:
- v0: initial velocity (m/s)
- theta: launch angle (radians)
- x0, y0: initial position
- g: gravitational acceleration (default 9.81 m/s²)
- dt: time step for simulation
- max_time: maximum simulation time
"""
function simulate_projectile(v0::Real, theta::Real;
                             x0::Real=0.0, y0::Real=0.0,
                             g::Real=9.81, dt::Real=0.01,
                             max_time::Real=10.0)
    # Analytical solution for projectile motion (fully differentiable)
    # Time of flight: t = 2 * v0 * sin(theta) / g
    # Range: x = v0 * cos(theta) * t = v0^2 * sin(2*theta) / g

    vx = v0 * cos(theta)
    vy = v0 * sin(theta)

    # Time to hit ground (from y0=0)
    t_flight = 2 * vy / g

    # Impact position
    x_impact = x0 + vx * t_flight

    # For trajectory (only when called with concrete Float64 for plotting)
    trajectory = [(x0, y0)]  # Placeholder

    return (x_impact, zero(x_impact), trajectory)
end

# Version that returns full trajectory (for visualization, not inference)
function simulate_projectile_trajectory(v0::Float64, theta::Float64;
                                        x0::Float64=0.0, y0::Float64=0.0,
                                        g::Float64=9.81, dt::Float64=0.01)
    vx = v0 * cos(theta)
    vy = v0 * sin(theta)

    x, y = x0, y0
    trajectory = [(x, y)]

    while y >= 0.0
        x += vx * dt
        vy -= g * dt
        y += vy * dt
        push!(trajectory, (x, y))
    end

    return trajectory
end

"""
Simulate projectile with air resistance (more complex version).
Uses simple quadratic drag model.
"""
function simulate_projectile_with_drag(v0::Float64, theta::Float64;
                                       x0::Float64=0.0, y0::Float64=0.0,
                                       g::Float64=9.81, drag::Float64=0.01,
                                       dt::Float64=0.01, max_time::Float64=10.0)
    vx = v0 * cos(theta)
    vy = v0 * sin(theta)
    x, y = x0, y0

    trajectory = [(x, y)]
    t = 0.0

    while y >= 0.0 && t < max_time
        t += dt

        # Speed and drag force
        speed = sqrt(vx^2 + vy^2)
        if speed > 0
            drag_x = -drag * speed * vx
            drag_y = -drag * speed * vy
        else
            drag_x, drag_y = 0.0, 0.0
        end

        # Update velocities
        vx += drag_x * dt
        vy += (-g + drag_y) * dt

        # Update position
        x += vx * dt
        y += vy * dt

        push!(trajectory, (x, y))
    end

    # Interpolate impact point
    if length(trajectory) >= 2 && trajectory[end][2] < 0
        x_prev, y_prev = trajectory[end-1]
        x_curr, y_curr = trajectory[end]
        t_impact = y_prev / (y_prev - y_curr)
        x_impact = x_prev + t_impact * (x_curr - x_prev)
    else
        x_impact = trajectory[end][1]
    end

    return (x_impact, 0.0, trajectory)
end

# =============================================================================
# PROBABILISTIC MODEL
# =============================================================================

"""
Probabilistic model for projectile inference.

Given observed impact positions, infer initial velocity and launch angle.

This model demonstrates:
- Multimodality: Multiple (v0, theta) pairs can hit the same target
- Parameter correlation: Higher angle requires different velocity
"""
@gen function projectile_model(observed_impacts::Vector{Float64},
                               noise_std::Float64=1.0)
    # Prior on initial velocity (m/s) - reasonable range for artillery
    v0 = ({:v0} ~ uniform(10.0, 50.0))

    # Prior on launch angle (radians) - between 10° and 80°
    theta = ({:theta} ~ uniform(deg2rad(10.0), deg2rad(80.0)))

    # Simulate trajectory
    x_impact, _, _ = simulate_projectile(v0, theta)

    # Likelihood: observed impacts are noisy measurements
    for (i, obs) in enumerate(observed_impacts)
        {(:impact, i)} ~ normal(x_impact, noise_std)
    end

    return (v0, theta, x_impact)
end

"""
More challenging version with both velocity and angle uncertainty,
plus uncertain drag coefficient.
"""
@gen function projectile_model_with_drag(observed_impacts::Vector{Float64},
                                         noise_std::Float64=1.0)
    # Priors
    v0 = ({:v0} ~ uniform(10.0, 50.0))
    theta = ({:theta} ~ uniform(deg2rad(10.0), deg2rad(80.0)))
    drag = ({:drag} ~ uniform(0.0, 0.05))  # Drag coefficient

    # Simulate with drag
    x_impact, _, _ = simulate_projectile_with_drag(v0, theta; drag=drag)

    # Observations
    for (i, obs) in enumerate(observed_impacts)
        {(:impact, i)} ~ normal(x_impact, noise_std)
    end

    return (v0, theta, drag, x_impact)
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
Generate synthetic observations from known parameters.
"""
function generate_observations(v0_true::Float64, theta_true::Float64,
                               n_obs::Int, noise_std::Float64)
    x_impact, _, _ = simulate_projectile(v0_true, theta_true)
    observations = x_impact .+ randn(n_obs) .* noise_std
    return observations, x_impact
end

"""
Create observation choicemap for inference.
"""
function make_observations(impacts::Vector{Float64})
    obs = choicemap()
    for (i, impact) in enumerate(impacts)
        obs[(:impact, i)] = impact
    end
    return obs
end

"""
Run inference comparison across different methods.
"""
function run_inference_comparison(observed_impacts::Vector{Float64};
                                  num_samples::Int=1000,
                                  noise_std::Float64=1.0)
    observations = make_observations(observed_impacts)

    results = Dict()

    # 1. Importance Resampling
    println("Running Importance Resampling...")
    is_traces = []
    for _ in 1:num_samples
        (trace, _) = importance_resampling(
            projectile_model,
            (observed_impacts, noise_std),
            observations,
            100
        )
        push!(is_traces, (trace[:v0], trace[:theta]))
    end
    results["importance_sampling"] = is_traces

    # 2. Metropolis-Hastings
    println("Running Metropolis-Hastings...")
    (trace, _) = generate(projectile_model, (observed_impacts, noise_std), observations)
    mh_traces = []
    mh_accepted = 0
    for i in 1:num_samples
        (trace, accepted) = mh(trace, select(:v0, :theta))
        mh_accepted += accepted
        push!(mh_traces, (trace[:v0], trace[:theta]))
    end
    results["metropolis_hastings"] = mh_traces
    println("  MH acceptance rate: $(mh_accepted/num_samples)")

    # 3. HMC (if available for continuous parameters)
    println("Running HMC...")
    (trace, _) = generate(projectile_model, (observed_impacts, noise_std), observations)
    hmc_traces = []
    hmc_accepted = 0
    for i in 1:num_samples
        (trace, accepted) = hmc(trace, select(:v0, :theta))
        hmc_accepted += accepted
        push!(hmc_traces, (trace[:v0], trace[:theta]))
    end
    results["hmc"] = hmc_traces
    println("  HMC acceptance rate: $(hmc_accepted/num_samples)")

    return results
end

# =============================================================================
# DEMONSTRATION OF MULTIMODALITY
# =============================================================================

"""
Demonstrate that multiple (v0, theta) combinations can hit the same target.
This is the key challenge for inference.
"""
function demonstrate_multimodality(target_distance::Float64=50.0)
    println("\n=== Demonstrating Multimodality ===")
    println("Target distance: $target_distance m")
    println("\nSearching for (v0, theta) pairs that hit the target...")

    solutions = []

    # Grid search for solutions
    for v0 in 15.0:1.0:45.0
        for theta_deg in 15.0:1.0:75.0
            theta = deg2rad(theta_deg)
            x_impact, _, _ = simulate_projectile(v0, theta)

            if abs(x_impact - target_distance) < 1.0
                push!(solutions, (v0=v0, theta_deg=theta_deg, x_impact=x_impact))
            end
        end
    end

    println("\nFound $(length(solutions)) approximate solutions:")
    for sol in solutions
        println("  v0=$(sol.v0) m/s, θ=$(sol.theta_deg)°, impact=$(round(sol.x_impact, digits=2)) m")
    end

    return solutions
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

function main()
    println("=" ^ 60)
    println("PROJECTILE/ARTILLERY INFERENCE MODEL")
    println("=" ^ 60)

    # True parameters (unknown to inference)
    v0_true = 30.0      # m/s
    theta_true = deg2rad(45.0)  # 45 degrees

    println("\nTrue parameters:")
    println("  v0 = $v0_true m/s")
    println("  theta = $(rad2deg(theta_true))°")

    # Generate synthetic observations
    n_observations = 5
    noise_std = 2.0
    observed_impacts, true_impact = generate_observations(
        v0_true, theta_true, n_observations, noise_std
    )

    println("\nTrue impact distance: $(round(true_impact, digits=2)) m")
    println("Observed impacts: $(round.(observed_impacts, digits=2))")

    # Demonstrate multimodality
    demonstrate_multimodality(true_impact)

    # Run inference
    println("\n" * "=" ^ 60)
    println("RUNNING INFERENCE")
    println("=" ^ 60)

    results = run_inference_comparison(observed_impacts;
                                       num_samples=500,
                                       noise_std=noise_std)

    # Summary statistics
    println("\n" * "=" ^ 60)
    println("RESULTS SUMMARY")
    println("=" ^ 60)

    for (method, traces) in results
        v0s = [t[1] for t in traces]
        thetas = [rad2deg(t[2]) for t in traces]

        println("\n$method:")
        println("  v0:    mean=$(round(mean(v0s), digits=2)), std=$(round(std(v0s), digits=2))")
        println("  theta: mean=$(round(mean(thetas), digits=2))°, std=$(round(std(thetas), digits=2))°")
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
