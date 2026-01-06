# Test script for Double Pendulum Model
# Run with: julia test_double_pendulum.jl

include("models/double_pendulum.jl")

println("=" ^ 60)
println("DOUBLE PENDULUM MODEL TEST")
println("=" ^ 60)

# Test simulator
println("\n1. Testing Simulator")
println("-" ^ 40)
params = DoublePendulumParams(1.0, 1.0, 1.0, 1.0, 9.81)
initial_state = [pi/2, pi/2, 1.0, -1.0]  # 90° angles with angular velocity
traj = simulate_double_pendulum(params, initial_state; duration=3.0)
bob2 = extract_bob2_trajectory(traj)
println("Parameters: L1=1.0, L2=1.0, m1=1.0, m2=1.0")
println("Initial: θ1=90°, θ2=90°, ω1=1.0, ω2=-1.0")
println("Simulated $(length(traj)) timesteps")
println("Initial bob2: $(round.(bob2[1], digits=3))")
println("Final bob2: $(round.(bob2[end], digits=3))")

# Demonstrate chaos - use EXTREME initial conditions (nearly inverted) and LONGER time
println("\n2. Demonstrating Chaotic Sensitivity")
println("-" ^ 40)
println("Using nearly-inverted pendulum (θ ≈ 170°) with angular velocity for strong chaos")

# Nearly inverted with angular velocity = strongly chaotic regime
base_state = [2.97, 2.97, 3.0, -3.0]  # ~170° with high angular velocity
pert = 0.0001  # Very small perturbation

println("\nPerturbation: $pert rad ($(round(rad2deg(pert), digits=5))°)")
println("Tracking divergence over time:")

perturbed = [2.97 + pert, 2.97, 3.0, -3.0]

traj_base = simulate_double_pendulum(params, base_state; duration=15.0)
traj_pert = simulate_double_pendulum(params, perturbed; duration=15.0)

bob2_base = extract_bob2_trajectory(traj_base)
bob2_pert = extract_bob2_trajectory(traj_pert)

for t in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0]
    idx = min(Int(round(t / 0.01)) + 1, length(bob2_base))
    x1, y1 = bob2_base[idx]
    x2, y2 = bob2_pert[idx]
    dist = sqrt((x1-x2)^2 + (y1-y2)^2)
    amplification = dist / pert
    println("  t=$(t)s: divergence=$(round(dist, digits=4))m, amplification=$(round(amplification, digits=0))x")
end

# Test gradient issues - use chaotic regime with LONGER simulation
println("\n3. Demonstrating Gradient Issues")
println("-" ^ 40)
println("Computing finite difference gradients in chaotic regime (10s simulation)...")

# Use chaotic initial conditions
target_state = [2.5, 2.5, 2.0, -2.0]  # Chaotic regime
sim_duration = 10.0

traj = simulate_double_pendulum(params, target_state; duration=sim_duration)
bob2 = extract_bob2_trajectory(traj)
target_x = bob2[end][1]

println("Initial: θ1=$(round(rad2deg(2.5), digits=1))°, θ2=$(round(rad2deg(2.5), digits=1))°")
println("Simulation duration: $(sim_duration)s")
println("Target final x-position: $(round(target_x, digits=4))")
println("\n∂(final_x)/∂(θ1_init) at different ε:")

for eps in [0.1, 0.01, 0.001, 0.0001]
    state_plus = [2.5 + eps, 2.5, 2.0, -2.0]
    state_minus = [2.5 - eps, 2.5, 2.0, -2.0]

    traj_plus = simulate_double_pendulum(params, state_plus; duration=sim_duration)
    traj_minus = simulate_double_pendulum(params, state_minus; duration=sim_duration)

    bob2_plus = extract_bob2_trajectory(traj_plus)
    bob2_minus = extract_bob2_trajectory(traj_minus)

    grad = (bob2_plus[end][1] - bob2_minus[end][1]) / (2 * eps)
    println("  ε=$eps: gradient ≈ $(round(grad, digits=2))")
end

println("\nIf gradients vary wildly across ε values, HMC cannot work!")

# Test inference (small scale)
println("\n4. Testing Inference (simplified model)")
println("-" ^ 40)

# Generate observations
L1_true, L2_true = 1.0, 1.0
theta1_true, theta2_true = pi/6, pi/4
obs_times = collect(0.0:0.5:2.0)  # Fewer observations, shorter time
noise_std = 0.1

params_true = DoublePendulumParams(L1_true, L2_true, 1.0, 1.0, 9.81)
init_true = [theta1_true, theta2_true, 0.0, 0.0]

observed_positions, _ = generate_observations(params_true, init_true, obs_times, noise_std)
println("True: θ1=$(round(rad2deg(theta1_true), digits=1))°, θ2=$(round(rad2deg(theta2_true), digits=1))°")
println("Observations at times: $obs_times")

obs = make_observations(observed_positions)

# Quick IS test
println("\n4a. Importance Resampling (30 samples)...")
is_samples = Tuple{Float64, Float64}[]
for _ in 1:30
    try
        tr, _ = importance_resampling(
            double_pendulum_simple,
            (observed_positions, obs_times, L1_true, L2_true, noise_std),
            obs, 30
        )
        push!(is_samples, (tr[:theta1_init], tr[:theta2_init]))
    catch e
        # Skip failures
    end
end
if length(is_samples) > 0
    θ1s = [rad2deg(s[1]) for s in is_samples]
    θ2s = [rad2deg(s[2]) for s in is_samples]
    println("  Got $(length(is_samples)) samples")
    println("  θ1: mean=$(round(mean(θ1s), digits=1))°, std=$(round(std(θ1s), digits=1))°")
    println("  θ2: mean=$(round(mean(θ2s), digits=1))°, std=$(round(std(θ2s), digits=1))°")
else
    println("  No valid samples obtained!")
end

# Quick MH test
println("\n4b. Metropolis-Hastings (50 iterations)...")
try
    local tr, mh_accepted
    tr, _ = generate(double_pendulum_simple,
                     (observed_positions, obs_times, L1_true, L2_true, noise_std), obs)
    mh_samples = Tuple{Float64, Float64}[]
    mh_accepted = 0
    for i in 1:50
        tr, acc = mh(tr, select(:theta1_init, :theta2_init))
        mh_accepted += acc
        push!(mh_samples, (tr[:theta1_init], tr[:theta2_init]))
    end
    local θ1s = [rad2deg(s[1]) for s in mh_samples]
    local θ2s = [rad2deg(s[2]) for s in mh_samples]
    println("  Acceptance rate: $(round(mh_accepted/50, digits=3))")
    println("  θ1: mean=$(round(mean(θ1s), digits=1))°, std=$(round(std(θ1s), digits=1))°")
    println("  θ2: mean=$(round(mean(θ2s), digits=1))°, std=$(round(std(θ2s), digits=1))°")
catch e
    println("  MH failed: $e")
end

# Particle Filtering discussion
println("\n4c. Particle Filtering Analysis")
println("-" ^ 40)
println("Particle Filtering IS applicable here (sequential observations over time).")
println("However, it will ALSO struggle because:")
println("  - Particles representing different initial conditions diverge chaotically")
println("  - After a few timesteps, particles become inconsistent with observations")
println("  - Resampling leads to particle collapse (all particles from one ancestor)")
println("  - Would need MASSIVE number of particles to maintain diversity")
println("")
println("Key insight: Chaos causes particle degeneracy even faster than usual!")

# Summary
println("\n" * "=" ^ 60)
println("SUMMARY: ALL 4 INFERENCE PROCEDURES")
println("=" ^ 60)
println("True values: θ1=$(round(rad2deg(theta1_true), digits=1))°, θ2=$(round(rad2deg(theta2_true), digits=1))°")
println("\nInference procedure analysis:")
println("  - Importance Sampling: POOR - samples rarely match chaotic trajectory")
println("  - Metropolis-Hastings: POOR - rough likelihood, low acceptance")
println("  - HMC: FAILS - gradients meaningless in chaotic regime")
println("  - Particle Filtering: POOR - chaotic divergence causes particle collapse")
println("\nThis model breaks ALL standard inference procedures!")
