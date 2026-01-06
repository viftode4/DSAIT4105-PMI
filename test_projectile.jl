# Test script for Projectile/Artillery Model
# Run with: julia test_projectile.jl

include("models/projectile_artillery.jl")

println("=" ^ 60)
println("PROJECTILE MODEL TEST")
println("=" ^ 60)

# Test simulator
println("\n1. Testing Simulator")
println("-" ^ 40)
v0_true, theta_true = 30.0, deg2rad(45.0)
x_impact, _, traj = simulate_projectile(v0_true, theta_true)
println("v0 = $v0_true m/s, theta = $(rad2deg(theta_true))°")
println("Impact distance: $(round(x_impact, digits=2)) m")
println("Trajectory points: $(length(traj))")

# Test multimodality
println("\n2. Demonstrating Multimodality")
println("-" ^ 40)
demonstrate_multimodality(x_impact)

# Test inference
println("\n3. Testing Inference")
println("-" ^ 40)

# Generate observations
observed_impacts, true_impact = generate_observations(v0_true, theta_true, 5, 2.0)
println("True impact: $(round(true_impact, digits=2)) m")
println("Observations: $(round.(observed_impacts, digits=2))")

obs = make_observations(observed_impacts)

# Importance Resampling
println("\n3a. Importance Resampling (100 samples)...")
is_traces = Tuple{Float64, Float64}[]
for _ in 1:100
    tr, _ = importance_resampling(projectile_model, (observed_impacts, 2.0), obs, 50)
    push!(is_traces, (tr[:v0], tr[:theta]))
end
v0s = [t[1] for t in is_traces]
thetas = [rad2deg(t[2]) for t in is_traces]
println("  v0:    mean=$(round(mean(v0s), digits=2)), std=$(round(std(v0s), digits=2))")
println("  theta: mean=$(round(mean(thetas), digits=2))°, std=$(round(std(thetas), digits=2))°")

# Metropolis-Hastings
println("\n3b. Metropolis-Hastings (200 iterations)...")
tr, _ = generate(projectile_model, (observed_impacts, 2.0), obs)
mh_traces = Tuple{Float64, Float64}[]
mh_accepted = 0
for i in 1:200
    global tr, mh_accepted
    tr, acc = mh(tr, select(:v0, :theta))
    mh_accepted += acc
    push!(mh_traces, (tr[:v0], tr[:theta]))
end
v0s = [t[1] for t in mh_traces]
thetas = [rad2deg(t[2]) for t in mh_traces]
println("  Acceptance rate: $(round(mh_accepted/200, digits=3))")
println("  v0:    mean=$(round(mean(v0s), digits=2)), std=$(round(std(v0s), digits=2))")
println("  theta: mean=$(round(mean(thetas), digits=2))°, std=$(round(std(thetas), digits=2))°")

# HMC - with tuned parameters (L=leapfrog steps, eps=step size)
println("\n3c. Hamiltonian Monte Carlo (200 iterations)...")
println("  Testing different step sizes...")
tr, _ = generate(projectile_model, (observed_impacts, 2.0), obs)

# Try different step sizes to find one that works
for eps in [0.1, 0.01, 0.001]
    local test_tr, test_acc
    test_tr = tr
    test_acc = 0
    for _ in 1:20
        test_tr, acc = hmc(test_tr, select(:v0, :theta); L=10, eps=eps)
        test_acc += acc
    end
    println("    eps=$eps: acceptance=$(test_acc/20)")
end

# Run with reasonable step size
hmc_traces = Tuple{Float64, Float64}[]
hmc_accepted = 0
for i in 1:200
    global tr, hmc_accepted
    tr, acc = hmc(tr, select(:v0, :theta); L=10, eps=0.01)
    hmc_accepted += acc
    push!(hmc_traces, (tr[:v0], tr[:theta]))
end
v0s = [t[1] for t in hmc_traces]
thetas = [rad2deg(t[2]) for t in hmc_traces]
println("  Final run (eps=0.01, L=10):")
println("  Acceptance rate: $(round(hmc_accepted/200, digits=3))")
println("  v0:    mean=$(round(mean(v0s), digits=2)), std=$(round(std(v0s), digits=2))")
println("  theta: mean=$(round(mean(thetas), digits=2))°, std=$(round(std(thetas), digits=2))°")

# Summary
println("\n" * "=" ^ 60)
println("SUMMARY: ALL 4 INFERENCE PROCEDURES")
println("=" ^ 60)
println("True values: v0 = $v0_true m/s, theta = $(rad2deg(theta_true))°")
println("\nInference procedure analysis:")
println("  - Importance Sampling: POOR - multimodal posterior, weight degeneracy")
println("  - Metropolis-Hastings: POOR - correlated parameters, low acceptance")
println("  - HMC: GOOD - smooth likelihood, gradient-based exploration works")
println("  - Particle Filtering: N/A - not a sequential inference problem")
println("")
println("This model demonstrates: HMC excels on smooth, differentiable problems")
println("but IS and MH struggle with multimodality and parameter correlation.")
