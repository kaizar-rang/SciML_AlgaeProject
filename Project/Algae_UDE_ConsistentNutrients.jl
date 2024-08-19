using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, ComponentArrays, OptimizationOptimisers

# Initial conditions for algae biomass and nutrients
A0 = 0.01  # Initial algae concentration
N0 = 1.0   # Initial nutrient concentration
u0 = [A0, N0]  # [Algae, Nutrient]

# Adjusted parameters for the model
p0 = Float64[
    0.5,  # Intrinsic growth rate
    1.0,  # Carrying capacity
    0.5,  # Half-saturation constant for nutrient uptake
    0.05, # Nutrient depletion rate
    0.1   # Constant nutrient input rate
]

rng = Random.MersenneTwister(1234)

# Time span for simulation
N_days = 50
datasize = 40
tspan = (0.0, Float64(N_days))
t = range(tspan[1], tspan[2], length=datasize)

# Define the ODE function for algal bloom growth with continuous nutrient input
function AlgalBloom!(du, u, p, t)
    A, N = u  # Algae and Nutrient
    r, K, k, dN, inputN = p  # Parameters

    # Algal growth with nutrient limitation and continuous input
    du[1] = r * A * (1 - A / K) * (N / (N + k))  # dA/dt
    du[2] = -dN * A + inputN  # dN/dt
end

prob = ODEProblem(AlgalBloom!, u0, tspan, p0)
sol = solve(prob, Tsit5(), saveat=t)

# Extract the solution data (synthetic data)
Algae_Data = sol[1, :]  # Algae concentration data
Nutrient_Data = sol[2, :]  # Nutrient concentration data

plot(t, sol[1, :], label="Algae Concentration A(t)")
plot!(t, sol[2, :], label="Nutrient Concentration N(t)")
xlabel!("Time (days)")
ylabel!("Concentration")
title!("Compressed Algal Bloom Growth Model")

