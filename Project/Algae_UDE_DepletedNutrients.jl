using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, ComponentArrays, OptimizationOptimisers

# Adjusted number of days to simulate and number of data points
N_days = 50  # Keep the original duration to see the full model behavior
datasize = 40  # Desired number of data points

# Initial conditions for algae biomass and nutrients
A0 = 0.01  # Initial algae concentration
N0 = 1.0   # Initial nutrient concentration
u0 = [A0, N0]  # [Algae, Nutrient]

# Adjusted parameters for the model to fit significant dynamics into fewer data points, including continuous nutrient input
p0 = Float64[
    0.5,  # Increased r - Intrinsic growth rate to accelerate algae growth
    1.0,  # K - Carrying capacity
    0.5,  # k - Half-saturation constant for nutrient uptake
    0.05, # Increased dN - Nutrient depletion rate to accelerate nutrient consumption
    0.1   # Constant nutrient input rate
]

rng = Random.MersenneTwister(1234)

# Time span for simulation
tspan = (0.0, Float64(N_days))
t = range(tspan[1], tspan[2], length=datasize)

# Define the ODE function for algal bloom growth with continuous nutrient input
function AlgalBloom!(du, u, p, t)
    A, N = u  # Algae and Nutrient
    r, K, k, dN, inputN = p  # Updated parameters list to include nutrient input rate

    # Algal growth with nutrient limitation and continuous input
    du[1] = r * A * (1 - A / K) * (N / (N + k))  # dA/dt
    du[2] = -dN * A + inputN  # dN/dt - Nutrient depletion due to algae growth and constant input
end

prob = ODEProblem(AlgalBloom!, u0, tspan, p0)
sol = solve(prob, Tsit5(), saveat=t)

# Plotting the results
plot(t, sol[1, :], label="Algae Concentration A(t)")
plot!(t, sol[2, :], label="Nutrient Concentration N(t)")
xlabel!("Time (days)")
ylabel!("Concentration")
title!("Compressed Algal Bloom Growth Model")