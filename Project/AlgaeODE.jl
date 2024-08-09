using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using ComponentArrays, OptimizationOptimisers

# Number of days to simulate
N_days = 300 #test

# Initial conditions for algae biomass and nutrients
A0 = 0.01  # Initial algae concentration
N0 = 1.0   # Initial nutrient concentration
u0 = [A0, N0]  # [Algae, Nutrient]

# Parameters for the model
p0 = Float64[
    0.1,  # r - Intrinsic growth rate
    1.0,  # K - Carrying capacity
    0.5,  # k - Half-saturation constant for nutrient uptake
    0.01, # dN - Nutrient depletion rate
]

# Time span for simulation
tspan = (0.0, Float64(N_days))
datasize = N_days
t = range(tspan[1], tspan[2], length=datasize)

# Define the ODE function for algal bloom growth
function AlgalBloom!(du, u, p, t)
    A, N = u  # Algae and Nutrient
    r, K, k, dN = p  # Parameters: growth rate, carrying capacity, nutrient uptake, nutrient depletion

    # Algal growth with nutrient limitation
    du[1] = r * A * (1 - A / K) * (N / (N + k))  # dA/dt
    du[2] = -dN * A  # dN/dt - Nutrient depletion due to algae growth
end

# Define the problem
prob = ODEProblem(AlgalBloom!, u0, tspan, p0)

# Solve the ODE
sol = solve(prob, Tsit5(), saveat=t)

plot(t, sol[1, :], label="Algae Concentration A(t)")
plot!(t, sol[2, :], label="Nutrient Concentration N(t)")
xlabel!("Time (days)")
ylabel!("Concentration")
title!("Algal Bloom Growth Model")