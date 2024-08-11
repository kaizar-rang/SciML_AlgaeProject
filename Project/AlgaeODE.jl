using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, ComponentArrays, OptimizationOptimisers

# Adjusted number of days to simulate and number of data points
N_days = 50  # Keep the original duration to see the full model behavior
datasize = 40  # Desired number of data points

# Initial conditions for algae biomass and nutrients
A0 = 0.01  # Initial algae concentration
N0 = 1.0   # Initial nutrient concentration
u0 = [A0, N0]  # [Algae, Nutrient]

# Adjusted parameters for the model to fit significant dynamics into fewer data points
p0 = Float64[
    0.5,  # Increased r - Intrinsic growth rate to accelerate algae growth
    1.0,  # K - Carrying capacity
    0.5,  # k - Half-saturation constant for nutrient uptake
    0.05, # Increased dN - Nutrient depletion rate to accelerate nutrient consumption
]

# Time span for simulation
tspan = (0.0, Float64(N_days))
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

# Plotting the results
plot(t, sol[1, :], label="Algae Concentration A(t)")
plot!(t, sol[2, :], label="Nutrient Concentration N(t)")
xlabel!("Time (days)")
ylabel!("Concentration")
title!("Compressed Algal Bloom Growth Model")

# Neural Network Architecture
dudt_nn = Lux.Chain(Lux.Dense(2, 50, relu), Lux.Dense(50, 2))
p_nn, st_nn = Lux.setup(rng, dudt_nn)

# Define the Neural ODE
proba_neuralode = NeuralODE(dudt_nn, tspan, Tsit5(), saveat=t)

# Prediction function
function predict_neuralode(p)
    Array(proba_neuralode(u0, p, st_nn)[1])
end

# Loss function
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, sol .- pred)
    return loss, pred
end

# Callback to observe training
callback = function (p, l, pred; doplot=true)
    display("Current Loss: $(l)")
    if doplot
        plt = plot(t, sol[1, :], label="Data", title="Prediction vs. Data", xlabel="Time (days)", ylabel="Concentration")
        plot!(plt, t, pred[1, :], linestyle=:dash, label="Prediction")
        display(plot(plt))
    end
    return false
end

# Initialize parameters
pinit = ComponentArray(p_nn)

# Optimization setup
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

# Train the model
result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0. 1), callback=callback, maxiters=1000)