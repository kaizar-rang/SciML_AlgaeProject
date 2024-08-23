using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, ComponentArrays, OptimizationOptimisers

# Parameters and initial conditions
N_days = 50  # Duration
datasize = 40  # Number of data points
A0 = 0.01  # Initial algae concentration
N0 = 1.0   # Initial nutrient concentration
u0 = [A0, N0]  # [Algae, Nutrient]
p0 = Float64[
    0.5,  # Increased r - Intrinsic growth rate
    1.0,  # K - Carrying capacity
    0.5,  # k - Half-saturation constant for nutrient uptake
    0.05, # Increased dN - Nutrient depletion rate
]
rng = Random.MersenneTwister(1234)

# Time span for simulation
tspan = (0.0, Float64(N_days))
t = range(tspan[1], tspan[2], length=datasize)

# Define the original ODE function for algal bloom growth (expanded)
function AlgalBloom!(du, u, p, t)
    A, N = u  # Algae and Nutrient
    r, K, k, dN = p  # Parameters: growth rate, carrying capacity, nutrient uptake, nutrient depletion

    # Expanded form of r * A * (1 - A / K) * (N / (N + k))
    term1 = r * A * (N / (N + k))  # r * A * N / (N + k)
    term2 = -r * (A^2 / K) * (N / (N + k))  # -r * (A^2 / K) * N / (N + k)
    du[1] = term1 + term2  # dA/dt = r * A * (N / (N + k)) - r * (A^2 / K) * (N / (N + k))

    # Nutrient depletion due to algae growth
    du[2] = -dN * A  # dN/dt - Nutrient depletion due to algae growth
end

# Solve the original ODE problem
prob_original = ODEProblem(AlgalBloom!, u0, tspan, p0)
sol_original = solve(prob_original, Tsit5(), saveat = t)

# Define the neural network
NN = Lux.Chain(Lux.Dense(1, 10, relu), Lux.Dense(10, 1))
p_nn, st_nn = Lux.setup(rng, NN)

# Wrap parameters into a ComponentArray
p0_vec = ComponentArray(p = p0, nn_params = p_nn)

# Modified ODE function with the neural network
function AlgalBloomUDE!(du, u, p, t)
    A, N = u  # Algae and Nutrient
    r, K, k, dN = p.p  # Parameters: growth rate, carrying capacity, nutrient uptake, nutrient depletion

    # Use neural network to replace the quadratic term A^2 / K
    NN_input = [A]  # Neural network takes A as input
    quadratic_term, st_nn = Lux.apply(NN, NN_input, p.nn_params, st_nn)  # Get the quadratic term from NN output

    # Algal growth with the neural network term replacing A^2 / K
    du[1] = r * A * (N / (N + k)) - r * quadratic_term[1] * (N / (N + k))  # dA/dt
    du[2] = -dN * A  # dN/dt - Nutrient depletion due to algae growth
end

# Define the ODE problem with the neural network (UDE)
prob_ude = ODEProblem(AlgalBloomUDE!, u0, tspan, p0_vec)

# Solve the UDE problem for comparison
sol_ude = solve(prob_ude, Tsit5(), saveat = t)

# Dummy data for algae (A) and nutrients (N) - replace with actual data
Algae_Data = sol_original[1, :]
Nutrient_Data = sol_original[2, :]

# Prediction function using the adjoint method
function predict_adjoint(θ)
    Array(solve(prob_ude, Tsit5(), p = θ, saveat = t, sensealg = InterpolatingAdjoint()))
end

# Loss function comparing model predictions to actual data
function loss_adjoint(θ)
    x = predict_adjoint(θ)
    loss = sum(abs2, (Algae_Data .- x[1,:])) + sum(abs2, (Nutrient_Data .- x[2,:]))
    return loss
end

# Callback function to monitor the training process
iter = 0
function callback2(θ, l)
    global iter
    iter += 1
    if iter % 100 == 0
        println("Epoch: $iter, Loss: $l")
    end
    return false
end

# Set up the optimization problem
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_adjoint(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p0_vec)
res = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.0001), callback = callback2, maxiters = 5000)

# Visualize the original and UDE model predictions
data_pred = predict_adjoint(res.u)

# Plotting original model results
plot(t, sol_original[1, :], label = "Original Algae Concentration A(t)", color=:blue)
plot!(t, sol_original[2, :], label = "Original Nutrient Concentration N(t)", color=:green)

# Plotting UDE model results
plot!(t, data_pred[1,:], label = "UDE Algae Prediction", color=:red, linestyle=:dash)
plot!(t, data_pred[2,:], label = "UDE Nutrient Prediction", color=:orange, linestyle=:dash)

xlabel!("Time (days)")
ylabel!("Concentration")
title!("Algal Bloom Model: Original vs UDE (Neural Network Term)")
