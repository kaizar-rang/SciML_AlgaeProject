using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, ComponentArrays, OptimizationOptimisers

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

rng = Random.MersenneTwister(1234)



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
sol = solve(prob_ude, Tsit5(), p = res.u, saveat = t, sensealg = InterpolatingAdjoint(autojacvec=ForwardDiffVJP(true)))

AlageData = sol[1, :]
NutrientData = sol[2, :]

# Plotting the results
plot(t, sol[1, :], label="Algae Concentration A(t)")
plot!(t, sol[2, :], label="Nutrient Concentration N(t)")
xlabel!("Time (days)")
ylabel!("Concentration")
title!("Algal Bloom Growth Model with Limited Nutrients")



NN = Lux.Chain(Lux.Dense(2, 10, relu), Lux.Dense(10, 1))
p_nn, st_nn = Lux.setup(rng, NN)

# Wrap parameters into a ComponentArray
p0_vec = ComponentArray(p = p0, nn_params = p_nn)

# Modified ODE function with the neural network
function AlgalBloomUDE!(du, u, p, t)
    A, N = u  # Algae and Nutrient
    r, K, k, dN = p.p  # Parameters: growth rate, carrying capacity, nutrient uptake, nutrient depletion

    # Use neural network to replace the quadratic term
    NN_input = [A, N]
    quadratic_term = Lux.apply(NN, NN_input, p.nn_params, st_nn)[1][1]

    # Algal growth with the neural network term
    du[1] = r * A * quadratic_term * (N / (N + k))  # dA/dt
    du[2] = -dN * A  # dN/dt - Nutrient depletion due to algae growth
end

# Define the ODE problem
prob_ude = ODEProblem(AlgalBloomUDE!, u0, tspan, p0_vec)

# Prediction function using the adjoint method
function predict_adjoint(θ)
    Array(solve(prob_ude, Tsit5(), p = θ, saveat = t, sensealg = InterpolatingAdjoint()))
end

# Dummy data for algae (A) and nutrients (N) - replace with actual data
Algae_Data = sol[1,:]
Nutrient_Data = sol[2,:]

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

# Visualize the predictions
data_pred = predict_adjoint(res.u)
plot(t, Algae_Data, label = "Algae Data", color=:blue)
plot!(t, data_pred[1,:], label = "Algae Prediction", color=:red)
xlabel!("Time (days)")
ylabel!("Concentration")
title!("Algal Bloom Model with UDE (Neural Network Term)")

