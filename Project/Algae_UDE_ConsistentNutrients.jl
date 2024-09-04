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
title!("Compressed Algal Bloom Growth Model with Unlimited Nutrients ")



function AlgalBloomUDE!(du, u, p, t)
    A = u[1]  # Algae
    N = u[2]  # Nutrient
    r, K, k, dN = p.p  # Parameters: growth rate, carrying capacity, nutrient uptake, nutrient depletion

    # Use neural network to directly output the term for the growth rate
    NN_input = [A]  # Neural network takes A as input
    growth_term, st_nn = Lux.apply(NN, NN_input, p.nn_params, p.st_nn)  # Get the growth term from NN output

    # Algal growth using the neural network output directly
    du[1] = r * A * (N / (N + k)) - growth_term[1] * (N / (N + k))  # dA/dt
    du[2] = -dN * A  # dN/dt - Nutrient depletion due to algae growth
end



# Define the ODE problem with the neural network (UDE)
prob_ude = ODEProblem(AlgalBloomUDE!, u0, tspan, p0_vec)

# Solve the UDE problem for comparison
sol_ude = solve(prob_ude, Tsit5(), saveat = t)


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
# plot(t, sol[1, :], label = "Original Algae Concentration A(t)", color=:blue)
# plot!(t, sol[2, :], label = "Original Nutrient Concentration N(t)", color=:green)

# Plotting UDE model results
plot!(t, data_pred[1,:], label = "UDE Algae Prediction", color=:red, linestyle=:dash)
plot!(t, data_pred[2,:], label = "UDE Nutrient Prediction", color=:orange, linestyle=:dash)

xlabel!("Time (days)")
ylabel!("Concentration")
title!("Algal Bloom Model: Original vs UDE (Neural Network Term)")


# Assuming `AlgalBloomUDE!` and `predict_adjoint` functions are defined
# and that `Algae_Data` and `Nutrient_Data` arrays are available from your initial full dataset run

# Helper function to partition data and run the UDE model
function train_and_forecast(train_percent)
    idx_split = floor(Int, length(t) * train_percent / 100)
    t_train = t[1:idx_split]
    t_forecast = t[idx_split+1:end]

    # Define the problem for the training phase
    u0_train = [Algae_Data[1], Nutrient_Data[1]]  # Using the initial condition from the actual data
    prob_train = ODEProblem(AlgalBloomUDE!, u0_train, (t[1], t[idx_split]), p0_vec)
    sol_train = solve(prob_train, Tsit5(), saveat = t_train)

    # Define the problem for the forecast phase
    u0_forecast = [sol_train[1,end], sol_train[2,end]]  # Using the last point of training as initial condition
    prob_forecast = ODEProblem(AlgalBloomUDE!, u0_forecast, (t[idx_split], t[end]), p0_vec)
    sol_forecast = solve(prob_forecast, Tsit5(), saveat = t_forecast)

    # Plotting
    plot(t, Algae_Data, label="Underlying Data", linestyle=:dash, color=:black)
    plot!(t_train, sol_train[1,:], label="Training Prediction (till $(train_percent)%)", color=:blue)
    plot!(t_forecast, sol_forecast[1,:], label="Forecasting Prediction", color=:red)
    xlabel!("Time (days)")
    ylabel!("Algae Concentration")
    title!("Training with $(train_percent)% of Data")
end

# Execute for each case
train_and_forecast(90)  # Case 1
train_and_forecast(70)  # Case 2
train_and_forecast(50)  # Case 3
train_and_forecast(30)  # Case 4
train_and_forecast(10)  # Case 5

display(plot())