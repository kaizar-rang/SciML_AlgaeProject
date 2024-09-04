using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using ComponentArrays

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

# Extract time and data for training
Algae_Data = sol_original[1, :]
Nutrient_Data = sol_original[2, :]

# Visualize the original data
plot(t, Algae_Data, label = "Algae (A)", xlabel = "Time", ylabel = "Concentration")
plot!(t, Nutrient_Data, label = "Nutrient (N)")

# Define neural networks for the UDE
NN1 = Lux.Chain(Lux.Dense(1, 10, relu), Lux.Dense(10, 1))
p1, st1 = Lux.setup(rng, NN1)

p0_vec = (r = 0.5, K = 1.0, k = 0.5, dN = 0.05, layer_1 = p1)
p0_vec = ComponentArray(p0_vec)

# Modified ODE function with the neural network
function AlgalBloomUDE!(du, u, p, t)
    A, N = u  # Algae and Nutrient
    r, K, k, dN = p.r, p.K, p.k, p.dN  # Parameters: growth rate, carrying capacity, nutrient uptake, nutrient depletion

    # Neural network replaces the quadratic term A^2 / K
    NN_input = [A]  # Neural network takes A as input
    quadratic_term, _ = Lux.apply(NN1, NN_input, p.layer_1, st1)  # NN output for quadratic term

    # Algal growth with the neural network term replacing A^2 / K
    du[1] = r * A * (N / (N + k)) - r * quadratic_term[1] * (N / (N + k))  # dA/dt
    du[2] = -dN * A  # dN/dt - Nutrient depletion due to algae growth
end

# Initialize the ODE problem with neural network (UDE)
prob_ude = ODEProblem(AlgalBloomUDE!, u0, tspan, p0_vec)

# Prediction function using the adjoint method
function predict_adjoint(θ)
    Array(solve(prob_ude, Tsit5(), p=θ, saveat=t, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end

# Loss function comparing model predictions to actual data
function loss_adjoint(θ)
    x = predict_adjoint(θ)
    loss = sum(abs2, (Algae_Data .- x[1, :])) + sum(abs2, (Nutrient_Data .- x[2, :]))
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

# Train the neural networks within the UDE using the generated data
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_adjoint(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p0_vec)
res = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.001), callback = callback2, maxiters = 5000)

# Compare the predictions with the actual data
data_pred = predict_adjoint(res.u)

# Plot the results
plot(t, Algae_Data, label="Algae Data", color=:blue, alpha=0.5)
plot!(t, Nutrient_Data, label="Nutrient Data", color=:red, alpha=0.5)

plot!(t, data_pred[1, :], label="Algae Prediction", color=:blue)
plot!(t, data_pred[2, :], label="Nutrient Prediction", color=:red)

# Plot interaction term comparisons (optional)
true_interaction = p0[1] * Algae_Data .* (Nutrient_Data ./ (Nutrient_Data .+ p0[3]))
NN_output = [Lux.apply(NN1, [Algae_Data[i]], res.u.layer_1, st1)[1][1] for i in 1:length(Algae_Data)]

plot(t, true_interaction, label="True Interaction", color=:green, alpha=0.5)
plot!(t, NN_output, label="NN Interaction Prediction", color=:purple)




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