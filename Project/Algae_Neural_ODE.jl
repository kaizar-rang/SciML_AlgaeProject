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
sol = solve(prob, Tsit5(), saveat=t)

# Plotting the results
plot(t, sol[1, :], label="Algae Concentration A(t)")
plot!(t, sol[2, :], label="Nutrient Concentration N(t)")
xlabel!("Time (days)")
ylabel!("Concentration")
title!("Compressed Algal Bloom Growth Model")

noise_level = 0.05
Algae_Noise = sol[1, :] .+ noise_level * randn(length(sol[1, :]))
Nutrient_Noise = sol[2, :] .+ noise_level * randn(length(sol[2, :]))

# Plotting the results with noise
plot(t, Algae_Noise, label="Algae Concentration A(t) with Noise", linestyle=:solid)
xlabel!("Time (days)")
ylabel!("Concentration")
title!("Compressed Algal Bloom Growth Model with Noise")








# Neural Network Architecture


dudt_nn = Lux.Chain(Lux.Dense(2, 50, tanh), Lux.Dense(50, 50, tanh), Lux.Dense(50, 2))
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
 
epoch_counter = Ref(0)

# Callback to observe training
callback = function (p, l, pred; doplot=true)
    epoch_counter[] += 1  # Increment the epoch counter
    display("Epoch: $(epoch_counter[]), Current Loss: $(l)")  # Display the epoch and the current loss
    if doplot
        plt = plot(t, sol[1, :], label="Data", title= "Prediction vs. Data", xlabel="Time (days)", ylabel="Concentration")
        plot!(plt, t, pred[1, :], linestyle=:dash, label="Prediction")
        display(plot(plt))
    end
    return false  # Return false to continue training
end

# Initialize parameters
pinit = ComponentArray(p_nn)

# Optimization setup
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

# Train the model
result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.001), callback=callback, maxiters=15000)


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