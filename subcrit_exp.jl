"""
This file is for the experiment for validating the subcritical point with power-law degree distribution.
"""

#Adjust the path to the environment and utils.jl as needed
script_dir = @__DIR__
env_dir = joinpath(script_dir, "env")
# ---------------------------------------------------------

using Pkg
Pkg.activate(env_dir)
include(joinpath(script_dir, "utils.jl"))  # Adjusted to look one directory up from script_dir
using Random 
using Distributions
using Plots
using Statistics
using LinearAlgebra
using JLD2
using Base.Threads
using Roots
using Combinatorics


function one_part(id, rK3, γ)
    Random.seed!(1234)
    println("Threads: ", Threads.nthreads())
    n = 20000
    mean_deg = round(Int, 0.025 * n)
    upbd = n-1
    f(k_min) = k_min -  mean_deg*(γ-2)/(γ-1)*((upbd/k_min)^(1-γ)-1)/((upbd/k_min)^(2-γ)-1)
    lwbd = round(Int, find_zeros(f,1,n)[1])


    A2, degree2, target = create_L_CL_PL_bound(n, γ, lwbd, upbd)
    A3, degree3 = create_3D(n, target)

    kp2 = Float64(mean(BigFloat.(degree2) .^ 2))
    kp = Float64(mean(BigFloat.(degree2)))

    K2Crit_mean = K2C * kp / kp2
    println("K2Crit mean field: ", K2Crit_mean)

    kpp = Float64(mean(BigFloat.(degree3)))
    kpp2 = Float64(mean(BigFloat.(degree3) .^ 2))
    kpp3 = Float64(mean(BigFloat.(degree3) .^ 3))
    kpp4 = Float64(mean(BigFloat.(degree3) .^ 4))
    K3Crit_mean = kpp4 * kpp^2 / (kpp2^2 * kpp3)
    println("K3c mean field: ", K3Crit_mean)

    #clear some memory

    kp = nothing;
    kpp = nothing;
    kpp2 = nothing;
    kpp3 = nothing;
    kpp4 = nothing;
    
    params = (n, degree2, A2, degree3, A3)
    sumdeg2 = sum(degree2)

    θ = rand(Uniform(0, 2π), n)
    ω = zeros(Float64, n)
    for i in 1:n 
        ω[i] = tan(π*(2i-n-1)/(n+1))
    end

     λ, Un = Un_gen(n, degree2, A2)
    den = den_cal(n, degree3, A3, Un)

    K2Crit = 2.0/λ
    println("K2Crit: ", K2Crit)
    K3Crit = λ * sum(Un .^ 4)/den * K2Crit
    println("K3Crit: ", K3Crit)

    k2 = 0.995 * K2Crit
    k3 = rK3 * K3Crit

    dt = 0.01
    Tend = 15.0
    time_steps = Int(Tend/dt)
    
    Ntest = 100
    R2_col = []
    R2_tat = []
    R2_fda = []

    # Pre-allocate these ONCE outside the loop
      
    
    for i in 1:Ntest
        percentage = i/Ntest
        println("Percentage: ", percentage * 100, " %")
        n_frac = round(Int, percentage * n)
        θ[1:n_frac] = ones(n_frac) * π
        Ravg =0.0

        Hn = zeros(Float64, n)
        Ψn = zeros(Float64, n)
        Rn = zeros(ComplexF64, n) 
        

        @time for t in 1:time_steps
            cal_order_param!(Hn, Ψn, Rn, k2, k3, θ, params)
            # θ = RK4_step(θ, ω, dt, Hn, Ψn, Rn, k2, k3, params)
            θ = RK4_step_fast!(θ, ω, dt, Hn, Ψn, Rn, k2, k3, params)

            R = 0.0 + 0.0*1im
            for i in 1:n
                R += Rn[i]
            end
            R /= sumdeg2;
            R = abs(R)

            if t > 0.3 * time_steps
                Ravg = (t - 0.3 * time_steps) * Ravg/(t - 0.3 * time_steps + 1) + R/ (t - 0.3 * time_steps + 1)
            end
        end
        push!(R2_col, Ravg)
        init_val = 0.01 * percentage
        println("Init val: ", init_val)
        rtat = R1_th_sum(init_val, k2 ,k3, ω, params)
        rfda = R1_th_Int(init_val, k2, k3, params)
        push!(R2_tat, rtat)
        push!(R2_fda, rfda)
    end
    println("NUM: ", mean(R2_col))
    println("NUM_std: ", std(R2_col))
    println("TAT: ", mean(R2_tat))
    println("TAT_std: ", std(R2_tat))
    println("FDA: ", mean(R2_fda))
    println("FDA_std: ", std(R2_fda))
    println("R2_col: ", R2_col)
    println("R2_tat: ", R2_tat)
    println("R2_fda: ", R2_fda)
    key_params = Dict("K2Crit" => K2Crit, "K3Crit" => K3Crit, "n" => n, "K2Crit_mean" => K2Crit_mean,
                      "K3Crit_mean" => K3Crit_mean, "Mean_deg" => mean(degree2), "Mean_deg3" => mean(degree3), "target_deg" => target,
                      "gamma" => γ, "rK3" => rK3)
    @save joinpath(script_dir, "results/bistat_pw_g_$(γ)_n_$(n)_ID_$(id)_rK3_$(rK3).jld2") R2_col R2_tat R2_fda key_params
end


# Read rK3 value from command line argument
const rK3_cli = if length(ARGS) >= 2
    parse(Float64, ARGS[1])
else
    error("Please provide rK3 as a command-line argument")
end

# Read rK3 value from command line argument
const Gamma_cli = if length(ARGS) >= 2
    parse(Float64, ARGS[2])
else
    error("Please provide Gamma as a command-line argument")
end

function main_single()
    id = 1282026
    println("Threads: ", Threads.nthreads())
    println("Running with rK3 = ", rK3_cli)
    println("Running with Gamma = ", Gamma_cli)
    one_part(id, rK3_cli, Gamma_cli)
end

main_single()

    