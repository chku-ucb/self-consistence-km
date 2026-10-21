"""
This file is for the experiment for validating the critical coupling constant and hysteresis with uniform distribution.
"""

# Adjust the path to the environment and utils.jl as needed
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

function main(rK3)
    Random.seed!(1234)  # Set a random seed for reproducibility
    println("Threads: ", Threads.nthreads())
    n = 10000
    lwbd = round(Int, 0.015*n)
    upbd = round(Int, 0.025*n)

    A2, degree2, target = create_L_CL_UN(n, lwbd, upbd)
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

    k2min = 0.0 * K2Crit
    k2max = 2.0 * K2Crit
    dk = 0.1 * K2Crit
    k2 = k2min
    k3 = rK3 * K3Crit 

    dt = 0.01
    Tend = 30.0
    time_steps = Int(Tend/dt)

    k2rf = []
    R_tatf = []
    R_fdaf = []
    Rf_num = []
    init_valt = 5e-3
    init_valf = 5e-3

    Ravg = 0.0

    println("---------- Forward ------------")
    while (k2 < k2max)

        for t in 1:time_steps
            Hn, Ψn, Rn = cal_order_param(k2, k3, θ, params)
            θ = Heun_step(θ, ω, dt, Hn, Ψn, Rn, k2, k3, params)

            R = 0.0 + 0.0*1im
            for i in 1:n
                R += Rn[i]
            end
            R /= sumdeg2;
            R = abs(R)

            if t > 0.5 * time_steps
                Ravg = (t - 0.5 * time_steps) * Ravg/(t - 0.5 * time_steps + 1) + R/ (t - 0.5 * time_steps + 1)
            end
        end

        push!(Rf_num, Ravg)
        push!(k2rf, k2/K2Crit)
        Rt = R1_th_sum(init_valt, k2 ,k3, ω, params)
        push!(R_tatf, Rt)
        Rf = R1_th_Int(init_valf, k2, k3, params)
        push!(R_fdaf, Rf)
        println("K2/K2Crit: ", k2/K2Crit)
        println("R: ", Ravg)
        println("TAT: ", Rt, " FDA: ", Rf)
        k2 += dk
    end

    k2rb = []
    R_tatb = []
    R_fdab = []
    Rb_num = []
    init_valt = 5e-3
    init_valf = 5e-3

    println("---------- Backward ------------")
    while (k2 > k2min)

        for t in 1:time_steps
            Hn, Ψn, Rn = cal_order_param(k2, k3, θ, params)
            θ = Heun_step(θ, ω, dt, Hn, Ψn, Rn, k2, k3, params)

            R = 0.0 + 0.0*1im
            for i in 1:n
                R += Rn[i]
            end
            R /= sumdeg2;
            R = abs(R)

            if t > 0.5 * time_steps
                Ravg = (t - 0.5 * time_steps) * Ravg/(t - 0.5 * time_steps + 1) + R/ (t - 0.5 * time_steps + 1)
            end
        end

        push!(Rb_num, Ravg)
        push!(k2rb, k2/K2Crit)
        Rt = R1_th_sum(init_valt, k2 ,k3, ω, params)
        init_valt = Rt > 0 ? Rt : init_valt
        push!(R_tatb, Rt)
        Rf = R1_th_Int(init_valf, k2, k3, params)
        init_valf = Rf > 0 ? Rf : init_valf
        push!(R_fdab, Rf)
        println("K2/K2Crit: ", k2/K2Crit)
        println("R: ", Ravg)
        println("TAT: ", Rt, " FDA: ", Rf)
        k2 -= dk
    end

    # Save results
    key_params = Dict("K2Crit" => K2Crit, "K3Crit" => K3Crit, "n" => n, "K2Crit_mean" => K2Crit_mean,
                      "K3Crit_mean" => K3Crit_mean, "Mean_deg" => mean(degree2), "Mean_deg3" => mean(degree3), "target_deg" => target)
    ID = rand(1:1000000)
    @save joinpath(script_dir,"results/res_n_$(n)_ID_$(ID)_rK3_$(rK3).jld2") k2rf R_tatf R_fdaf Rf_num k2rb R_tatb R_fdab Rb_num key_params

end


main(0.5)
# main(1.0)
# main(1.5)
# main(2.0)
# main(2.5)