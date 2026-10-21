"""
This file is for the experiment for validating the change of subcritical depend on the triadic interaction structure.
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


function main(way)
    ID = rand(1:1000000)
    Random.seed!(1234)
    n = 2000
    mean_deg = 50
    Np = 2
    max_deg = 70
    min_deg = 30
    ps = [min_deg, max_deg]
    prbk = (mean_deg - min_deg)/(max_deg - min_deg)
    println("prbk: ", prbk)
    # ubd = 70
    # lwbd = 30
    target = Int[]
    for i in 1:n
        if rand() < prbk
            push!(target, ps[2])
        else
            push!(target, ps[1])
        end
    end

    A2, degree2, target = create_L_CL(n, target)
    A3, degree3 = create_3D(n, target)

    kp2 = Float64(mean(BigFloat.(degree2) .^ 2))
    kp = Float64(mean(BigFloat.(degree2)))

    K2Crit = K2C * kp / kp2
    println("K2Crit mean field: ", K2Crit)

    kpp = Float64(mean(BigFloat.(degree3)))
    kpp2 = Float64(mean(BigFloat.(degree3) .^ 2))
    kpp3 = Float64(mean(BigFloat.(degree3) .^ 3))
    kpp4 = Float64(mean(BigFloat.(degree3) .^ 4))
    K3Crit_mean = kpp4 * kpp^2 / (kpp2^2 * kpp3)

    println("K3c mean field: ", K3Crit_mean)

    params = (n, degree2, A2, degree3, A3)
    sumdeg2 = sum(degree2)
    println("mean_deg: ", sumdeg2/n)

    θ = rand(Uniform(0, 2π), n)
    θp = rand(Uniform(0, 2π), n)
    ω = zeros(Float64, n)
    ωp = zeros(Float64, n)
    for i in 1:n
        ω[i] = tan(π*(2i-n-1)/(n+1))
        ωp[i] = tan(π*(2i-n-1)/(n+1))
    end

    λ, Un = Un_gen(n, degree2, A2)
    Un = abs.(Un)
    den = den_cal(n, degree3, A3, Un)
    

    K2Crit = 2.0/λ
    K3Crit = λ * sum(Un .^ 4)/den * K2Crit

    new_A3, deg3p = rewire_3D_Ω(n, A3, degree3, Un, way)
    Rd = 20
    ir = 0
    while ir < Rd
        new_A3, deg3p = rewire_3D_Ω(n, new_A3, deg3p, Un, way)
        ir += 1
    end
    paramsp = (n, degree2, A2, deg3p, new_A3)
    den = den_cal(n, deg3p, new_A3, Un)
    K2Critp = 2.0/λ
    K3Critp = λ * sum(Un .^ 4)/den * K2Critp

    println("Preserve degree: ", all(degree3 .== deg3p))

    println("K3Crit: ", K3Crit)
    println("K3Critp: ", K3Critp)

    k2min = -0.5 * K2Crit
    k2max = 2.0 * K2Crit
    k2minp = -0.5 * K2Critp
    k2maxp = 2.0 * K2Critp
    dk = 0.005 * K2Crit
    dkp = 0.005 * K2Critp
    k2 = k2min
    k2p = k2minp
    if way == :decrease
        k3 = 2.5 * K3Critp
    else
        k3 = 2.5 * K3Crit
    end


    println("K3 = ", k3)
    println("K3C = ", K3Crit)

    dt = 0.05
    Tend = 50.0
    time_steps = Int(Tend/dt)

    k2rf = []
    R_tatf = []
    R_tatpf = []
    R_fdaf = []
    R_fdapf = []
    Rf_num = []
    Rf_nump = []
    Ravg = 0.0;
    Ravgp = 0.0;

    init_valt = 1e-3
    init_valtp =1e-3
    init_valf =1e-3
    init_valfp =1e-3

    println("-------- Forward ---------")
    while (k2 < k2max)
        for t in 1:time_steps
            Hn, Ψn, Rn = cal_order_param(k2, k3, θ, params)
            θ = Heun_step(θ, ω, dt, Hn, Ψn, Rn, k2, k3, params)
            Hnp, Ψnp, Rnp = cal_order_param(k2, k3, θp, paramsp)
            θp = Heun_step(θp, ω, dt, Hnp, Ψnp, Rnp, k2, k3, paramsp)

            R = 0.0 + 0.0*1im
            Rp = 0.0 + 0.0*1im
            for i in 1:n
                R += Rn[i]
                Rp += Rnp[i]
                # R1p += abs(Rn[i])
            end
            R /= sumdeg2;
            R = abs(R)
            Rp /= sumdeg2;
            Rp = abs(Rp)

            if t > 0.5 * time_steps
                Ravg = (t - 0.5 * time_steps) * Ravg/(t - 0.5 * time_steps + 1) + R/ (t - 0.5 * time_steps + 1)
                Ravgp = (t - 0.5 * time_steps) * Ravgp/(t - 0.5 * time_steps + 1) + Rp/ (t - 0.5 * time_steps + 1)
                # Ravg1 = (t - 0.5 * time_steps) * Ravg1/(t - 0.5 * time_steps + 1) + R1p/ (t - 0.5 * time_steps + 1)
            end
        end

        push!(Rf_num, Ravg)
        push!(Rf_nump, Ravgp)
        

        push!(k2rf, k2/K2Crit)
        Rt = R1_th_sum(init_valt, k2 ,k3, ω, params)
        # init_valt = Rt > 0 ? Rt : init_valt
        push!(R_tatf, Rt)
        Rtp = R1_th_sum(init_valtp, k2 ,k3, ω, paramsp)
        # init_valtp = Rtp > 0 ? Rtp : init_valtp
        push!(R_tatpf, Rtp)
        Rf = R1_th_Int(init_valf, k2, k3, params)
        # init_valf = Rf > 0 ? Rf : init_valf
        push!(R_fdaf, Rf)
        Rfp = R1_th_Int(init_valfp, k2, k3, paramsp)
        # init_valfp = Rfp > 0 ? Rfp : init_valfp
        push!(R_fdapf, Rfp)
        println("K2/K2Crit: ", k2/K2Crit)
        # println("R: ", Ravg, " Rp: ", Ravgp)
        println("TAT: ", Rt, " FDA: ", Rf)
        println("TATp: ", Rtp, " FDAp: ", Rfp)
        k2 += dk
        k2p += dkp
    end

    k2rb = []
    R_tatb = []
    R_tatpb = []
    R_fdab = []
    R_fdapb = []
    Rb_num = [] 
    Rb_nump = []

    init_valt = 10.0
    init_valtp = 10.0
    init_valf = 10.0
    init_valfp = 10.0
    println("-------- Backward ---------")
    while (k2 > k2min)
        for t in 1:time_steps
            Hn, Ψn, Rn = cal_order_param(k2, k3, θ, params)
            θ = Heun_step(θ, ω, dt, Hn, Ψn, Rn, k2, k3, params)
            Hnp, Ψnp, Rnp = cal_order_param(k2, k3, θp, paramsp)
            θp = Heun_step(θp, ω, dt, Hnp, Ψnp, Rnp, k2, k3, paramsp)

            R = 0.0 + 0.0*1im
            Rp = 0.0 + 0.0*1im
            for i in 1:n
                R += Rn[i]
                Rp += Rnp[i]
                # R1p += abs(Rn[i])
            end
            R /= sumdeg2;
            R = abs(R)
            Rp /= sumdeg2;
            Rp = abs(Rp)

            if t > 0.5 * time_steps
                Ravg = (t - 0.5 * time_steps) * Ravg/(t - 0.5 * time_steps + 1) + R/ (t - 0.5 * time_steps + 1)
                Ravgp = (t - 0.5 * time_steps) * Ravgp/(t - 0.5 * time_steps + 1) + Rp/ (t - 0.5 * time_steps + 1)
                # Ravg1 = (t - 0.5 * time_steps) * Ravg1/(t - 0.5 * time_steps + 1) + R1p/ (t - 0.5 * time_steps + 1)
            end
        end

        push!(Rb_num, Ravg)
        push!(Rb_nump, Ravgp)

        push!(k2rb, k2/K2Crit)
        Rt = R1_th_sum(init_valt, k2 ,k3, ω, params)
        init_valt = Rt
        push!(R_tatb, Rt)
        Rtp = R1_th_sum(init_valtp, k2 ,k3, ω, paramsp)
        init_valtp = Rtp
        push!(R_tatpb, Rtp)
        Rf = R1_th_Int(init_valf, k2, k3, params)
        init_valf = Rf
        push!(R_fdab, Rf)
        Rfp = R1_th_Int(init_valfp, k2, k3, paramsp)
        init_valfp = Rfp
        push!(R_fdapb, Rfp)
        println("K2/K2Crit: ", k2/K2Crit)
        # println("R: ", Ravg, " Rp: ", Ravgp)
        println("TAT: ", Rt, " FDA: ", Rf)
        println("TATp: ", Rtp, " FDAp: ", Rfp)
        k2 -= dk
        k2p -= dkp
    end

    keyparams = Dict("K2Crit" => K2Crit, "K3Crit" => K3Crit, "K2Critp" => K2Critp, "K3Critp" => K3Critp, "mean_deg" => mean_deg,  "max_deg" => max_deg,
                    "min_deg" => min_deg)
    
    println("ID: ", ID)
    @save "rewiring_3D_Ω_$(way)_ID_$(ID).jld2" k2rf R_tatf R_tatpf R_fdaf R_fdapf k2rb R_tatb R_tatpb R_fdab R_fdapb Rf_num Rb_num Rf_nump Rb_nump keyparams
end

#main(:increase)
main(:decrease)
