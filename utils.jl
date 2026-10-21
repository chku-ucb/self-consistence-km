using Random, Distributions, Plots, Statistics, LinearAlgebra, JLD2, Base.Threads, Roots, Combinatorics, StatsBase

println("Threads: ", nthreads())

const K2C = 2.0;

#################### UTILITIES ####################

"""
    CDF_power_bound(x, lwbd, upbd, γ)

Inverse CDF of a bounded power-law distribution on `[lwbd, upbd]` with exponent `γ`.
Maps `x ~ U(0,1)` to a power-law sample (`@.` gives element-wise support).
"""
@. function CDF_power_bound(x::Float64, lwbd::Int64, upbd::Int64, γ::Float64)
    return ((upbd^(1 - γ) - lwbd^(1 - γ)) * x + lwbd^(1 - γ))^(1 / (1 - γ))
end

"""
    CDF_pow_all(x, γ, mean_deg)

Inverse CDF of an unbounded power-law distribution with exponent `γ` whose
minimum is chosen so that the mean degree is `mean_deg`.
"""
@. function CDF_pow_all(x::Float64, γ::Float64, mean_deg::Union{Int,Int32})
    return mean_deg * (γ - 2) / (γ - 1) * x^(1 / (1 - γ))
end

"""
    safe_sqrt(x)

Square root that returns `0.0` for non-positive `x` to avoid domain errors.
"""
@. function safe_sqrt(x::Float64)
    return x > 0.0 ? sqrt(x) : 0.0
end

"""
    safe_log(x)

Natural logarithm that returns `0.0` for non-positive `x`.
"""
@. function safe_log(x::Float64)
    return x > 0.0 ? log(x) : 0.0
end

"""
    safe_div(x, y)

Division that returns `0.0` when the denominator `y` is non-positive.
"""
@. function safe_div(x::Float64, y::Float64)
    return y > 0.0 ? x / y : 0.0
end

"""
    CDF_power(x, lwbd, upbd, γ)

Scalar version of [`CDF_power_bound`](@ref).
"""
function CDF_power(x::Float64, lwbd::Int64, upbd::Int64, γ::Float64)
    return ((upbd^(1 - γ) - lwbd^(1 - γ)) * x + lwbd^(1 - γ))^(1 / (1 - γ))
end

"""
    vec_to_matrix(Adj_vec) -> Vector{Matrix{Int16}}

Convert a vector-of-tuples adjacency representation into a vector-of-matrices
representation used by the 3D Kuramoto solver. Shared helper to avoid code
duplication in `create_3D`, `create_3D_CF`, and `rewire_3D_Ω`.
"""
function vec_to_matrix(Adj_vec::Vector{Vector{Tuple{Int16,Int16}}}, n::Int)
    Adj = Vector{Matrix{Int16}}(undef, n)
    @inbounds for i in 1:n
        len = length(Adj_vec[i])
        M = Matrix{Int16}(undef, len, 2)
        for idx in 1:len
            M[idx, 1] = Adj_vec[i][idx][1]
            M[idx, 2] = Adj_vec[i][idx][2]
        end
        Adj[i] = M
    end
    return Adj
end

###################### NETWORK GENERATION ######################
# All Chung-Lu generators use the Miller-Hagberg O(N+M) algorithm.
# Reference: Miller & Hagberg, "Efficient Generation of Networks with
#            Given Expected Degrees", WAW 2011 (LNCS 6732, pp. 115-126).
#
# Core idea: sort weights in decreasing order, then for each source node u
# use a Geometric distribution to skip non-edges, jumping directly to the
# next edge candidate.  The acceptance/rejection step corrects for the
# varying probability across the sorted sequence.

"""
    fast_CL(n, target) -> Vector{Vector{Int32}}

Generate a Chung-Lu random graph on `n` nodes with expected-degree sequence
`target` in O(N+M) expected time using the Miller-Hagberg geometric-skip
algorithm.  Returns the adjacency list (undirected, no self-loops).
"""
function fast_CL(n::Int, target::AbstractVector{<:Real})
    Adj = [Int32[] for _ in 1:n]
    W = Float64(sum(target))
    if W == 0.0 || n == 0
        return Adj
    end

    rho = 1.0 / W
    order = sortperm(target, rev=true)
    seq = target[order]

    for u in 1:(n-1)
        v = u + 1
        factor = Float64(seq[u]) * rho
        p = min(Float64(seq[v]) * factor, 1.0)
        
        while v <= n && p > 0.0
            if p != 1.0
                r = rand()
                v += floor(Int, log(r) / log1p(-p))
            end
            if v <= n
                q = min(Float64(seq[v]) * factor, 1.0)
                if rand() < q / p
                    orig_u = order[u]
                    orig_v = order[v]
                    push!(Adj[orig_u], Int32(orig_v))
                    push!(Adj[orig_v], Int32(orig_u))
                end
                v += 1
                p = q
            end
        end
    end
    return Adj
end

"""Chung-Lu graph with uniform target degrees drawn from `lwbd:upbd`."""
function create_L_CL_UN(n::Int, lwbd::Int, upbd::Int)
    target = rand(lwbd:upbd, n)
    Adj = fast_CL(n, target)
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

"""Chung-Lu graph with uniform target degrees drawn from `lwbd:upbd`, sorted
ascending so node i carries the i-th smallest target degree."""
function create_L_CL_UN_sort(n::Int, lwbd::Int, upbd::Int)
    target = rand(lwbd:upbd, n)
    sort!(target)
    Adj = fast_CL(n, target)
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

"""Erdős-Rényi graph with uniform edge probability `p = mean_deg / n`,
generated in O(N+M) time via geometric skipping."""
function create_L_CL_ER(n::Int, mean_deg::Int)
    Adj = [Int32[] for _ in 1:n]
    p = mean_deg / n
    if p >= 1.0
        for i in 1:n, j in i+1:n
            push!(Adj[i], Int32(j))
            push!(Adj[j], Int32(i))
        end
    elseif p > 0.0
        for i in 1:n
            j = i + 1
            while j <= n
                r = rand()
                j += floor(Int, log(r) / log1p(-p))
                if j <= n
                    push!(Adj[i], Int32(j))
                    push!(Adj[j], Int32(i))
                    j += 1
                end
            end
        end
    end
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree
end

"""Chung-Lu graph with power-law target degrees (bounded)."""
function create_L_CL_PL_bound(n::Int, γ::Float64, lwbd::Int, upbd::Int)
    target = [round(Int32, CDF_power_bound(rand(Uniform(0.0, 1.0)), lwbd, upbd, γ)) for i in 1:n]
    Adj = fast_CL(n, target)
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

"""Chung-Lu graph with power-law target degrees (unbounded)."""
function create_L_CL_PL(n::Int, γ::Float64, mean_deg::Union{Int,Int32})
    target = [round(Int32, CDF_pow_all(rand(Uniform(0.0, 1.0)), γ, mean_deg)) for i in 1:n]
    Adj = fast_CL(n, target)
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

"""Chung-Lu graph with target degrees drawn from a discrete set `ps`."""
function create_L_CL_NP(n::Int, Np::Int, ps::Vector{Int})
    target = [ps[rand(1:Np)] for _ in 1:n]
    Adj = fast_CL(n, target)
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

"""Chung-Lu graph with an explicit target degree sequence."""
function create_L_CL(n::Int, target::Vector{Int})
    Adj = fast_CL(n, target)
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

#Configuration model

"""Configuration model: stub-matching with retry loop (up to 1000 rounds)."""
function create_L_CF(n::Int, target::Vector{Int})
    Adj = [Int32[] for _ in 1:n]

    if sum(target) % 2 != 0
        target[rand(1:n)] += 1
    end
    rd = 0
    remain_stubs = copy(target)
    while any(remain_stubs .> 0) && rd < 1000
        current_stubs = Int32[]
        sizehint!(current_stubs, sum(remain_stubs))
        for i in 1:n
            for _ in 1:remain_stubs[i]
                push!(current_stubs, Int32(i))
            end
        end
        shuffle!(current_stubs)

        k = 1
        while k <= length(current_stubs) - 1
            u = current_stubs[k]
            v = current_stubs[k+1]
            if u != v
                push!(Adj[u], v)
                push!(Adj[v], u)
                remain_stubs[u] -= 1
                remain_stubs[v] -= 1
            end
            k += 2
        end
        rd += 1
    end

    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

################## Triangle generation ##################

"""
    create_3D(n, target) -> (Adj, degree)

Generate 3-uniform hyperedges (triangles) using the Chung-Lu model with
Miller-Hagberg geometric skipping on the innermost loop.

Complexity: O(N² + M₃) expected time, where M₃ is the number of hyperedges.
This is a dramatic improvement over the naïve O(N³) triple-nested loop.
"""
function create_3D(n::Int, target::Union{Vector{Int},Vector{Int32}})
    Adj_vec = [Tuple{Int16, Int16}[] for _ in 1:n]
    sumdeg = Float64(sum(target))
    
    if sumdeg > 0.0 && n >= 3
        order = sortperm(target, rev=true)
        seq = target[order]
        
        for i in 1:(n-2)
            for j in (i+1):(n-1)
                factor = 2.0 * Float64(seq[i]) * Float64(seq[j]) / (sumdeg * sumdeg)
                k = j + 1
                p = min(Float64(seq[k]) * factor, 1.0)
                
                while k <= n && p > 0.0
                    if p != 1.0
                        r = rand()
                        k += floor(Int, log(r) / log1p(-p))
                    end
                    
                    if k <= n
                        q = min(Float64(seq[k]) * factor, 1.0)
                        if rand() < q / p
                            oi, oj, ok = order[i], order[j], order[k]
                            push!(Adj_vec[oi], (Int16(oj), Int16(ok)))
                            push!(Adj_vec[oi], (Int16(ok), Int16(oj)))
                            push!(Adj_vec[oj], (Int16(oi), Int16(ok)))
                            push!(Adj_vec[oj], (Int16(ok), Int16(oi)))
                            push!(Adj_vec[ok], (Int16(oi), Int16(oj)))
                            push!(Adj_vec[ok], (Int16(oj), Int16(oi)))
                        end
                        k += 1
                        p = q
                    end
                end
            end
        end
    end
    
    Adj = vec_to_matrix(Adj_vec, n)
    degree = [Int32(size(Adj[i], 1) ÷ 2) for i in 1:n]
    return Adj, degree
end

"""Configuration model for 3-uniform hyperedges with stub-matching."""
function create_3D_CF(n::Int, target::Union{Vector{Int},Vector{Int32}})
    Adj_vec = [Tuple{Int16, Int16}[] for _ in 1:n]
    if sum(target) % 3 == 2
        target[rand(1:n)] += 1
        target[rand(1:n)] += 1
    elseif sum(target) % 3 == 1
        target[rand(1:n)] += 1
    end
    rd = 0
    remain_stubs = copy(target)
    while any(remain_stubs .> 0) && rd < 1000
        current_stubs = Int32[]
        sizehint!(current_stubs, sum(remain_stubs))
        for i in 1:n
            for _ in 1:remain_stubs[i]
                push!(current_stubs, Int32(i))
            end
        end
        shuffle!(current_stubs)

        k = 1
        while k <= length(current_stubs) - 2
            u = current_stubs[k]
            v = current_stubs[k+1]
            w = current_stubs[k+2]
            if u != v && u != w && v != w
                push!(Adj_vec[u], (Int16(v), Int16(w)))
                push!(Adj_vec[u], (Int16(w), Int16(v)))
                push!(Adj_vec[v], (Int16(u), Int16(w)))
                push!(Adj_vec[v], (Int16(w), Int16(u)))
                push!(Adj_vec[w], (Int16(u), Int16(v)))
                push!(Adj_vec[w], (Int16(v), Int16(u)))

                remain_stubs[u] -= 1
                remain_stubs[v] -= 1
                remain_stubs[w] -= 1
            end
            k += 3
        end
        rd += 1
    end
    
    Adj = vec_to_matrix(Adj_vec, n)
    degree = [Int32(size(Adj[i], 1) ÷ 2) for i in 1:n]
    return Adj, degree
end

############### 3BD Kuramoto Coupling CONSTANT #################

"""Compute leading eigenvalue and eigenvector of the adjacency matrix using sparse power iteration."""
function Un_gen(n::Int, degree2::AbstractVector, A2::AbstractVector; tol::Float64=1e-10, maxiter::Int=5000)
    u = ones(Float64, n)
    u ./= norm(u)
    v = similar(u)
    lambda_old = 0.0
    lambda = 0.0
    
    for iteration in 1:maxiter
        Threads.@threads :static for i in 1:n
            s = 0.0
            @inbounds for j in A2[i]
                s += u[j]
            end
            v[i] = s
        end
        lambda = dot(u, v)
        nv = norm(v)
        nv == 0.0 && error("Network has no edges.")
        v ./= nv
        if abs(lambda - lambda_old) < tol * max(1.0, abs(lambda))
            u .= v
            break
        end
        u, v = v, u
        lambda_old = lambda
    end
    
    Threads.@threads :static for i in 1:n
        s = 0.0
        @inbounds for j in A2[i]
            s += u[j]
        end
        v[i] = s
    end
    lambda = dot(u, v) / dot(u, u)
    
    # L2-normalize to exactly match the output of eigen()
    u ./= norm(u)
    
    return lambda, u
end

"""Compute numerator for K3c using adjacency matrix (matrix form)."""
function num_cal(N, Un, Adj2)
    # Vectorised: num = Uᵀ · diag(U) · A · diag(U²) · A · U
    # kept explicit for clarity & thread safety
    num = 0.0
    @inbounds for i in 1:N
        for j in 1:N
            aij = Adj2[i, j]
            aij == 0 && continue          # skip zero entries
            uiuj2 = Un[i] * Un[j]^2
            for k in 1:N
                num += uiuj2 * Un[k] * aij * Adj2[j, k]
            end
        end
    end
    return num
end

"""
    num_cal(Un)

Sum of the fourth powers of the leading-eigenvector entries, Σᵢ uᵢ⁴.
"""
function num_cal(Un)
    return sum(Un .^ 4)
end

"""Compute denominator for K3c using hyperedge adjacency."""
function den_cal(N, deg3, A3, Un)
    den = 0.0
    @inbounds for i in 1:N
        for j in 1:size(A3[i], 1)
            den += Un[i] * Un[A3[i][j, 1]]^2 * Un[A3[i][j, 2]]
        end
    end
    return den
end

###################### 3BD Kuramoto Model ######################

"""
    dthdt(θi, ωi, Hi, ψi)

Phase velocity of oscillator `i` in the local mean-field form:
dθᵢ/dt = ωᵢ - Hᵢ sin(θᵢ - ψᵢ).
"""
@inline function dthdt(θi::Float64, ωi::Float64, Hi::Float64, ψi::Float64)
    return ωi - Hi * sin(θi - ψi)
end

"""
    cal_order_param(K2, K3, θ, params) -> (Hn, ψn, Rn)

Local mean field of every node from dyadic (`K2`) and triadic (`K3`) coupling:
Hᵢ e^{iψᵢ} = K2 Σⱼ Aᵢⱼ e^{iθⱼ} + K3 Σ_(j,k) e^{i(2θⱼ - θₖ)}.
`Rn[i]` is the dyadic part Σⱼ Aᵢⱼ e^{iθⱼ}. Allocating version of [`cal_order_param!`](@ref).
"""
function cal_order_param(K2::Float64, K3::Float64, θ::Vector{Float64}, params::Any)
    n, degree2, A2, degree3, A3 = params
    Hn = zeros(Float64, n)
    ψn = zeros(Float64, n)
    Rn = zeros(ComplexF64, n)
    @threads :dynamic for i in 1:n
        re2, im2 = 0.0, 0.0
        re3, im3 = 0.0, 0.0

        @inbounds for j in 1:degree2[i]
            idx = A2[i][j]
            angle_val = θ[idx]
            s, c = sincos(angle_val)
            re2 += c
            im2 += s
        end

        @inbounds adj3 = A3[i]
        @inbounds for j in 1:size(adj3, 1)
            idx1 = adj3[j, 1]
            idx2 = adj3[j, 2]
            angle_val = 2.0 * θ[idx1] - θ[idx2]
            s, c = sincos(angle_val)
            re3 += c
            im3 += s
        end

        total_re = K2 * re2 + K3 * re3
        total_im = K2 * im2 + K3 * im3

        @inbounds Hn[i] = sqrt(total_re^2 + total_im^2)
        @inbounds ψn[i] = atan(total_im, total_re)
        @inbounds Rn[i] = ComplexF64(re2, im2)
    end
    return Hn, ψn, Rn
end


"""
    cal_order_param!(Hn, ψn, Rn, K2, K3, θ, params)

In-place version of [`cal_order_param`](@ref): writes into the pre-allocated `Hn`, `ψn`, `Rn`.
"""
function cal_order_param!(Hn::Vector{Float64}, ψn::Vector{Float64}, Rn::Vector{ComplexF64},
    K2::Float64, K3::Float64, θ::Vector{Float64}, params::Any)
    n, degree2, A2, degree3, A3 = params

    @threads :dynamic for i in 1:n
        re2, im2 = 0.0, 0.0
        re3, im3 = 0.0, 0.0

        @inbounds for j in 1:degree2[i]
            idx = A2[i][j]
            angle_val = θ[idx]
            s, c = sincos(angle_val)
            re2 += c
            im2 += s
        end

        @inbounds adj3 = A3[i]
        @inbounds for j in 1:size(adj3, 1)
            idx1 = adj3[j, 1]
            idx2 = adj3[j, 2]
            angle_val = 2.0 * θ[idx1] - θ[idx2]
            s, c = sincos(angle_val)
            re3 += c
            im3 += s
        end

        total_re = K2 * re2 + K3 * re3
        total_im = K2 * im2 + K3 * im3

        @inbounds Hn[i] = sqrt(total_re^2 + total_im^2)
        @inbounds ψn[i] = atan(total_im, total_re)
        @inbounds Rn[i] = ComplexF64(re2, im2)
    end
    return nothing
end


"""
    RK4_step(θ, ω, dt, Hn, Ψn, Rn, K2, K3, params) -> θ

One 4th-order Runge-Kutta step of the 3-body Kuramoto model (allocates its stages).
`Hn`, `Ψn` must hold the local field evaluated at the current `θ`.
"""
function RK4_step(θ::Vector{Float64}, ω::Vector{Float64}, dt::Float64, Hn::Vector{Float64}, Ψn::Vector{Float64}, Rn::Vector{Complex{Float64}}, K2::Float64, K3::Float64, params::Any)
    n, degree2, A2, degree3, A3 = params

    k1 = zeros(Float64, n)
    @threads :dynamic for i in 1:n
        @inbounds k1[i] = dthdt(θ[i], ω[i], Hn[i], Ψn[i])
    end

    θ_k2 = θ .+ 0.5 .* dt .* k1
    H_k2, Ψ_k2, _ = cal_order_param(K2, K3, θ_k2, params)
    k2 = zeros(Float64, n)
    @threads :dynamic for i in 1:n
        @inbounds k2[i] = dthdt(θ_k2[i], ω[i], H_k2[i], Ψ_k2[i])
    end

    θ_k3 = θ .+ 0.5 .* dt .* k2
    H_k3, Ψ_k3, _ = cal_order_param(K2, K3, θ_k3, params)
    k3 = zeros(Float64, n)
    @threads :dynamic for i in 1:n
        @inbounds k3[i] = dthdt(θ_k3[i], ω[i], H_k3[i], Ψ_k3[i])
    end

    θ_k4 = θ .+ dt .* k3
    H_k4, Ψ_k4, _ = cal_order_param(K2, K3, θ_k4, params)
    k4 = zeros(Float64, n)
    @threads :dynamic for i in 1:n
        @inbounds k4[i] = dthdt(θ_k4[i], ω[i], H_k4[i], Ψ_k4[i])
    end

    @threads :dynamic for i in 1:n
        @inbounds θ[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
    end

    return θ
end

"""
    RK4_step_fast!(θ, ω, dt, Hn, Ψn, Rn, K2, K3, params, caches=nothing) -> θ

In-place RK4 step. Pass `caches = (H_k, Ψ_k, k1, k2, k3, k4, θ_tmp)` (seven length-`n`
`Float64` vectors) to avoid allocating them every step; with `caches = nothing` they are
allocated on each call. `Hn`, `Ψn` must hold the local field evaluated at the current `θ`.
"""
function RK4_step_fast!(θ::Vector{Float64}, ω::Vector{Float64}, dt::Float64, Hn::Vector{Float64}, Ψn::Vector{Float64}, Rn::Vector{Complex{Float64}}, K2::Float64, K3::Float64, params::Any, caches=nothing)
    n, degree2, A2, degree3, A3 = params

    if caches === nothing
        H_k = zeros(Float64, n)
        Ψ_k = zeros(Float64, n)
        k1 = zeros(Float64, n)
        k2 = zeros(Float64, n)
        k3 = zeros(Float64, n)
        k4 = zeros(Float64, n)
        θ_tmp = zeros(Float64, n)
    else
        H_k, Ψ_k, k1, k2, k3, k4, θ_tmp = caches
    end

    @threads :dynamic for i in 1:n
        @inbounds begin
            k1[i] = dthdt(θ[i], ω[i], Hn[i], Ψn[i])
            θ_tmp[i] = θ[i] + 0.5 * dt * k1[i]
        end
    end

    cal_order_param!(H_k, Ψ_k, Rn, K2, K3, θ_tmp, params)
    @threads :dynamic for i in 1:n
        @inbounds begin
            k2[i] = dthdt(θ_tmp[i], ω[i], H_k[i], Ψ_k[i])
            θ_tmp[i] = θ[i] + 0.5 * dt * k2[i]
        end
    end

    cal_order_param!(H_k, Ψ_k, Rn, K2, K3, θ_tmp, params)
    @threads :dynamic for i in 1:n
        @inbounds begin
            k3[i] = dthdt(θ_tmp[i], ω[i], H_k[i], Ψ_k[i])
            θ_tmp[i] = θ[i] + dt * k3[i]
        end
    end

    cal_order_param!(H_k, Ψ_k, Rn, K2, K3, θ_tmp, params)
    @threads :dynamic for i in 1:n
        @inbounds begin
            k4[i] = dthdt(θ_tmp[i], ω[i], H_k[i], Ψ_k[i])
            θ[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        end
    end

    return θ
end

"""
    evolve_RK4!(θ, ω, dt, time_steps, K2, K3, params, sumdeg2, Hn, Ψn, Rn, rk4_caches) -> Ravg

Integrate the phases `θ` in place for `time_steps` RK4 steps at fixed (K2, K3)
and return R = |Σ_i Σ_j A_ij e^{iθ_j}| / Σ_i k_i averaged over the second half.
"""
function evolve_RK4!(θ, ω, dt, time_steps, K2, K3, params, sumdeg2, Hn, Ψn, Rn, rk4_caches)
    avg_start = time_steps ÷ 2 + 1
    Rsum = 0.0

    for t in 1:time_steps
        cal_order_param!(Hn, Ψn, Rn, K2, K3, θ, params)
        if t >= avg_start
            Rsum += abs(sum(Rn)) / sumdeg2
        end
        RK4_step_fast!(θ, ω, dt, Hn, Ψn, Rn, K2, K3, params, rk4_caches)
    end

    return Rsum / (time_steps - avg_start + 1)
end


"""
    Heun_step(θ, ω, dt, Hn, Ψn, Rn, K2, K3, params) -> θ

One Heun (improved Euler) step of the 3-body Kuramoto model.
"""
function Heun_step(θ::Vector{Float64}, ω::Vector{Float64}, dt::Float64, Hn::Vector{Float64}, Ψn::Vector{Float64}, Rn::Vector{Complex{Float64}}, K2::Float64, K3::Float64, params::Any)
    n, degree2, A2, degree3, A3 = params
    params = (n, degree2, A2, degree3, A3)
    θ1 = zeros(Float64, n)
    dth1 = zeros(Float64, n)

    @threads :dynamic for i in 1:n
        @inbounds begin
            dth1[i] = dthdt(θ[i], ω[i], Hn[i], Ψn[i])
            θ1[i] = θ[i] + dth1[i] * dt
        end
    end

    H1, Ψ1, R1 = cal_order_param(K2, K3, θ1, params)
    @threads :dynamic for i in 1:n
        @inbounds begin
            dth2 = dthdt(θ1[i], ω[i], H1[i], Ψ1[i])
            θ[i] += 0.5 * (dth1[i] + dth2) * dt
        end
    end

    return θ
end

# FDA Helpers
"""
    fda_b(H)

Frequency-distribution-approximation (FDA) locking factor for Lorentzian frequencies,
b(H) = (√(H² + 1) - 1)/H = H/(√(H² + 1) + 1), written in the cancellation-free form.
"""
@inline fda_b(H::Float64) = H == 0.0 ? 0.0 : H/(sqrt(H*H + 1.0) + 1.0)

"""
    R1_th_Int(init_val, K2, K3, params) -> R

FDA self-consistent order parameter. Solves Hᵢ = K2 Σⱼ Aᵢⱼ b(Hⱼ) + K3 Σ_(m,k) b(Hₘ)² b(Hₖ)
by damped fixed-point iteration from Hᵢ = `init_val`·kᵢ, and returns R = Σᵢ Σⱼ Aᵢⱼ b(Hⱼ) / Σᵢ kᵢ.
"""
function R1_th_Int(init_val::Float64, K2::Float64, K3::Float64, params::Any)
    n, degree2, A2, degree3, A3 = params
    H = [init_val * Float64(degree2[i]) for i in 1:n]
    Hnew = zeros(Float64, n)
    b = zeros(Float64, n)
    R2_arr = zeros(Float64, n)
    
    maxiter = 300
    tol = 1e-10
    damping = 0.7

    for _ in 1:maxiter
        @threads :static for i in 1:n
            b[i] = fda_b(H[i])
        end

        @threads :static for i in 1:n
            r2 = 0.0
            r3 = 0.0

            @inbounds for j in 1:degree2[i]
                r2 += b[A2[i][j]]
            end

            @inbounds for j in 1:Int32(size(A3[i], 1))
                m = A3[i][j, 1]
                k = A3[i][j, 2]
                r3 += b[m]^2 * b[k]
            end

            R2_arr[i] = r2
            Hnew[i] = K2*r2 + K3*r3
        end

        err = maximum(abs.(Hnew .- H))
        scale = max(maximum(abs.(Hnew)), maximum(abs.(H)), 1e-8)
        
        @. H = damping*Hnew + (1.0-damping)*H
        
        if err <= tol*scale
            break
        end
    end

    R = sum(R2_arr) / sum(degree2)
    return R
end

# HMF Helpers

"""
    hmf_fixed_point(alpha0, beta0, K2, K3, k, q) -> (alpha, beta)

HMF fixed point from Eqs. (35)-(36), with k = realized dyadic degree and
q = realized triadic hyperdegree (number of hyperedges incident on a node).
`alpha` is the order parameter R.
"""
function hmf_fixed_point(alpha0::Float64, beta0::Float64, K2::Float64, K3::Float64,
                         k::AbstractVector{<:Real}, q::AbstractVector{<:Real};
                         maxiter::Int=3000, tol::Float64=1e-11, damping::Float64=0.6)
    alpha = alpha0
    beta = beta0

    sumk = sum(k)
    sumq = sum(q)

    for _ in 1:maxiter
        s1 = 0.0
        s2 = 0.0

        @inbounds for i in eachindex(k)
            H = K2*alpha*k[i] + K3*beta*q[i]
            b = fda_b(H)
            s1 += k[i]*b
            s2 += q[i]*b*b
        end

        alpha_new = s1 / sumk
        beta_new = 2.0*alpha * s2 / sumq

        err = max(abs(alpha_new - alpha), abs(beta_new - beta))
        scale = max(abs(alpha_new), abs(beta_new), abs(alpha), abs(beta), 1e-8)

        alpha = damping*alpha_new + (1.0-damping)*alpha
        beta  = damping*beta_new  + (1.0-damping)*beta

        err <= tol*scale && break
    end

    return alpha, beta
end

"""Eq. (53): HMF prediction for the onset of bistability."""
function K3crit_HMF(k::AbstractVector{<:Integer}, q::AbstractVector{<:Integer})
    kf = Float64.(k)
    qf = Float64.(q)

    k4  = mean(kf .^ 4)
    q1  = mean(qf)
    qk  = mean(qf .* kf)
    qk2 = mean(qf .* kf .^ 2)

    return k4 * q1^2 / (qk^2 * qk2)
end

"""
    compute_HMF_sweep(degree2, degree3, K3s, K2sf; K2Crit_mean, K3Crit_mean, ...)
        -> (D_hmf, R2f_all, R2b_all)

Forward and backward HMF sweeps over the K2 grid `K2sf` (in units of K2Crit_mean)
for every rK3 in `K3s` (in units of K3Crit_mean), using `hmf_fixed_point`.
Threaded over K3. `R2f_all[ik3]` / `R2b_all[ik3]` are indexed like `K2sf`, and
`D_hmf[ik3] = R2b_all[ik3] .- R2f_all[ik3]` is the hysteresis gap.

Each K2 point starts from `init_f` (forward) or `init_b` (backward). With
`continuation=true` it instead starts from the previous point's fixed point
(forward steps are floored at `seed_min` so the incoherent state can leave α = 0).
"""
function compute_HMF_sweep(degree2::AbstractVector{<:Real}, degree3::AbstractVector{<:Real},
                           K3s::AbstractVector, K2sf::AbstractVector;
                           K2Crit_mean::Float64, K3Crit_mean::Float64,
                           init_f::Float64=0.1, init_b::Float64=0.99,
                           continuation::Bool=false, seed_min::Float64=1e-6,
                           maxiter::Int=3000, tol::Float64=1e-11, damping::Float64=0.6)
    n_k3 = length(K3s)
    n_k2 = length(K2sf)

    R2f_all = [zeros(Float64, n_k2) for _ in 1:n_k3]
    R2b_all = [zeros(Float64, n_k2) for _ in 1:n_k3]
    D_hmf = Vector{Vector{Float64}}(undef, n_k3)

    @threads for ik3 in 1:n_k3
        K3 = K3s[ik3] * K3Crit_mean
        R2f = R2f_all[ik3]
        R2b = R2b_all[ik3]

        # Forward sweep (low -> high K2)
        α, β = init_f, init_f
        for ik2 in 1:n_k2
            K2 = K2sf[ik2] * K2Crit_mean
            # α = 0 is an exact fixed point, so keep a small seed for continuation
            α, β = continuation ? (max(α, seed_min), max(β, seed_min)) : (init_f, init_f)
            α, β = hmf_fixed_point(α, β, K2, K3, degree2, degree3;
                                   maxiter=maxiter, tol=tol, damping=damping)
            R2f[ik2] = α
        end

        # Backward sweep (high -> low K2)
        α, β = init_b, init_b
        for ik2 in n_k2:-1:1
            K2 = K2sf[ik2] * K2Crit_mean
            continuation || ((α, β) = (init_b, init_b))
            α, β = hmf_fixed_point(α, β, K2, K3, degree2, degree3;
                                   maxiter=maxiter, tol=tol, damping=damping)
            R2b[ik2] = α
        end

        D_hmf[ik3] = R2b .- R2f
    end

    return D_hmf, R2f_all, R2b_all
end

# TAT Helpers
"""
    tat_second_harmonic(w, H)

Second-harmonic contribution ⟨cos 2φ⟩ of an oscillator with frequency `w` in local field `H`
(locked branch for |w| ≤ H, drifting branch otherwise), used by [`R1_th_sum`](@ref).
"""
@inline function tat_second_harmonic(w::Float64, H::Float64)
    aw = abs(w)
    if H <= 0.0
        return 0.0
    elseif aw <= H
        x = w/H
        return 1.0 - 2.0*x*x
    else
        return 1.0 - 2.0*aw/(sqrt(max(aw*aw - H*H, 0.0)) + aw)
    end
end

"""
    R1_th_sum(init_val, K2, K3, omega, params) -> R

Self-consistent order parameter evaluated on the actual frequency sample `omega` (TAT).
Solves Hᵢ = K2 Σⱼ Aᵢⱼ √(1-(ωⱼ/Hⱼ)²) + K3 Σ_(m,k) s₂(ωₘ,Hₘ) √(1-(ωₖ/Hₖ)²) by damped
fixed-point iteration from Hᵢ = `init_val`·kᵢ, and returns the dyadic R normalised by Σᵢ kᵢ.
"""
function R1_th_sum(init_val::Float64, K2::Float64, K3::Float64, omega::Vector{Float64}, params::Any)
    n, degree2, A2, degree3, A3 = params
    H = [init_val * Float64(degree2[i]) for i in 1:n]
    Hnew = zeros(Float64, n)
    firstharm = zeros(Float64, n)
    secondharm = zeros(Float64, n)
    R2_arr = zeros(Float64, n)
    
    maxiter = 10000
    tol = 1e-15
    damping = 0.7

    for _ in 1:maxiter
        @threads :static for i in 1:n
            Hi = H[i]
            wi = omega[i]
            aw = abs(wi)

            if Hi > 0.0 && aw <= Hi
                x = wi/Hi
                firstharm[i] = sqrt(max(1.0 - x*x, 0.0))
            else
                firstharm[i] = 0.0
            end

            secondharm[i] = tat_second_harmonic(wi, Hi)
        end

        @threads :static for i in 1:n
            r2 = 0.0
            r3 = 0.0

            @inbounds for j in 1:degree2[i]
                r2 += firstharm[A2[i][j]]
            end

            @inbounds for j in 1:Int32(size(A3[i], 1))
                m = A3[i][j, 1]
                k = A3[i][j, 2]
                r3 += secondharm[m] * firstharm[k]
            end

            R2_arr[i] = r2
            Hnew[i] = K2*r2 + K3*r3
        end

        err = maximum(abs.(Hnew .- H))
        scale = max(maximum(abs.(Hnew)), maximum(abs.(H)), 1e-8)
        
        @. H = damping*Hnew + (1.0-damping)*H
        
        if err <= tol*scale
            break
        end
    end

    R = sum(R2_arr) / sum(degree2)
    return R
end


########################### REWIRE FUNCTIONS ###########################

"""Compute Ω contribution of a hyperedge (i,j,k) given eigenvector Un."""
@inline function Ω_cal(Un, i, j, k)
    ui = Un[i]
    uj = Un[j]
    uk = Un[k]
    return ui * uj * uk * 2.0 * (ui + uj + uk)
end

"""Push all 6 directed views of the hyperedge (i,j,k) into the adjacency lists."""
@inline function update_A3!(A3_vec, i, j, k)
    push!(A3_vec[i], (Int16(j), Int16(k)))
    push!(A3_vec[i], (Int16(k), Int16(j)))
    push!(A3_vec[j], (Int16(i), Int16(k)))
    push!(A3_vec[j], (Int16(k), Int16(i)))
    push!(A3_vec[k], (Int16(i), Int16(j)))
    push!(A3_vec[k], (Int16(j), Int16(i)))
end



"""
    generate_sets_adjusted(lst) -> (sets, count_duplicates)

Group leftover stubs `lst` into triples, preferring three distinct nodes. Returns the triples
and how many of them had to repeat a node.
"""
function generate_sets_adjusted(lst)
    # Count frequency of each number
    freq = countmap(lst)
    result = []              # List of sets
    count_duplicates = 0     # Counter for sets with duplicates

    # Continue while at least 3 numbers remain
    while sum(values(freq)) >= 3
        # Get distinct numbers with positive frequency as a sorted array
        available_keys = sort(collect(k for (k, v) in freq if v > 0))

        if length(available_keys) >= 3
            # Form set with 3 distinct numbers
            set = sample(available_keys, 3, replace=false)
            for num in set
                freq[num] -= 1
                if freq[num] == 0
                    delete!(freq, num)
                end
            end
        else
            # Form set with possible duplicates
            set = []
            for i in 1:3
                # Recompute available keys with remaining frequency
                available = sort(collect(k for (k, v) in freq if v > 0))
                num = available[1]  # Pick smallest available number
                push!(set, num)
                freq[num] -= 1
                if freq[num] == 0
                    delete!(freq, num)
                end
            end
            count_duplicates += 1  # Set has duplicates since distinct numbers < 3
        end
        push!(result, set)
    end

    return result, count_duplicates
end


"""
    rewire_3D_Ω(n, A3, degree3, Un, ways) -> (new_A3, deg3p)

Rewire hyperedges to increase or decrease K3crit by swapping third vertices
between randomly paired hyperedges when the Ω metric improves.

`ways` must be `:decrease` or `:increase`.
"""
function rewire_3D_Ω(n, A3, degree3, Un, ways)
    hyperedges = Tuple{Int, Int, Int}[]
    for i in 1:n
        @inbounds for j in 1:size(A3[i], 1)
            a, b, c = Int(i), Int(A3[i][j, 1]), Int(A3[i][j, 2])
            if a > b; a, b = b, a; end
            if b > c; b, c = c, b; end
            if a > b; a, b = b, a; end
            push!(hyperedges, (a, b, c))
        end
    end

    unique!(sort!(hyperedges))
    # Rewire the stubs
    new_A3_vec = [Tuple{Int16, Int16}[] for _ in 1:n]
    L = length(hyperedges) - (length(hyperedges) % 2)

    shuffle!(hyperedges)

    for i in 1:2:L

        # Tuples are immutable: shuffle a copy so any vertex can be the swapped one
        hyperedges[i] = Tuple(shuffle!(collect(hyperedges[i])))
        hyperedges[i+1] = Tuple(shuffle!(collect(hyperedges[i+1])))

        Ωi1 = Ω_cal(Un, hyperedges[i][1], hyperedges[i][2], hyperedges[i][3])
        Ωi2 = Ω_cal(Un, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i+1][3])

        Ωf1 = Ω_cal(Un, hyperedges[i][1], hyperedges[i][2], hyperedges[i+1][3])
        Ωf2 = Ω_cal(Un, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i][3])

        if ways == :decrease #decrease K3crit
            if Ωi1 + Ωi2 > Ωf1 + Ωf2
                update_A3!(new_A3_vec, hyperedges[i][1], hyperedges[i][2], hyperedges[i][3])
                update_A3!(new_A3_vec, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i+1][3])
            else
                update_A3!(new_A3_vec, hyperedges[i][1], hyperedges[i][2], hyperedges[i+1][3])
                update_A3!(new_A3_vec, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i][3])
            end
        elseif ways == :increase #increase K3crit
            if Ωi1 + Ωi2 < Ωf1 + Ωf2
                update_A3!(new_A3_vec, hyperedges[i][1], hyperedges[i][2], hyperedges[i][3])
                update_A3!(new_A3_vec, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i+1][3])
            else
                update_A3!(new_A3_vec, hyperedges[i][1], hyperedges[i][2], hyperedges[i+1][3])
                update_A3!(new_A3_vec, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i][3])
            end
        else
            error("Invalid way specified. Use :decrease or :increase.")
        end
    end
    # Handle the case where L is odd
    if length(hyperedges) % 2 != 0
        update_A3!(new_A3_vec, hyperedges[end][1], hyperedges[end][2], hyperedges[end][3])
    end

    deg3p = [Int(length(new_A3_vec[i]) ÷ 2) for i in 1:n]
    if all(degree3 .== deg3p)
        println("Preserve degree: ", all(degree3 .== deg3p))
    else
        idx_miss = findall(x -> x == 0, deg3p .== degree3)
        count_miss = abs.(deg3p .- degree3)[idx_miss]
        println("Missing degree: ", sum(count_miss))
        stubs = Int[]
        for i in 1:length(idx_miss)
            append!(stubs, [idx_miss[i] for _ in 1:count_miss[i]])
        end
        println("Complete stubs: ", length(stubs) % 3)

        limited_combinations, duplicated = generate_sets_adjusted(stubs)
        println("Replica hyperedges: ", duplicated)

        for i in 1:length(limited_combinations)
            lc = limited_combinations[i]
            update_A3!(new_A3_vec, lc[1], lc[2], lc[3])
        end
        deg3p = [Int(length(new_A3_vec[i]) ÷ 2) for i in 1:n]
    end
    
    new_A3 = vec_to_matrix(new_A3_vec, n)
    return new_A3, deg3p
end
