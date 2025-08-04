ENV["GKSwstype"] = "100"
using DifferentialEquations
using Plots
using Random
using CSV, DataFrames  
using Base.Threads
function main()
    # -- Spatial Setup --
    L = 100            # domain length
    nx = 8192            # number of spatial points
    dx = L / nx
    x = LinRange(0, L, nx)

    # -- Parameters (you can tweak these) --
    #lamda = 0.5
    df_init = CSV.read("P49_initial_condition_2.csv", DataFrame)
    #lamda_vec = 0.5 .+ (0.05 .* sin.(2 * pi .* x./ maximum(x)))
    lamda_vec = df_init.lamda
    #
    #lamda_vec = 0.2 .+ (0.0032 .* x)
    
    E = 10.0
    K_B = 10.0
    M_B = 10.0
    D_B = 1.0
    P = 68.0
    N = 25.0
    R = 0.1
    GamW = 10.0
    D_W = 150.0
    
    #gaussian growth rate profile
    #mu = L/2.0
    #width = L/6.0
    #max_M_B = 10.0
    #min_M_B = 0.01
    #gaussian_profile = max_M_B .* exp.(-((x .- mu).^2) ./ (2 * width^2))
    #M_B_vec = max.(gaussian_profile, min_M_B) 
    
    #triangular growth rate profile
    #M_B_vec1 = LinRange(0.1, 10, div(nx, 2))	
    #M_B_vec2 = LinRange(10.0, 0.1, div(nx, 2))
    #M_B_vec = vcat(M_B_vec1, M_B_vec2)
    #M_B_vec = LinRange(0.001, 10.0, nx)
    

    # -- Initial Conditions --
    #Random.seed!(42)
    #B0 = 0.5 .+ 0.01 .* sin.(2 * pi .* 100 .* x ./ maximum(x))
    #B0 = fill(0.6, nx)
    #W0 = fill(1.5, nx)

    df_init = CSV.read("P49_two_stable_spot.csv", DataFrame)
    B0 = df_init.B
    #W0 = df_init.W
    W0 = fill(1.5, nx)
    u0 = vcat(B0, W0)  # Flatten into a single state vector

    function lap(v)
	[v[end]; v; v[1]] #pad for periodic BC
    end

    # -- PDE system as ODE vector field --
    function reaction_diffusion!(du, u, p, t)
        B = @view u[1:nx]
        W = @view u[nx+1:end]
        dB = @view du[1:nx]
        dW = @view du[nx+1:end]

        # Periodic Laplacian
        #function lap(v)
        #    [v[end]; v; v[1]]  # pad for periodic BC
        #end

        #B_pad = lap(B)
        #W_pad = lap(W)
	
	#inv_dx2 = 1.0/dx^2
	#RK_inv = R / K_B
	#inv_K_B = 1.0 / K_B

        #laplaceB = (B_pad[3:end] .- 2 .* B_pad[2:end-1] .+ B_pad[1:end-2]) .* inv_dx2
        #laplaceW = (W_pad[3:end] .- 2 .* W_pad[2:end-1] .+ W_pad[1:end-2]) .* inv_dx2

	#vegetation_equation
	#growth = lamda .* B .* W .* (1 .+ (E .* B).^2) .* (1 .- (B ./ K_B))
	#mortality = M_B_vec .* B
	#@. dB = growth .- mortality .+ (D_B .* laplaceB)

	#soil_water_equation
	#evap = N .* W ./(1 .+ (R .* (B ./ K_B)))
	#uptake = GamW .* B .* W .* (1 .+ (E .* B)).^2
	#@. dW = P .- evap .- uptake .+ (D_W .* laplaceW)

        #@inbounds for i in 1:nx
	#    B_i = B[i]
	#    W_i = W[i]

	#    EB = 1 + (E * B_i)
	#    EB2 = EB^2

	#    growth = lamda * B_i * W_i * EB2 * (1 - (B_i * inv_K_B))
	#    mortality = M_B_vec[i] * B_i
	#    uptake = GamW * B_i * W_i * EB2
	#    evap = N * W_i / (1 + (RK_inv * B_i))

        #    dB[i] = growth - mortality + D_B * laplaceB[i]
        #    dW[i] = P - evap - uptake + D_W * laplaceW[i]
	
	@threads for i in 1:nx
            ip1 = (i == nx) ? 1 : i + 1
            im1 = (i == 1) ? nx : i - 1

            lapB = (B[im1] - (2 * B[i]) + B[ip1]) / dx^2
            lapW = (W[im1] - (2 * W[i]) + W[ip1]) / dx^2

            growth = lamda_vec[i] * B[i] * W[i] * (1 + (E * B[i]))^2 * (1 - B[i]/K_B) 
            mortality = M_B * B[i]
            uptake = GamW * B[i] * W[i] * (1 + (E * B[i]))^2
            evap = N * W[i] / (1 + (R * B[i]/ K_B))

            dB[i] = growth - mortality + (D_B * lapB)
            dW[i] = P - evap - uptake + (D_W * lapW)

        end
    end

    # -- Time Domain --
    tspan = (0.0, 500)

    # -- Solve the system --
    prob = ODEProblem(reaction_diffusion!, u0, tspan)
    sol = solve(prob, Rodas5(), saveat=0.1)  # Use a stable method like BS3()

    # -- Animate Biomass (B) over time --
    anim = @animate for u in sol.u
        B = u[1:nx]
        plot(x, B, label="B(x,t)", title="Biomass over time")
    end
    mp4(anim, "biomass_evolution_P$(Int(10*P))_hetero.mp4", fps=10)

    u_final = sol.u[end]
    B_final = u_final[1:nx]
    W_final = u_final[nx+1:end]
    df = DataFrame(x = x, B = B_final, W = W_final)
    CSV.write("P$(Int(10*P))_data.csv", df) 
    println("Simulation complete. Video saved")
end
using BenchmarkTools
@time main()
