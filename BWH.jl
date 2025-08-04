ENV["GKSwstype"] = "100"  # Prevent GR display errors in WSL

using DifferentialEquations
using Plots
using Random
using DelimitedFiles
using Sundials
using OrdinaryDiffEq

function main()
    # -- Spatial Setup --
    L = 200
    nx = 2048
    dx = L / nx
    x = LinRange(0, L, nx)

    # -- Model Parameters --
    lamda = 0.6	
    E = 10.0
    K_B = 10.0
    M_B = 11.0
    #M_B_vec1 = LinRange(0.01, 10.0, div(nx, 2))
    #M_B_vec2 = LinRange(10.0, 0.01, div(nx, 2))
    #M_B_vec = vcat(M_B_vec1, M_B_vec2)
    D_B = 1.2

    P = 180.0
    N = 20.0
    R = 0.01
    GamW = 10.0
    D_W = 150

    M_H = 5.0
    GamH = 0.3
    K_H = 150
    alpha = 23.5
    beta = 0.3
    D_HB = 5000
    D_HH = 100000
    kappa = 0.0001
    zeta = 0.001

    # -- Initial Conditions --
    Random.seed!(42)
    B0 = 0.6 .+ (0.01 .* sin.(2 * pi .* 25 .* x ./ maximum(x)))
    W0 = fill(1.5, nx)
    H0 = fill(0.45, nx)

    u0 = vcat(B0, W0, H0)  # Full state vector

    # -- Reaction-Diffusion System --
    function BWH!(du, u, p, t)
        B = @view u[1:nx]
        W = @view u[nx+1:2nx]
        H = @view u[2nx+1:3nx]

        dB = @view du[1:nx]
        dW = @view du[nx+1:2nx]
        dH = @view du[2nx+1:3nx]

        # Periodic Laplacian & Gradient
        function lap(v)
            [v[end]; v; v[1]]
        end
        function grad(v)
            v_pad = [v[end]; v; v[1]]
            (v_pad[3:end] .- v_pad[1:end-2]) ./ (2*dx)
        end

        B_pad = lap(B)
        W_pad = lap(W)
        H_pad = lap(H)
        laplaceB = (B_pad[3:end] .- 2 .* B_pad[2:end-1] .+ B_pad[1:end-2]) ./ dx^2
        laplaceW = (W_pad[3:end] .- 2 .* W_pad[2:end-1] .+ W_pad[1:end-2]) ./ dx^2
        gradB = grad(B)
        gradH = grad(H)
	
	# vegetation equation
	growth = lamda .* B .* W .* (1 .+ (E .* B)).^2 .* (1.0 .- B ./K_B)
	mortality = M_B .* B
	consumption = alpha .* B .* H ./(beta .+ B)
	@. dB = growth .- mortality .- consumption .+ (D_B .* laplaceB)

	# soil water equation 
	evap = N .* W ./ (1 .+ (R .* B ./K_B))
	uptake = GamW .* B .* W .* (1 .+ (E .* B)).^2
	@. dW = P .- evap .- uptake .+ (D_W .* laplaceW)

	# herbivore equation
	veg_flux = D_HB .* H .* (kappa ./ (kappa .+ B)) .* gradB
	rand_flux = -D_HH .* (zeta^2 ./ (zeta^2 .+ B.^2)) .* gradH
	div_flux = -grad(veg_flux .+ rand_flux)
	mortality = -M_H .* H
	proliferation = GamH .* consumption .* (1 .- H ./K_H)
	@. dH = mortality .+ proliferation .+ div_flux

        #for i in 1:nx
            # -- B equation --
            #G_B = alpha * B[i] / (beta + B[i])
	    #growth = lamda * B[i] * W[i] * (1 + E * B[i])^2 * (1 - B[i] / K_B)
	    #mortality_B = M_B_vec[i] * B[i]
	    #consumption = G_B * H[i]
            #dB[i] = growth - mortality_B - consumption + D_B * laplaceB[i]

            # -- W equation --
            #evap = N * W[i] / (1 + R * B[i] / K_B)
            #uptake = GamW * B[i] * W[i] * (1 + E * B[i])^2
            #dW[i] = P - evap - uptake + D_W * laplaceW[i]

            # -- H equation --
            #G_B = α * B[i] / (β + H[i])
	    #veg_flux = D_HB * H[i] * (κ / (κ + B[i])) * ∇B[i]
	    #rand_flux = -D_HH * (ξ^2 / (ξ^2 + B[i]^2)) * ∇H[i]
            #div_flux = -grad(veg_flux .+ rand_flux)

            #dH[i] = -M_H * H[i] + A * consumption * (1 - H[i] / K_H) + div_flux[i]
        #end
    end

    # -- Time Integration --
    tspan = (0.0, 500.0)
    prob = ODEProblem(BWH!, u0, tspan)
    sol = solve(prob, Rodas5(), adaptive=true,saveat=1.0)

    # -- Save Results --
    writedlm("herbivore_final.csv", sol.u[end][2nx+1:3nx], ',')
    println("✅ Simulation complete. Final state saved.")
    # -- Animate B and H in a single plot --
    anim = @animate for i in 1:length(sol.t)
	    u = sol.u[i]
	    B = u[1:nx]
	    H = u[2nx+1:3nx]

	    plot(
		 plot(x, B, ylim=(0,1.2), label="B(x,t)", title="Biomass", lw=2),
		 plot(x, H, ylim=(0,1.5), label="H(x,t)", title="Herbivore", lw=2),
		layout = (2, 1), size=(800, 600)
		)
    end
    gif(anim, "biomass_herbivore.gif", fps=10)
    println("✅ Combined animation saved as biomass_herbivore.gif")

end

main()

