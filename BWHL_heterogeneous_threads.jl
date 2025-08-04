ENV["GKSwstype"] = "100"  # Prevent GR display errors in WSL

using DifferentialEquations
using Plots
using Random
using DelimitedFiles
using Sundials
using OrdinaryDiffEq
using CSV, DataFrames  # <- Added for reading initial conditions
using Base.Threads
function main()
    # -- Spatial Setup --
    L = 100
    nx = 8192
    dx = L / nx
    x = LinRange(0, L, nx)

    # -- Model Parameters --
    # reading the growth rate from an external file (brut force approach)
    df_init = CSV.read("brut_force_growth.csv", DataFrame)
    lamda_vec = df_init.lamda
    #lamda = 0.6	
    E = 10.0
    K_B = 10.0
    M_B = 10.0
    D_B = 1.0
    P = 67.6
    N = 25.0
    R = 0.1
    GamW = 10.0
    D_W = 150.0


    #M_B_vec1 = LinRange(0.01, 10.0, div(nx, 2))
    #M_B_vec2 = LinRange(10.0, 0.01, div(nx, 2))
    #M_B_vec = vcat(M_B_vec1, M_B_vec2)
    #mu = L/2.0
    #width = L/6.0
    #min_M_B = 0.01
    #max_M_B = 10.0
    #gaussian_profile = max_M_B .* exp.(-((x .- mu).^2) ./ (2 * width^2))
    #M_B_vec = max.(gaussian_profile, min_M_B)
    #D_B = 1.2

    #P = 150.0
    #N = 20.0
    #R = 0.01
    #GamW = 10.0
    #D_W = 150

    M_H = 0.7
    GamH = 0.4
    K_H = 150.0
    alpha = 60.0
    beta = 0.2
    D_HB = 50000.0
    D_HH = 100000.0
    kappa = 0.45
    zeta = 0.5

    alpha_L = 5.0
    m_L = 0.5
    D_L = 10.0

    # -- Load Initial B and W from CSV --
    df_init = CSV.read("Bini_P676.csv", DataFrame)
    B0 = df_init.B
    W0 = df_init.W

    mu = 40
    width = 1.0
    max_H0 = 0.5
    min_H0 = 0.0
    gaussian_profile = max_H0 .* exp.(-((x .- mu).^2) ./(2 * width^2))
    H0 = max.(gaussian_profile, min_H0)
    #H0 = fill(0.45, nx)  # still initialized manually
    
    L0 = fill(0.3, nx)
    u0 = vcat(B0, W0, H0, L0)

    # -- Reaction-Diffusion System --
    function BWH!(du, u, p, t)
        B = @view u[1:nx]
        W = @view u[nx+1:2*nx]
        H = @view u[2*nx+1:3*nx]
        L = @view u[3*nx+1:4*nx]

        dB = @view du[1:nx]
        dW = @view du[nx+1:2*nx]
        dH = @view du[2*nx+1:3*nx]
        dL = @view du[3*nx+1:4*nx]

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
        L_pad = lap(L)

        laplaceB = (B_pad[3:end] .- 2 .* B_pad[2:end-1] .+ B_pad[1:end-2]) ./ dx^2
        laplaceW = (W_pad[3:end] .- 2 .* W_pad[2:end-1] .+ W_pad[1:end-2]) ./ dx^2
        laplaceL = (L_pad[3:end] .- 2 .* L_pad[2:end-1] .+ L_pad[1:end-2]) ./ dx^2

        gradB = grad(B)
        gradH = grad(H)
        gradL = grad(L)

        veg_flux = D_HB .* H .* (L ./ (L .+ kappa)) .* gradB
        rand_flux = -(D_HH ./ (1 .+ L)) .* gradH
        div_flux = -grad(veg_flux .+ rand_flux)
        
        
        @threads for i in 1:nx
            @inbounds begin 
                #G_B = alpha * B[i]/(beta + B[i])
                G_B = alpha * B[i] * L[i]/(L[i] + kappa)
                growth = lamda_vec[i] * B[i] * W[i] * (1 + (E * B[i]))^2 * (1 - B[i]/K_B)
                mortality_B = M_B * B[i]
                consumption = G_B * H[i]
                dB[i] = growth - mortality_B - consumption + (D_B * laplaceB[i])

                evap = N * W[i] /(1 + (R * B[i]/K_B))
                uptake = GamW * B[i] * W[i] * (1 + (E * B[i]))^2
                dW[i] = P - evap - uptake + (D_W * laplaceW[i])

                dH[i] = (-M_H * H[i]) + (GamH * consumption * (1 - H[i]/K_H)) + div_flux[i]
                        
                learn = alpha_L * H[i] * B[i]
                forget = m_L * L[i]
                dL[i] = learn - forget + (D_L * laplaceL[i])
            end
        end
        #for i in 1:nx
        #    G_B = alpha * B[i] / (beta + B[i])
        #    growth = lamda_vec[i] * B[i] * W[i] * (1 + E * B[i])^2 * (1 - B[i] / K_B)
        #    mortality_B = M_B * B[i]
        #    consumption = G_B * H[i]
        #    dB[i] = growth - mortality_B - consumption + D_B * laplaceB[i]

        #    evap = N * W[i] / (1 + R * B[i] / K_B)
        #    uptake = GamW * B[i] * W[i] * (1 + E * B[i])^2
        #    dW[i] = P - evap - uptake + D_W * laplaceW[i]

        #    dH[i] = -M_H * H[i] + GamH * consumption * (1 - H[i] / K_H) + div_flux[i]
        #end
    end

    # -- Time Integration --
    tspan = (0.0, 10.0)
    prob = ODEProblem(BWH!, u0, tspan)
    @time sol = solve(prob, Rodas5(), dt=1e-6, adaptive=true, saveat=0.001)
    #sol = solve(prob, TRBDF2(), abstol=1e-6, reltol=1e-6, saveat=0.1)
    
    # -- Save Final Herbivore Profile --
    #
    B_final = sol.u[end][1:nx]
    W_final = sol.u[end][nx+1:2*nx]
    H_final = sol.u[end][2*nx+1:3*nx]
    L_final = sol.u[end][3*nx+1:4*nx]
    # Combine into a matrix with columns: x, B, W, H, L
    data_out = hcat(x, B_final, W_final, H_final, L_final)
    # Write with header
    header = ["x", "B", "W", "H", "L"]
    #CSV.write("learning_P$(P).csv", Tables.table(data_out, header))
    df_out = DataFrame(data_out, Symbol.(header))
    CSV.write("learning_P$(P).csv", df_out)
    println("Final state saved as learning_P$(P).csv")

    #writedlm("learning_P$(P).csv", sol.u[end][2nx+1:3nx], ',')
    #println("Simulation complete. Final state saved.")

    #computing global min and max of B, H and L
    Bs = [u[1:nx] for u in sol.u]
    Hs = [u[2*nx+1:3*nx] for u in sol.u]
    Ls = [u[3*nx+1:4*nx] for u in sol.u]

    Bmin, Bmax = minimum([minimum(b) for b in Bs]), maximum([maximum(b) for b in Bs])
    Hmin, Hmax = minimum([minimum(h) for h in Hs]), maximum([maximum(h) for h in Hs])
    Lmin, Lmax = minimum([minimum(l) for l in Ls]), maximum([maximum(l) for l in Ls])
    # -- Animate B and H in a single plot --
    anim = @animate for u in sol.u
        B = u[1:nx]
        H = u[2nx+1:3nx]
        L = u[3nx+1:4nx]

        plot(
             plot(x, B, label="B(x,t)", title="Biomass", lw=2, color="green", ylim=(Bmin, Bmax)),
             plot(x, H, label="H(x,t)", title="Herbivore", lw=2, color="black", ylim=(Hmin, Hmax)),
             plot(x, L, label="L(x,t)", title="Learning", lw=2, color="red", ylim=(Lmin, Lmax)),
             layout = (3, 1), size=(900, 800)
        )
    end
    mp4(anim, "learning_P$(P).mp4", fps=10)
    println("Combined animation saved as biomass_herbivore.mp4")
end

main()
