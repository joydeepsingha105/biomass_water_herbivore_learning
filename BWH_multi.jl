ENV["GKSwstype"] = "100"  # Prevent GR display errors in WSL

using DifferentialEquations
using Plots
using Random
using DelimitedFiles
using Sundials
using OrdinaryDiffEq
using Base.Threads  # For multithreading

function main()
    println("🧵 Using ", Threads.nthreads(), " Julia threads")

    # -- Spatial Setup --
    L = 200
    global nx = 4096
    global dx = L / nx
    x = LinRange(0, L, nx)

    # -- Model Parameters --
    global lamda = 0.6    
    global E = 10.0
    global K_B = 10.0
    global M_B = 11.0
    global D_B = 1.2
    global P = 180.0
    global N = 20.0
    global R = 0.01
    global GamW = 10.0
    global D_W = 150
    global M_H = 5.0
    global GamH = 0.3
    global K_H = 150
    global alpha = 23.5
    global beta = 0.3
    global D_HB = 5000
    global D_HH = 100000
    global kappa = 0.0001
    global zeta = 0.001

    # -- Initial Conditions --
    Random.seed!(42)
    B0 = 0.6 .+ (0.01 .* sin.(2 * pi .* 10 .* x ./ maximum(x)))
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

        combined_flux = zeros(nx)
        mort = zeros(nx)
        prol = zeros(nx)

        Threads.@threads for i in 1:nx
            ip = (i == nx) ? 1 : i + 1
            im = (i == 1)  ? nx : i - 1

            b = B[i]
            w = W[i]
            h = H[i]

            lapB = (B[ip] - 2B[i] + B[im]) / dx^2
            lapW = (W[ip] - 2W[i] + W[im]) / dx^2
            gradB = (B[ip] - B[im]) / (2dx)
            gradH = (H[ip] - H[im]) / (2dx)

            growth = lamda * b * w * (1 + E * b)^2 * (1 - b / K_B)
            mortality_b = M_B * b
            consumption = alpha * b * h / (beta + b)
            dB[i] = growth - mortality_b - consumption + D_B * lapB

            evap = N * w / (1 + (R * b / K_B))
            uptake = GamW * b * w * (1 + E * b)^2
            dW[i] = P - evap - uptake + D_W * lapW

            veg_flux = D_HB * h * (kappa / (kappa + b)) * gradB
            rand_flux = -D_HH * (zeta^2 / (zeta^2 + b^2)) * gradH
            combined_flux[i] = veg_flux + rand_flux

            mort[i] = -M_H * h
            prol[i] = GamH * consumption * (1 - h / K_H)
        end

        Threads.@threads for i in 1:nx
            ip = (i == nx) ? 1 : i + 1
            im = (i == 1)  ? nx : i - 1

            div_flux = -(combined_flux[ip] - combined_flux[im]) / (2dx)
            dH[i] = mort[i] + prol[i] + div_flux
        end
    end

    # -- Time Integration --
    tspan = (0.0, 2.0)
    prob = ODEProblem(BWH!, u0, tspan)
    sol = solve(prob, Rodas5(), adaptive=true, saveat=1.0)

    # -- Save Results --
    writedlm("herbivore_final_multi.csv", sol.u[end][2nx+1:3nx], ',')
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
    gif(anim, "biomass_herbivore_multi.gif", fps=10)
    println("✅ Combined animation saved as biomass_herbivore.gif")
end

main()
