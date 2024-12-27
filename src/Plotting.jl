function _get_full_primer_history(nodes::Vector{PrimerNode})
    tfull = Vector{Float64}()
    pmagfull = Vector{Float64}()
    impulses = Vector{Float64}()
    for i in 1:length(nodes)-1
        dv0 = nodes[i].outVelocity - nodes[i].inVelocity
        dv1 = nodes[i+1].outVelocity - nodes[i+1].inVelocity

        if norm(dv0) > 0
            push!(impulses, nodes[i].time)
        end
        if i == length(nodes) - 1 && norm(dv1) > 0
            push!(impulses, nodes[i+1].time)
        end

        p0 = norm(dv0) == 0 ? zeros(3) : unit(dv0)
        p1 = norm(dv1) == 0 ? zeros(3) : unit(dv1)
        x0p = [nodes[i].position; nodes[i].outVelocity]
        tof = nodes[i+1].time - nodes[i].time
        p0dot = _calculate_p0dot(p0, p1, tof, x0p)
        stateHist = _get_full_state(p0, p0dot, x0p, tof)

        tfull = [tfull; (stateHist.t .+ nodes[i].time)]
        pmagfull = [pmagfull; norm.(eachcol(stateHist.primers[1:3, :]))]
    end
    return tfull, pmagfull, impulses
end
function plotFullPrimerHistory(nodes::Vector{PrimerNode}; legend::Bool=false)
    tfull, pmagfull, impulses = _get_full_primer_history(nodes)
    set_theme!(theme_black())
    fig = Figure(size=(700, 600))

    ax = Axis(fig[1, 1])
    ax.xlabel = "Transfer Time (s)"
    ax.ylabel = "Primer Vector Magnitude"
    lines!(ax, tfull, pmagfull, color=:cyan, label="Primer History")
    scatter!(ax, impulses, ones(length(impulses)), color=:orange, label="Impulses")
    legend && axislegend(ax, position=:cb)
    display(GLMakie.Screen(), fig)
end

function plot_results(
    x0,
    x1,
    tof;
    primerNodes=nothing,
    collocationState=nothing,
    pseudoState=nothing,
    psPrimer=nothing,
)
    r0 = x0[1:3]
    rf = x1[1:3]
    v0 = x0[4:6]
    vf = x1[4:6]
    x1 = [x1...]
    mu = 3.986e5
    n = _get_default_half_revs(0.5 * (norm(x0[1:3]) + norm(x1[1:3])), tof, mu=mu)

    # initial orbit
    prob = ODEProblem(dstate, x0, (0, tof), 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    t = 0.0:10:tof
    initial = sol.(t)
    initialx = [initial[i][1] for i in 1:length(t)]
    initialy = [initial[i][2] for i in 1:length(t)]
    initialz = [initial[i][3] for i in 1:length(t)]

    # final orbit
    prob = ODEProblem(dstate, x1, (tof, 0), 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    final = sol.(t)
    finalx = [final[i][1] for i in 1:length(t)]
    finaly = [final[i][2] for i in 1:length(t)]
    finalz = [final[i][3] for i in 1:length(t)]

    vxfer, vstop = basic_lambert(r0, v0, rf, tof, n, verbose=false, v2=vf)
    x0p = [r0; vxfer]
    prob = ODEProblem(dstate, x0p, (0, tof), 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    lambert = sol.(t)
    lambertx = [lambert[i][1] for i in 1:length(t)]
    lamberty = [lambert[i][2] for i in 1:length(t)]
    lambertz = [lambert[i][3] for i in 1:length(t)]

    set_theme!(theme_black())
    fig = Figure(size=(800, 800))

    ax = Axis3(fig[1, 1])
    scatter!(ax, x0[1], x0[2], x0[3], color=:yellow, label="Initial Orbit")
    lines!(ax, initialx, initialy, initialz, color=:yellow)
    scatter!(ax, x1[1], x1[2], x1[3], color=:magenta, label="Final Orbit")
    lines!(ax, finalx, finaly, finalz, color=:magenta)

    lines!(ax, lambertx, lamberty, lambertz, color=:cyan, label="Lambert Soln")

    if !isnothing(primerNodes)
        for i in 1:length(primerNodes)-1
            x0p = [primerNodes[i].position...; primerNodes[i].outVelocity...]
            prob = ODEProblem(dstate, x0p, (primerNodes[i].time, primerNodes[i+1].time), 0)
            sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
            t = range(primerNodes[i].time, primerNodes[i+1].time, 100)
            lambert = sol.(t)
            lambertx = [lambert[i][1] for i in 1:length(t)]
            lamberty = [lambert[i][2] for i in 1:length(t)]
            lambertz = [lambert[i][3] for i in 1:length(t)]
            if i == 1
                lines!(ax, lambertx, lamberty, lambertz, color=:green, label="Primer")
            else
                lines!(ax, lambertx, lamberty, lambertz, color=:green)
            end
        end
    end

    if !isnothing(collocationState)
        lines!(ax, collocationState[:, 1], collocationState[:, 2], collocationState[:, 3], color=:orange, label="Collocation")
    end
    if !isnothing(pseudoState)
        lines!(ax, pseudoState[:, 1], pseudoState[:, 2], pseudoState[:, 3], color=:red, label="Pseudospectral")
    end

    if !isnothing(psPrimer)
        for i in 1:length(psPrimer)-1
            x0p = [psPrimer[i].position...; psPrimer[i].outVelocity...]
            prob = ODEProblem(dstate, x0p, (psPrimer[i].time, psPrimer[i+1].time), 0)
            sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
            t = range(psPrimer[i].time, psPrimer[i+1].time, 100)
            lambert = sol.(t)
            lambertx = [lambert[i][1] for i in 1:length(t)]
            lamberty = [lambert[i][2] for i in 1:length(t)]
            lambertz = [lambert[i][3] for i in 1:length(t)]
            if i == 1
                lines!(ax, lambertx, lamberty, lambertz, color=:salmon, label="PSPrimer")
            else
                lines!(ax, lambertx, lamberty, lambertz, color=:salmon)
            end
        end
    end
    axislegend(ax, position=:lb)

    xlims!(ax, -43000, 43000)
    ylims!(ax, -43000, 43000)
    # zlims!(ax, -43000, 43000)

    display(GLMakie.Screen(), fig)
end

function plot_results2(
    x0,
    x1,
    tof;
    primerNodes=nothing,
    collocationState=nothing,
    pseudoState=nothing,
    psPrimer=nothing,
)
    r0 = x0[1:3]
    rf = x1[1:3]
    v0 = x0[4:6]
    vf = x1[4:6]
    x1 = [x1...]
    mu = 3.986e5
    n = _get_default_half_revs(0.5 * (norm(x0[1:3]) + norm(x1[1:3])), tof, mu=mu)

    # initial orbit
    prob = ODEProblem(dstate, x0, (0, tof), 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    t = 0.0:10:tof
    initial = sol.(t)
    initialx = [initial[i][1] for i in 1:length(t)]
    initialy = [initial[i][2] for i in 1:length(t)]
    initialz = [initial[i][3] for i in 1:length(t)]

    # final orbit
    prob = ODEProblem(dstate, x1, (tof, 0), 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    final = sol.(t)
    finalx = [final[i][1] for i in 1:length(t)]
    finaly = [final[i][2] for i in 1:length(t)]
    finalz = [final[i][3] for i in 1:length(t)]

    vxfer, vstop = basic_lambert(r0, v0, rf, tof, n, verbose=false, v2=vf)
    x0p = [r0; vxfer]
    prob = ODEProblem(dstate, x0p, (0, tof), 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    lambert = sol.(t)
    lambertx = [lambert[i][1] for i in 1:length(t)]
    lamberty = [lambert[i][2] for i in 1:length(t)]
    lambertz = [lambert[i][3] for i in 1:length(t)]

    set_theme!(theme_black())
    fig = Figure(size=(800, 800))

    ax = Axis(fig[1, 1])
    lines!(ax, t, initialx, color=:yellow, label="Initial Orbit")
    lines!(ax, t, finalx, color=:magenta, label="Final Orbit")

    lines!(ax, t, lambertx, color=:cyan, label="Lambert Soln")

    ax2 = Axis(fig[2, 1])
    lines!(ax2, t, initialy, color=:yellow)
    lines!(ax2, t, finaly, color=:magenta)

    lines!(ax2, t, lamberty, color=:cyan, label="Lambert Soln")

    ax3 = Axis(fig[3, 1])
    lines!(ax3, t, initialz, color=:yellow)
    lines!(ax3, t, finalz, color=:magenta)

    lines!(ax3, t, lambertz, color=:cyan, label="Lambert Soln")

    if !isnothing(primerNodes)
        for i in 1:length(primerNodes)-1
            x0p = [primerNodes[i].position...; primerNodes[i].outVelocity...]
            prob = ODEProblem(dstate, x0p, (primerNodes[i].time, primerNodes[i+1].time), 0)
            sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
            t = range(primerNodes[i].time, primerNodes[i+1].time, 100)
            lambert = sol.(t)
            lambertx = [lambert[i][1] for i in 1:length(t)]
            lamberty = [lambert[i][2] for i in 1:length(t)]
            lambertz = [lambert[i][3] for i in 1:length(t)]
            if i == 1
                lines!(ax, t, lambertx, color=:green, label="Primer")
                lines!(ax2, t, lamberty, color=:green, label="Primer")
                lines!(ax3, t, lambertz, color=:green, label="Primer")
            else
                lines!(ax, t, lambertx, color=:green)
                lines!(ax2, t, lamberty, color=:green)
                lines!(ax3, t, lambertz, color=:green)
            end
        end
    end

    if !isnothing(collocationState)
        lines!(ax, t, collocationState[:, 1], color=:orange, label="Collocation")
        lines!(ax2, t, collocationState[:, 2], color=:orange, label="Collocation")
        lines!(ax3, t, collocationState[:, 3], color=:orange, label="Collocation")
    end
    if !isnothing(pseudoState)
        lines!(ax, t, pseudoState[:, 1], color=:red, label="Pseudospectral")
        lines!(ax2, t, pseudoState[:, 2], color=:red, label="Pseudospectral")
        lines!(ax3, t, pseudoState[:, 3], color=:red, label="Pseudospectral")
    end

    if !isnothing(psPrimer)
        for i in 1:length(psPrimer)-1
            x0p = [psPrimer[i].position...; psPrimer[i].outVelocity...]
            prob = ODEProblem(dstate, x0p, (psPrimer[i].time, psPrimer[i+1].time), 0)
            sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
            t = range(psPrimer[i].time, psPrimer[i+1].time, 100)
            lambert = sol.(t)
            lambertx = [lambert[i][1] for i in 1:length(t)]
            lamberty = [lambert[i][2] for i in 1:length(t)]
            lambertz = [lambert[i][3] for i in 1:length(t)]
            if i == 1
                lines!(ax, t, lambertx, color=:salmon, label="PSPrimer")
                lines!(ax2, t, lamberty, color=:salmon, label="PSPrimer")
                lines!(ax3, t, lambertz, color=:salmon, label="PSPrimer")
            else
                lines!(ax, t, lambertx, color=:salmon)
                lines!(ax2, t, lamberty, color=:salmon)
                lines!(ax3, t, lambertz, color=:salmon)
            end
        end
    end
    axislegend(ax, position=:lb)

    # zlims!(ax, -43000, 43000)

    display(GLMakie.Screen(), fig)
end
