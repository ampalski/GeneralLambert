# Testing 10th order
# a = 1 / 256 * [-63, 0, 3465, 0, -30030, 0, 90090, 0, -109395, 0, 46189]
# p = Polynomial(a)
# pd = derivative(p)
# xd = roots(pd)
# sort!(xd)
# pushfirst!(xd, -1)
# push!(xd, 1)
function _get_nodes_and_weights(N::Int)
    return gausslobatto(N + 1)
end

function _differentiation_matrix(tN)
    N = length(tN) - 1
    kl0 = -N * (N + 1.0) / 4.0
    klN = N * (N + 1.0) / 4.0
    D = zeros(N + 1, N + 1)

    for k in 1:N+1
        for l in 1:N+1
            if k == l && k == 1
                D[k, l] = kl0
            elseif k == l && k == N + 1
                D[k, l] = klN
            elseif k != l
                D[k, l] = Pl(tN[k], N) / Pl(tN[l], N) / (tN[k] - tN[l])
            end
        end
    end
    return D
end

# included here to compare against infiniteopt single-node to single-node
function testInfOpt2(; plot::Bool=true)

    # DEFINE THE PROBLEM CONSTANTS
    xw = [1 4; 1 3] # positions
    tw = [0, 60]    # times

    # INITIALIZE THE MODEL
    # m = InfiniteModel(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    m = InfiniteModel(Ipopt.Optimizer)

    # INITIALIZE THE PARAMETERS
    @infinite_parameter(m, t in [0, 60], num_supports = 11, derivative_method = OrthogonalCollocation(4))
    # @infinite_parameter(model, t in [0, 60], num_supports = 61, derivative_method = OrthogonalCollocation(2))

    # INITIALIZE THE VARIABLES
    @variable(m, x[1:2], Infinite(t), start = 1)
    @variable(m, v[1:2], Infinite(t), start = 0)
    @variable(m, u[1:2], Infinite(t), start = 0)

    # SET THE OBJECTIVE
    # @objective(model, Min, integral(sum(u[i]^2 for i in I), t))
    @objective(m, Min, ∫(u[1]^2 + u[2]^2, t))

    # SET THE INITIAL CONDITIONS
    @constraint(m, [i = 1:2], v[i](0) == 0)
    @constraint(m, [i = 1:2], v[i](60) == 0)

    # SET THE PROBLEM CONSTRAINTS
    @constraint(m, [i = 1:2], ∂(x[i], t) == v[i])
    @constraint(m, [i = 1:2], ∂(v[i], t) == u[i])
    # @constraint(m, [i = 1], ∂(v[i], t) == u[i] - x[i] / (x[i]^2 + x[i+1]^2))
    # @constraint(m, [i = 1:2], ∂(v[i], t) == u[i] - x[i] / (x[1] + x[2]^2)^(3 / 2))
    # @constraint(m, [i = 2], ∂(v[i], t) == u[i] - x[i] / (x[1] + x[2]^2))
    @constraint(m, [i = 1:2, j = eachindex(tw)], x[i](tw[j]) == xw[i, j])
    @constraint(m, x[1]^2 + x[2]^2 >= 0.05)

    # SOLVE THE MODEL
    # optimize!(model)
    optimize!(m)

    # GET THE RESULTS
    # termination_status(model)
    termination_status(m)
    # opt_obj = objective_value(model)
    opt_obj = objective_value(m)
    x_opt = value.(x)
    v_opt = value.(v)
    u_opt = value.(u)
    u_ts = supports.(u)
    t_opt = value(t)

    if plot
        set_theme!(theme_black())
        fig = Figure(size=(800, 800))
        ax = Axis(fig[1, 1])
        ax2 = Axis(fig[2, 1])
        lines!(ax, t_opt, x_opt[1])
        lines!(ax, t_opt, x_opt[2])
        lines!(ax2, t_opt, v_opt[1])
        lines!(ax2, t_opt, v_opt[2])
        display(GLMakie.Screen(), fig)
    end
    return m
end


function testPseudospectral(; plot::Bool=true)
    #Constants
    N = 10
    xw = [1 4; 1 3] # positions
    tw = [0, 60]    # times
    tN, wN = _get_nodes_and_weights(N)
    D = _differentiation_matrix(tN)

    m = Model(Ipopt.Optimizer)

    @variable(m, x1[1:N+1], start = 1)
    @variable(m, x2[1:N+1], start = 1)
    @variable(m, v1[1:N+1], start = 0)
    @variable(m, v2[1:N+1], start = 0)
    @variable(m, u1[1:N+1], start = 0)
    @variable(m, u2[1:N+1], start = 0)

    fix(x1[1], xw[1, 1]; force=true)
    fix(x2[1], xw[2, 1]; force=true)
    fix(v1[1], 0.0; force=true)
    fix(v2[1], 0.0; force=true)

    fix(x1[end], xw[1, 2]; force=true)
    fix(x2[end], xw[2, 2]; force=true)
    fix(v1[end], 0.0; force=true)
    fix(v2[end], 0.0; force=true)

    @objective(m, Min, sum(i -> (u1[i]^2 + u2[i]^2) * wN[i], 1:N+1))

    for k in 1:N+1
        @constraint(m, (tw[2] - tw[1]) / 2 * v1[k] - sum(l -> D[k, l] * x1[l], 1:N+1) == 0)
        @constraint(m, (tw[2] - tw[1]) / 2 * v2[k] - sum(l -> D[k, l] * x2[l], 1:N+1) == 0)
        @constraint(m, (tw[2] - tw[1]) / 2 * u1[k] - sum(l -> D[k, l] * v1[l], 1:N+1) == 0)
        @constraint(m, (tw[2] - tw[1]) / 2 * u2[k] - sum(l -> D[k, l] * v2[l], 1:N+1) == 0)
    end

    optimize!(m)

    termination_status(m)
    # opt_obj = objective_value(model)
    opt_obj = objective_value(m)
    x1_opt = value.(x1)
    x2_opt = value.(x2)
    v1_opt = value.(v1)
    v2_opt = value.(v2)
    u_opt = value.(u)
    u_ts = supports.(u)

    if plot
        set_theme!(theme_black())
        fig = Figure(size=(800, 800))
        ax = Axis(fig[1, 1])
        ax2 = Axis(fig[2, 1])
        lines!(ax, tN, x1_opt)
        lines!(ax, tN, x2_opt)
        lines!(ax2, tN, v1_opt)
        lines!(ax2, tN, v2_opt)
        display(GLMakie.Screen(), fig)
    end
    return m
end
