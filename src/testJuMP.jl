using QuadGK, Interpolations
using InfiniteOpt, Ipopt, GLMakie, DifferentialEquations, StaticArrays
function testInfOpt(; plot::Bool=true)

    # DEFINE THE PROBLEM CONSTANTS
    xw = [1 4 6 1; 1 3 0 1] # positions
    tw = [0, 25.5, 50, 60]    # times
    grav = [0, -9.8]
    # xw = [1 6; 1 6]
    # tw = [0, 60]

    # INITIALIZE THE MODEL
    # m = InfiniteModel(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    m = InfiniteModel(Ipopt.Optimizer)

    # INITIALIZE THE PARAMETERS
    @infinite_parameter(m, t in [0, 60], num_supports = 11, derivative_method = OrthogonalCollocation(4))
    # @infinite_parameter(model, t in [0, 60], num_supports = 61, derivative_method = OrthogonalCollocation(2))

    # INITIALIZE THE VARIABLES
    # @variables(m, begin
    #     # state variables
    #     x[1:2], Infinite(t)
    #     v[1:2], Infinite(t)
    #     # control variables
    #     u[1:2], Infinite(t), (start = 0)
    # end)
    @variable(m, x[1:2], Infinite(t), start = 1)
    @variable(m, v[1:2], Infinite(t), start = 0)
    @variable(m, u[1:2], Infinite(t), start = 0)

    # SET THE OBJECTIVE
    # @objective(model, Min, integral(sum(u[i]^2 for i in I), t))
    @objective(m, Min, ∫(u[1]^2 + u[2]^2, t))

    # SET THE INITIAL CONDITIONS
    @constraint(m, [i = 1:2], v[i](0) == 0)
    @constraint(m, [i = 1:2], v[i](60) == 0)
    # @constraint(model, [i in I], x[i](0, ξ) == x0[i])
    # @constraint(model, [i in I], v[i](0, ξ) == v0[i])

    # SET THE PROBLEM CONSTRAINTS
    # @constraint(model, c1[i in I], @deriv(x[i], t) == v[i])
    # @constraint(model, c2[i in I], ξ * @deriv(v[i], t) == u[i])
    # @constraint(model, c3[w in W], y[w] == sum((x[i](tw[w], ξ) - p[i, w])^2 for i in I))
    # @constraint(model, c4, expect(sum(y[w] for w in W), ξ) <= ϵ)
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
    u_opt = value.(u)
    u_ts = supports.(u)
    t_opt = value(t)

    if plot
        set_theme!(theme_black())
        fig = Figure(size=(800, 800))
        ax = Axis(fig[1, 1])
        scatter!(ax, xw[1, :], xw[2, :])
        lines!(ax, x_opt[1], x_opt[2])
        display(GLMakie.Screen(), fig)
    end
    return m
end
function testInfOptOrbit(; plot::Bool=true)

    # DEFINE THE PROBLEM CONSTANTS
    tof = 43200.0 / 2
    r0 = [-4927.10914, 6032.53541, -1828.1225]
    v0 = [-1.9319808, -3.36505695, -5.89586078]
    xf0 = [-30726.027555, 28902.72485, -3.75426, -2.10596, -2.23904, -0.001]
    rf, vf = universalkepler(SA[xf0[1:3]...], SA[xf0[4:6]...], tof, 3.986e5)
    # rf = [30923.13840, -28681.24808, 3.849952]
    # vf = [2.090757, 2.2539514889, 0.00099825]
    vxfer, _ = basic_lambert(r0, v0, rf, tof, 1, verbose=false, v2=vf)
    x0p = [r0; vxfer]
    prob = ODEProblem(dstate, x0p, (0, tof), 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    guess_traj = [(t) -> sol(t)[i] for i in 1:6]
    # xw = [1 6; 1 6]
    # tw = [0, 60]
    μ = 3.986e5
    Rearth = 6378.0
    minaltsq = (Rearth + 200)^2

    # INITIALIZE THE MODEL
    # m = InfiniteModel(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    m = InfiniteModel(Ipopt.Optimizer)

    # INITIALIZE THE PARAMETERS
    @infinite_parameter(m, t in [0, tof], num_supports = 251, derivative_method = OrthogonalCollocation(4))
    # @infinite_parameter(model, t in [0, 60], num_supports = 61, derivative_method = OrthogonalCollocation(2))

    # INITIALIZE THE VARIABLES
    # @variable(m, r[1:3], Infinite(t), start = guess_traj[1:3])
    # @variable(m, v[1:3], Infinite(t), start = guess_traj[4:6])
    @variable(m, r[1:3], Infinite(t))
    @variable(m, v[1:3], Infinite(t))
    @variable(m, u[1:3], Infinite(t), start = 0)
    set_start_value_function(r[1], guess_traj[1])
    set_start_value_function(r[2], guess_traj[2])
    set_start_value_function(r[3], guess_traj[3])
    set_start_value_function(v[1], guess_traj[4])
    set_start_value_function(v[2], guess_traj[5])
    set_start_value_function(v[3], guess_traj[6])

    # SET THE OBJECTIVE
    # @objective(model, Min, integral(sum(u[i]^2 for i in I), t))
    @objective(m, Min, ∫(u[1]^2 + u[2]^2 + u[3]^2, t))

    # SET THE BOUNDARY CONDITIONS
    @constraint(m, [i = 1:3], r[i](0) == r0[i])
    @constraint(m, [i = 1:3], r[i](tof) == rf[i])
    @constraint(m, [i = 1:3], v[i](0) == v0[i])
    @constraint(m, [i = 1:3], v[i](tof) == vf[i])

    # SET THE PROBLEM CONSTRAINTS
    @constraint(m, [i = 1:3], ∂(r[i], t) == v[i])
    @constraint(m, [i = 1:3], ∂(v[i], t) == u[i] - μ * r[i] / (r[1]^2 + r[2]^2 + r[3]^2)^(3 / 2))
    @constraint(m, r[1]^2 + r[2]^2 + r[3]^2 >= minaltsq)

    # SOLVE THE MODEL
    # optimize!(model)
    optimize!(m)

    # GET THE RESULTS
    # termination_status(model)
    termination_status(m)
    # opt_obj = objective_value(model)
    opt_obj = objective_value(m)
    r_opt = value.(r)
    u_opt = value.(u)
    u3 = [norm([u_opt[1][i], u_opt[2][i], u_opt[3][i]]) for i in 1:length(value(t))]
    u_ts = supports.(u)
    t_opt = value(t)
    dt = t_opt[2] - t_opt[1]
    t3 = 0:dt:tof
    integrand = cubic_spline_interpolation(t3, u3)
    dv, _ = quadgk(x -> integrand(x), t_opt[1], t_opt[end], rtol=1e-8)

    if plot
        set_theme!(theme_black())
        fig = Figure(size=(800, 800))
        ax = Axis3(fig[1, 1])
        scatter!(ax, r0[1], r0[2], r0[3]; color=:orange)
        scatter!(ax, rf[1], rf[2], rf[3]; color=:orange)
        lines!(ax, r_opt[1], r_opt[2], r_opt[3])
        display(GLMakie.Screen(), fig)
    end
    return m, r, u, t, dv
    # Need to add a numeric integrator to calculate the actual value of u:
    # take the norm at each timestep, integrate that.
end
