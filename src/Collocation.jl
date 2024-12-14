function collocation_lambert(
    x0::AbstractVector, #Starting state vector (position and velocity)
    x1::AbstractVector, #Final state vector (position or position and velocity)
    tof::AbstractFloat; #time of flight
    n::Int=-1, # Number of half revolutions between r1 and r2
    mu::AbstractFloat=3.986e5, #grav parameter for units used
    verbose::Bool=false, #whether or not to print debug statements
    num_supports::Int=251,
    constrain_u::Bool=false,
)
    # Input Checking
    if length(x0) != 6
        error("Initial state vector must be of length 6")
    end
    if length(x1) == 3
        positionOnly = true
    elseif length(x1) == 6
        positionOnly = false
    else
        error("Final state vector must be of length 3 or 6")
    end

    # CONVERT TO CANONICAL IF NEEDED 
    # (this is messy, will need to clean up if this becomes real later)
    if mu == 1.0
        x0c = [x0...] ./ DISTANCE_UNIT
        x0c[4:6] .*= TIME_UNIT
        x1c = [x1...] ./ DISTANCE_UNIT
        x1c[4:6] .*= TIME_UNIT
        tof /= TIME_UNIT
    else
        x0c = [x0...]
        x1c = [x1...]
    end

    if n < 0
        n = _get_default_half_revs(0.5 * (norm(x0c[1:3]) + norm(x1c[1:3])), tof, mu=mu)
    end

    # DEFINE THE PROBLEM CONSTANTS
    r0 = x0c[1:3]
    rf = x1c[1:3]
    v0 = x0c[4:6]
    vf = x1c[4:6]
    verbose && (@info "Setting up initial seed")
    vxfer, vstop = basic_lambert(r0, v0, rf, tof, n, verbose=true, v2=vf, mu=mu)
    x0p = [r0; vxfer]
    prob = ODEProblem(dstate_canonical, x0p, (0, tof), 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    guess_traj = [(t) -> sol(t)[i] for i in 1:6]
    minaltsq = (1 + 200 / DISTANCE_UNIT)^2
    dv0 = vxfer - x0c[4:6]
    dv1 = x1c[4:6] - vstop
    dvinit = norm(dv0) + norm(dv1)
    verbose && (@info "Initial Δv cost: $dvinit km/s")

    # INITIALIZE THE MODEL
    verbose && (@info "Creating the model")
    if verbose
        m = InfiniteModel(Ipopt.Optimizer)
    else
        m = InfiniteModel(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    end

    # INITIALIZE THE PARAMETERS
    @infinite_parameter(m, t in [0, tof], num_supports = num_supports, derivative_method = OrthogonalCollocation(4))

    # INITIALIZE THE VARIABLES
    @variable(m, r[1:3], Infinite(t))
    @variable(m, v[1:3], Infinite(t))
    if constrain_u
        maxAccel = 10 * dvinit / tof
        # @variable(m, -0.00001 <= u[1:3] <= 0.00001, Infinite(t), start = 0)
        @variable(m, -maxAccel <= u[1:3] <= maxAccel, Infinite(t), start = 0)
    else
        @variable(m, u[1:3], Infinite(t), start = 0)
    end
    set_start_value_function(r[1], guess_traj[1])
    set_start_value_function(r[2], guess_traj[2])
    set_start_value_function(r[3], guess_traj[3])
    set_start_value_function(v[1], guess_traj[4])
    set_start_value_function(v[2], guess_traj[5])
    set_start_value_function(v[3], guess_traj[6])

    # SET THE OBJECTIVE
    # @objective(m, Min, ∫(sqrt(1e-11 + u[1]^2 + u[2]^2 + u[3]^2), t))
    @objective(m, Min, ∫(u[1]^2 + u[2]^2 + u[3]^2, t))

    # SET THE BOUNDARY CONDITIONS
    @constraint(m, [i = 1:3], r[i](0) == r0[i])
    @constraint(m, [i = 1:3], r[i](tof) == rf[i])
    @constraint(m, [i = 1:3], v[i](0) == v0[i])
    @constraint(m, [i = 1:3], v[i](tof) == vf[i])

    # SET THE PROBLEM CONSTRAINTS
    @constraint(m, [i = 1:3], ∂(r[i], t) == v[i])
    @constraint(m, [i = 1:3], ∂(v[i], t) == u[i] - mu * r[i] / (r[1]^2 + r[2]^2 + r[3]^2)^(3 / 2))
    @constraint(m, r[1]^2 + r[2]^2 + r[3]^2 >= minaltsq)

    # SOLVE THE MODEL
    verbose && (@info "Solving...")
    optimize!(m)

    # GET THE RESULTS
    verbose && (@info "$(termination_status(m))")
    r_opt = value.(r)
    r_opt = [r_opt[1] r_opt[2] r_opt[3]]
    v_opt = value.(v)
    v_opt = [v_opt[1] v_opt[2] v_opt[3]]
    u_opt = value.(u)
    u_opt = [u_opt[1] u_opt[2] u_opt[3]]
    u_ts = supports.(u)
    u_ts = [u_ts[1][i][1] for i in 1:length(u_ts[1])]
    u_ts *= TIME_UNIT
    u_opt *= DISTANCE_UNIT / TIME_UNIT^2
    r_opt *= DISTANCE_UNIT
    v_opt *= DISTANCE_UNIT / TIME_UNIT
    verbose && (@info "Final Δv cost: $(get_collocation_cost(u_ts, u_opt)) km/s")

    return m, u_ts, u_opt, r_opt, v_opt
end

function get_collocation_cost(t, u)
    t3 = range(0, t[end], length(t))
    u3 = [norm(u[i, :]) for i in 1:length(t)]
    integrand = cubic_spline_interpolation(t3, u3)
    dv, _ = quadgk(x -> integrand(x), t[1], t[end], rtol=1e-8)
    return dv
end
