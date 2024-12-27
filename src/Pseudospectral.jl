function pseudospectral_continuous_lambert(
    x0::AbstractVector, #Starting state vector (position and velocity)
    x1::AbstractVector, #Final state vector (position or position and velocity)
    tof::AbstractFloat; #time of flight
    n::Int=-1, # Number of half revolutions between r1 and r2
    mu::AbstractFloat=1.0, #grav parameter for units used, for canonical units
    verbose::Bool=false, #whether or not to print debug statements
    poly_order::Int=60,
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
    N = poly_order
    tN, wN = _get_nodes_and_weights(N)
    D = _differentiation_matrix(tN)
    tauN = 0.5 .* (tof .* tN .+ tof)

    # INITIALIZE THE MODEL
    verbose && (@info "Creating the model")
    m = Model(Ipopt.Optimizer)
    if !verbose
        set_silent(m)
    end

    # INITIALIZE THE VARIABLES
    @variable(m, a1[1:N+1])
    @variable(m, a2[1:N+1])
    @variable(m, a3[1:N+1])
    @variable(m, a4[1:N+1])
    @variable(m, a5[1:N+1])
    @variable(m, a6[1:N+1])

    vals = [guess_traj[1](tau) for tau in tauN]
    set_start_value.(a1, vals)
    vals = [guess_traj[2](tau) for tau in tauN]
    set_start_value.(a2, vals)
    vals = [guess_traj[3](tau) for tau in tauN]
    set_start_value.(a3, vals)
    vals = [guess_traj[4](tau) for tau in tauN]
    set_start_value.(a4, vals)
    vals = [guess_traj[5](tau) for tau in tauN]
    set_start_value.(a5, vals)
    vals = [guess_traj[6](tau) for tau in tauN]
    set_start_value.(a6, vals)

    if constrain_u
        maxDV = 10 * dvinit / tof
        @variable(m, -maxDV <= b1[1:N+1] <= maxDV, start = 0)
        @variable(m, -maxDV <= b2[1:N+1] <= maxDV, start = 0)
        @variable(m, -maxDV <= b3[1:N+1] <= maxDV, start = 0)
    else
        @variable(m, b1[1:N+1], start = dvinit / tof)
        @variable(m, b2[1:N+1], start = dvinit / tof)
        @variable(m, b3[1:N+1], start = dvinit / tof)
    end

    # Constrain beginning and end

    fix(a1[1], r0[1]; force=true)
    fix(a2[1], r0[2]; force=true)
    fix(a3[1], r0[3]; force=true)
    fix(a4[1], v0[1]; force=true)
    fix(a5[1], v0[2]; force=true)
    fix(a6[1], v0[3]; force=true)

    fix(a1[end], rf[1]; force=true)
    fix(a2[end], rf[2]; force=true)
    fix(a3[end], rf[3]; force=true)
    fix(a4[end], vf[1]; force=true)
    fix(a5[end], vf[2]; force=true)
    fix(a6[end], vf[3]; force=true)

    # Set objective
    # WN = diagm(wN)
    # @expression(m, quad_term, b1' * WN * b1 + b2' * WN * b2 + b3' * WN * b3)
    # @objective(m, Min, (tof - 0) / 2 * sum(i -> sqrt(1e-12 + (b1[i]^2 + b2[i]^2 + b3[i]^2)) * wN[i], 1:N+1))
    @objective(m, Min, (tof - 0) / 2 * sum(i -> (b1[i]^2 + b2[i]^2 + b3[i]^2) * wN[i], 1:N+1))
    # @NLobjective(m, Min, (tof - 0) / 2 * sqrt(quad_term))

    # Set dynamic constraints

    @constraint(m, c1[k=1:N+1], (tof - 0) / 2 * a4[k] - sum(l -> D[k, l] * a1[l], 1:N+1) == 0)
    @constraint(m, c2[k=1:N+1], (tof - 0) / 2 * a5[k] - sum(l -> D[k, l] * a2[l], 1:N+1) == 0)
    @constraint(m, c3[k=1:N+1], (tof - 0) / 2 * a6[k] - sum(l -> D[k, l] * a3[l], 1:N+1) == 0)
    @constraint(m, c4[k=1:N+1], (tof - 0) / 2 * (b1[k] - mu * a1[k] / (a1[k]^2 + a2[k]^2 + a3[k]^2)^(3 / 2)) - sum(l -> D[k, l] * a4[l], 1:N+1) == 0)
    @constraint(m, c5[k=1:N+1], (tof - 0) / 2 * (b2[k] - mu * a2[k] / (a1[k]^2 + a2[k]^2 + a3[k]^2)^(3 / 2)) - sum(l -> D[k, l] * a5[l], 1:N+1) == 0)
    @constraint(m, c6[k=1:N+1], (tof - 0) / 2 * (b3[k] - mu * a3[k] / (a1[k]^2 + a2[k]^2 + a3[k]^2)^(3 / 2)) - sum(l -> D[k, l] * a6[l], 1:N+1) == 0)

    # for k in 1:N+1
    #     @constraint(m, (tof - 0) / 2 * a4[k] - sum(l -> D[k, l] * a1[l], 1:N+1) == 0)
    #     @constraint(m, (tof - 0) / 2 * a5[k] - sum(l -> D[k, l] * a2[l], 1:N+1) == 0)
    #     @constraint(m, (tof - 0) / 2 * a6[k] - sum(l -> D[k, l] * a3[l], 1:N+1) == 0)
    #     @constraint(m, (tof - 0) / 2 * (b1[k] - mu * a1[k] / (a1[k]^2 + a2[k]^2 + a3[k]^2)^(3 / 2)) - sum(l -> D[k, l] * a4[l], 1:N+1) == 0)
    #     @constraint(m, (tof - 0) / 2 * (b2[k] - mu * a2[k] / (a1[k]^2 + a2[k]^2 + a3[k]^2)^(3 / 2)) - sum(l -> D[k, l] * a5[l], 1:N+1) == 0)
    #     @constraint(m, (tof - 0) / 2 * (b3[k] - mu * a3[k] / (a1[k]^2 + a2[k]^2 + a3[k]^2)^(3 / 2)) - sum(l -> D[k, l] * a6[l], 1:N+1) == 0)
    #
    # end
    # Path constraints
    @constraint(m, [k = 1:N+1], a1[k]^2 + a2[k]^2 + a3[k]^2 >= minaltsq)


    # SOLVE THE MODEL
    verbose && (@info "Solving...")
    optimize!(m)

    # GET THE RESULTS
    verbose && (@info "$(termination_status(m))")
    x_pos = value.(a1) * DISTANCE_UNIT
    y_pos = value.(a2) * DISTANCE_UNIT
    z_pos = value.(a3) * DISTANCE_UNIT
    # r_opt = [r_opt[1] r_opt[2] r_opt[3]]
    # u_opt = value.(u)
    # u_opt = [u_opt[1] u_opt[2] u_opt[3]]
    # u_ts = supports.(u)
    # u_ts = [u_ts[1][i][1] for i in 1:length(u_ts[1])]
    u1 = value.(b1) * DISTANCE_UNIT / TIME_UNIT^2
    u2 = value.(b2) * DISTANCE_UNIT / TIME_UNIT^2
    u3 = value.(b3) * DISTANCE_UNIT / TIME_UNIT^2

    # verbose && (@info "Final Δv cost: $(get_collocation_cost(u_ts, u_opt)) km/s")

    # return m, u_ts, u_opt, r_opt
    return m, [x_pos y_pos z_pos], [u1 u2 u3]
end
