# function pseudospectral_continuous_lambert(
#     x0::AbstractVector, #Starting state vector (position and velocity)
#     x1::AbstractVector, #Final state vector (position or position and velocity)
#     tof::AbstractFloat; #time of flight
#     n::Int=-1, # Number of half revolutions between r1 and r2
#     mu::AbstractFloat=3.986e5, #grav parameter for units used
#     verbose::Bool=false, #whether or not to print debug statements
#     poly_order::Int=60,
#     constrain_u::Bool=false,
# )
#     # Input Checking
#     if length(x0) != 6
#         error("Initial state vector must be of length 6")
#     end
#     if length(x1) == 3
#         positionOnly = true
#     elseif length(x1) == 6
#         positionOnly = false
#     else
#         error("Final state vector must be of length 3 or 6")
#     end
#
#     if n < 0
#         n = _get_default_half_revs(0.5 * (norm(x0[1:3]) + norm(x1[1:3])), tof, mu=mu)
#     end
#
#     # DEFINE THE PROBLEM CONSTANTS
#     r0 = x0[1:3]
#     rf = x1[1:3]
#     v0 = x0[4:6]
#     vf = x1[4:6]
#     verbose && (@info "Setting up initial seed")
#     vxfer, vstop = basic_lambert(r0, v0, rf, tof, n, verbose=false, v2=vf)
#     x0p = [r0; vxfer]
#     prob = ODEProblem(dstate, x0p, (0, tof), 0)
#     sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
#     guess_traj = [(t) -> sol(t)[i] for i in 1:6]
#     Rearth = 6378.0
#     minaltsq = (Rearth + 200)^2
#     dv0 = vxfer - x0[4:6]
#     dv1 = x1[4:6] - vstop
#     dvinit = norm(dv0) + norm(dv1)
#     verbose && (@info "Initial Δv cost: $dvinit km/s")
#     N = poly_order
#     tN, wN = _get_nodes_and_weights(N)
#     D = _differentiation_matrix(tN)
#     tauN = 0.5 .* (tof .* tN .+ tof)
#
#     # INITIALIZE THE MODEL
#     verbose && (@info "Creating the model")
#     m = Model(Ipopt.Optimizer)
#     if !verbose
#         set_silent(m)
#     end
#
#     # INITIALIZE THE VARIABLES
#     @variable(m, a1[1:N+1])
#     @variable(m, a2[1:N+1])
#     @variable(m, a3[1:N+1])
#     @variable(m, a4[1:N+1])
#     @variable(m, a5[1:N+1])
#     @variable(m, a6[1:N+1])
#
#     vals = [guess_traj[1](tau) for tau in tauN]
#     set_start_value.(a1, vals)
#     vals = [guess_traj[2](tau) for tau in tauN]
#     set_start_value.(a2, vals)
#     vals = [guess_traj[3](tau) for tau in tauN]
#     set_start_value.(a3, vals)
#     vals = [guess_traj[4](tau) for tau in tauN]
#     set_start_value.(a4, vals)
#     vals = [guess_traj[5](tau) for tau in tauN]
#     set_start_value.(a5, vals)
#     vals = [guess_traj[6](tau) for tau in tauN]
#     set_start_value.(a6, vals)
#
#     if constrain_u
#         maxDV = 10 * dvinit / tof
#         @variable(m, -maxDV <= b1[1:N+1] <= maxDV, start = 0)
#         @variable(m, -maxDV <= b2[1:N+1] <= maxDV, start = 0)
#         @variable(m, -maxDV <= b3[1:N+1] <= maxDV, start = 0)
#     else
#         @variable(m, b1[1:N+1], start = dvinit / tof)
#         @variable(m, b2[1:N+1], start = dvinit / tof)
#         @variable(m, b3[1:N+1], start = dvinit / tof)
#     end
#
#     # Constrain beginning and end
#
#     fix(a1[1], r0[1]; force=true)
#     fix(a2[1], r0[2]; force=true)
#     fix(a3[1], r0[3]; force=true)
#     fix(a4[1], v0[1]; force=true)
#     fix(a5[1], v0[2]; force=true)
#     fix(a6[1], v0[3]; force=true)
#
#     fix(a1[end], rf[1]; force=true)
#     fix(a2[end], rf[2]; force=true)
#     fix(a3[end], rf[3]; force=true)
#     fix(a4[end], vf[1]; force=true)
#     fix(a5[end], vf[2]; force=true)
#     fix(a6[end], vf[3]; force=true)
#
#     # Set objective
#     # WN = diagm(wN)
#     # @expression(m, quad_term, b1' * WN * b1 + b2' * WN * b2 + b3' * WN * b3)
#     # @objective(m, Min, (tof - 0) / 2 * sum(i -> sqrt(1e-12 + (b1[i]^2 + b2[i]^2 + b3[i]^2)) * wN[i], 1:N+1))
#     @objective(m, Min, (tof - 0) / 2 * sum(i -> (b1[i]^2 + b2[i]^2 + b3[i]^2) * wN[i], 1:N+1))
#     # @NLobjective(m, Min, (tof - 0) / 2 * sqrt(quad_term))
#
#     # Set dynamic constraints
#
#     for k in 1:N+1
#         @constraint(m, (tof - 0) / 2 * a4[k] - sum(l -> D[k, l] * a1[l], 1:N+1) == 0)
#         @constraint(m, (tof - 0) / 2 * a5[k] - sum(l -> D[k, l] * a2[l], 1:N+1) == 0)
#         @constraint(m, (tof - 0) / 2 * a6[k] - sum(l -> D[k, l] * a3[l], 1:N+1) == 0)
#         @constraint(m, (tof - 0) / 2 * (b1[k] - mu * a1[k] / (a1[k]^2 + a2[k]^2 + a3[k]^2)^(3 / 2)) - sum(l -> D[k, l] * a4[l], 1:N+1) == 0)
#         @constraint(m, (tof - 0) / 2 * (b2[k] - mu * a2[k] / (a1[k]^2 + a2[k]^2 + a3[k]^2)^(3 / 2)) - sum(l -> D[k, l] * a5[l], 1:N+1) == 0)
#         @constraint(m, (tof - 0) / 2 * (b3[k] - mu * a3[k] / (a1[k]^2 + a2[k]^2 + a3[k]^2)^(3 / 2)) - sum(l -> D[k, l] * a6[l], 1:N+1) == 0)
#
#     end
#     # Path constraints
#     @constraint(m, [k = 1:N+1], a1[k]^2 + a2[k]^2 + a3[k]^2 >= minaltsq)
#
#
#     # SOLVE THE MODEL
#     verbose && (@info "Solving...")
#     optimize!(m)
#
#     # GET THE RESULTS
#     verbose && (@info "$(termination_status(m))")
#     x_pos = value.(a1)
#     y_pos = value.(a2)
#     z_pos = value.(a3)
#     # r_opt = [r_opt[1] r_opt[2] r_opt[3]]
#     # u_opt = value.(u)
#     # u_opt = [u_opt[1] u_opt[2] u_opt[3]]
#     # u_ts = supports.(u)
#     # u_ts = [u_ts[1][i][1] for i in 1:length(u_ts[1])]
#     u1 = value.(b1)
#     u2 = value.(b2)
#     u3 = value.(b3)
#
#     # verbose && (@info "Final Δv cost: $(get_collocation_cost(u_ts, u_opt)) km/s")
#
#     # return m, u_ts, u_opt, r_opt
#     return m, [x_pos y_pos z_pos], [u1 u2 u3]
# end

function pseudospectral_impulsive_lambert(
    x0::AbstractVector, #Starting state vector (position and velocity)
    x1::AbstractVector, #Final state vector (position or position and velocity)
    tof::AbstractFloat; #time of flight
    n::Int=-1, # Number of half revolutions between r1 and r2
    mu::AbstractFloat=3.986e5, #grav parameter for units used
    verbose::Bool=false, #whether or not to print debug statements
    poly_order::Int=20,
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

    if n < 0
        n = _get_default_half_revs(0.5 * (norm(x0[1:3]) + norm(x1[1:3])), tof, mu=mu)
    end

    # DEFINE THE PROBLEM CONSTANTS
    r0 = x0[1:3]
    rf = x1[1:3]
    v0 = x0[4:6]
    vf = x1[4:6]
    verbose && (@info "Setting up initial seed")
    vxfer, vstop = basic_lambert(r0, v0, rf, tof, n, verbose=false, v2=vf)
    Rearth = 6378.0
    minaltsq = (Rearth + 200)^2
    dv0 = vxfer - x0[4:6]
    dv1 = x1[4:6] - vstop
    dvinit = norm(dv0) + norm(dv1)
    verbose && (@info "Initial Δv cost: $dvinit km/s")
    N = poly_order
    tN, wN = _get_nodes_and_weights(N)
    D = _differentiation_matrix(tN)
    nodes = Vector{PrimerNode}()
    push!(nodes, PrimerNode(0.0, x0[1:3], x0[4:6], vxfer))
    push!(nodes, PrimerNode(tof, x1[1:3], vstop, x1[4:6]))

    converged = false
    numBurns = 2


    while !converged && numBurns < 6

        # INITIALIZE THE MODEL
        verbose && (@info "Optimizing burn $numBurns")
        m = Model(Ipopt.Optimizer)
        if !verbose
            set_silent(m)
        end

        # INITIALIZE THE VARIABLES
        numSegments = length(nodes) - 1
        totalN = numSegments * (N + 1)
        @variable(m, a1[1:totalN])
        @variable(m, a2[1:totalN])
        @variable(m, a3[1:totalN])
        @variable(m, a4[1:totalN])
        @variable(m, a5[1:totalN])
        @variable(m, a6[1:totalN])

        if constrain_u
            maxDV = 2 * dvinit
            @variable(m, -maxDV <= b1[1:numBurns] <= maxDV)
            @variable(m, -maxDV <= b2[1:numBurns] <= maxDV)
            @variable(m, -maxDV <= b3[1:numBurns] <= maxDV)
            @variable(m, 0 <= b4[1:numBurns] <= tof)
        else
            @variable(m, b1[1:numBurns])
            @variable(m, b2[1:numBurns])
            @variable(m, b3[1:numBurns])
            @variable(m, 0 <= b4[1:numBurns] <= tof)
        end

        # Constrain beginning and end

        fix(a1[1], r0[1]; force=true)
        fix(a2[1], r0[2]; force=true)
        fix(a3[1], r0[3]; force=true)

        fix(a1[end], rf[1]; force=true)
        fix(a2[end], rf[2]; force=true)
        fix(a3[end], rf[3]; force=true)

        fix(b4[1], 0.0; force=true)
        fix(b4[end], tof; force=true)
        @constraint(m, [i = 2:numBurns], b4[i] >= b4[i-1])

        # Set objective
        @objective(m, Min, sum(i -> sqrt(1e-14 + (b1[i]^2 + b2[i]^2 + b3[i]^2)), 1:numBurns))

        # Path constraint
        @constraint(m, [k = 1:totalN], a1[k]^2 + a2[k]^2 + a3[k]^2 >= minaltsq)

        # Maneuver Starting Values
        for i in 1:numBurns
            dv = nodes[i].outVelocity - nodes[i].inVelocity
            set_start_value(b1[i], dv[1])
            set_start_value(b2[i], dv[2])
            set_start_value(b3[i], dv[3])
        end

        for i in 1:numSegments
            # PROVIDE STARTING VALUES
            x0p = [nodes[i].position...; nodes[i].outVelocity...]
            tau0 = nodes[i].time
            tauf = nodes[i+1].time
            prob = ODEProblem(dstate, x0p, (tau0, tauf), 0)
            sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
            guess_traj = [(t) -> sol(t)[i] for i in 1:6]
            tauN = 0.5 .* ((tauf - tau0) .* tN .+ (tauf + tau0))

            for j in 1:N+1
                ind = (i - 1) * (N + 1) + j
                tau = tauN[j]
                set_start_value(a1[ind], guess_traj[1](tau))
                set_start_value(a2[ind], guess_traj[2](tau))
                set_start_value(a3[ind], guess_traj[3](tau))
                set_start_value(a4[ind], guess_traj[4](tau))
                set_start_value(a5[ind], guess_traj[5](tau))
                set_start_value(a6[ind], guess_traj[6](tau))
            end

            # Endpoint Constraints
            if i == 1
                @constraint(m, a4[1] - x0[4] - b1[1] == 0.0)
                @constraint(m, a5[1] - x0[5] - b2[1] == 0.0)
                @constraint(m, a6[1] - x0[6] - b3[1] == 0.0)
            else
                ind = (i - 1) * (N + 1)
                @constraint(m, a1[ind+1] - a1[ind] == 0.0)
                @constraint(m, a2[ind+1] - a2[ind] == 0.0)
                @constraint(m, a3[ind+1] - a3[ind] == 0.0)
                @constraint(m, a4[ind+1] - a4[ind] - b1[i] == 0.0)
                @constraint(m, a5[ind+1] - a5[ind] - b2[i] == 0.0)
                @constraint(m, a6[ind+1] - a6[ind] - b3[i] == 0.0)
            end
            if i == numSegments
                @constraint(m, x1[4] - a4[end] - b1[end] == 0.0)
                @constraint(m, x1[5] - a5[end] - b2[end] == 0.0)
                @constraint(m, x1[6] - a6[end] - b3[end] == 0.0)
            end

            # Set dynamic constraints
            baseInd = (i - 1) * (N + 1)
            iterlo = i == 1 ? 2 : 1
            iterhi = i == numSegments ? N : N + 1
            for k in iterlo:iterhi
                ind = baseInd + k
                @constraint(m, (b4[i+1] - b4[i]) / 2 * a4[ind] - sum(l -> D[k, l] * a1[baseInd+l], 1:N+1) == 0)
                @constraint(m, (b4[i+1] - b4[i]) / 2 * a5[ind] - sum(l -> D[k, l] * a2[baseInd+l], 1:N+1) == 0)
                @constraint(m, (b4[i+1] - b4[i]) / 2 * a6[ind] - sum(l -> D[k, l] * a3[baseInd+l], 1:N+1) == 0)
                @constraint(m, (b4[i+1] - b4[i]) / 2 * (-mu * a1[ind] / (a1[ind]^2 + a2[ind]^2 + a3[ind]^2)^(3 / 2)) - sum(l -> D[k, l] * a4[baseInd+l], 1:N+1) == 0)
                @constraint(m, (b4[i+1] - b4[i]) / 2 * (-mu * a2[ind] / (a1[ind]^2 + a2[ind]^2 + a3[ind]^2)^(3 / 2)) - sum(l -> D[k, l] * a5[baseInd+l], 1:N+1) == 0)
                @constraint(m, (b4[i+1] - b4[i]) / 2 * (-mu * a3[ind] / (a1[ind]^2 + a2[ind]^2 + a3[ind]^2)^(3 / 2)) - sum(l -> D[k, l] * a6[baseInd+l], 1:N+1) == 0)

            end
        end # SOLVE THE MODEL
        verbose && (@info "Solving...")
        optimize!(m)

        if termination_status(m) != LOCALLY_SOLVED &&
           termination_status(m) != ALMOST_LOCALLY_SOLVED
            return m
            error("Couldn't find solution.")
        end

        verbose && (@info "Looking to add a maneuver...")
        u1 = value.(b1)
        u2 = value.(b2)
        u3 = value.(b3)
        u4 = value.(b4)
        x_pos = value.(a1)
        y_pos = value.(a2)
        z_pos = value.(a3)
        vx = value.(a4)
        vy = value.(a5)
        vz = value.(a6)
        for i in 1:numBurns
            index = (i - 1) * (N + 1) + 1
            nodes[i].time = u4[i]
            if i > 1
                nodes[i].inVelocity = [vx[index-1], vy[index-1], vz[index-1]]
            end
            if i < numBurns
                nodes[i].outVelocity = [vx[index], vy[index], vz[index]]
                nodes[i].position = [x_pos[index], y_pos[index], z_pos[index]]
            else
                nodes[i].position = [x_pos[index-1], y_pos[index-1], z_pos[index-1]]
            end
        end
        tfull, pmagfull = _get_full_primer_history(nodes)
        ind = argmax(pmagfull)
        pm = pmagfull[ind]
        if pm <= 1.01
            converged = true
            continue
        end

        numBurns += 1 # find the node it's in
        tmax = tfull[ind]
        baseInd = findlast([nodes[i].time for i in 1:length(nodes)] .< tmax)

        dv0 = nodes[baseInd].outVelocity - nodes[baseInd].inVelocity
        dv1 = nodes[baseInd+1].outVelocity - nodes[baseInd+1].inVelocity
        p0 = norm(dv0) == 0 ? zeros(3) : unit(dv0)
        p1 = norm(dv1) == 0 ? zeros(3) : unit(dv1)
        tof = nodes[baseInd+1].time - nodes[baseInd].time
        x0p = [nodes[baseInd].position; nodes[baseInd].outVelocity]
        p0dot = _calculate_p0dot(p0, p1, tof, x0p)
        stateHist = _get_full_state(p0, p0dot, x0p, tof)
        tm, xm, drm, _ = _get_initial_mid_point(stateHist)

        # try here to just add the basic version and let ipopt take care of connecting

        push!(nodes, PrimerNode(nodes[baseInd].time + tm, xm[1:3] + drm, xm[4:6], xm[4:6]))
        sort!(nodes)
        _correct_nodes!(nodes)
        # r_opt = [x_pos y_pos z_pos vx_pos vy_pos vz_pos]
        # return m, [u1 u2 u3], u4, r_opt
    end


    return nodes

    # GET THE RESULTS
    verbose && (@info "$(termination_status(m))")
    x_pos = value.(a1)
    y_pos = value.(a2)
    z_pos = value.(a3)
    # r_opt = [r_opt[1] r_opt[2] r_opt[3]]
    # u_opt = value.(u)
    # u_opt = [u_opt[1] u_opt[2] u_opt[3]]
    # u_ts = supports.(u)
    # u_ts = [u_ts[1][i][1] for i in 1:length(u_ts[1])]
    u1 = value.(b1)
    u2 = value.(b2)
    u3 = value.(b3)

    # verbose && (@info "Final Δv cost: $(get_collocation_cost(u_ts, u_opt)) km/s")

    # return m, u_ts, u_opt, r_opt
    return m, [x_pos y_pos z_pos], [u1 u2 u3]
end
function _correct_nodes!(nodes::Vector{PrimerNode})
    for i in 1:length(nodes)-1
        r0 = nodes[i].position
        v0 = nodes[i].inVelocity
        rf = nodes[i+1].position
        vf = nodes[i+1].outVelocity
        tof = nodes[i+1].time - nodes[i].time
        n = _get_default_half_revs(0.5 * (norm(r0) + norm(rf)), tof)
        vxfer, vstop = basic_lambert(r0, v0, rf, tof, n, verbose=false, v2=vf)
        nodes[i].outVelocity = vxfer
        nodes[i+1].inVelocity = vstop
    end
end

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