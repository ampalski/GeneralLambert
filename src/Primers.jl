# This function appears to be deprecated
function _primer_propagation_errors(p0, p0dot, pf, x0, tof)
    t = (0.0, tof)
    prob = ODEProblem(dxPrimer, [x0; p0; p0dot], t, 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    newpf = sol.u[end][7:9]

    return newpf - pf

end

function _build_STM(x0p, tof)
    t = (0.0, tof)

    stm0 = Matrix{Float64}(I, 6, 6)

    combined0 = [x0p; reshape(stm0, 36, 1)]

    t = (0.0, tof)
    prob = ODEProblem(dxStm, combined0, t, 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    combinedf = sol.u[end]

    stmf = reshape(combinedf[7:end], 6, 6)

    return stmf
end

#p0 and pf are 3-length vectors, tof in seconds, x0p the full 6-length vector
function _calculate_p0dot(p0, pf, tof, x0p)
    stmf = _build_STM(x0p, tof)

    M = stmf[1:3, 1:3]
    N = stmf[1:3, 4:6]

    p0dot = N \ (pf - M * p0)
    # p0dot2 = shootMain(p0dot, x -> _primer_propagation_errors(p0, x, pf, x0p, tof)) #may not be necessary
    # @info "Difference between paper and shooting is $(p0dot2-p0dot)"
    return p0dot
end

function _get_full_state(p0, p0dot, x0p, tof)
    t = (0.0, tof)
    prob = ODEProblem(dxPrimer, [x0p; p0; p0dot], t, 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    fullSol = stack(sol.u, dims=2)

    return statePrimerHist(sol.t, fullSol[1:6, :], fullSol[7:end, :])
end

function _connect_intermediate(node0, nodef, xm, tm, tof, drm)
    n = _get_default_half_revs(0.5 * (norm(node0.position) + norm(nodef.position)), tm)
    n2 = _get_default_half_revs(0.5 * (norm(node0.position) + norm(nodef.position)), tof - tm)
    vxfer0, vstop0 = basic_lambert(node0.position, node0.inVelocity, xm[1:3] + drm, tm, n)
    vxfer1, vstop1 = basic_lambert(xm[1:3] + drm, vstop0, nodef.position, tof - tm, n2, v2=nodef.outVelocity)

    return (vxfer0, vstop0, vxfer1, vstop1)
end

# This function appears to be deprecated
function _primer_gradient(node0, nodef, xm, tm, tof, drm)
    stateHist1, stateHist2, _, _, _, _, _ =
        _get_split_histories(node0, nodef, xm, tm, tof, drm)

    pmdot_minus = stateHist1.primers[4:6, end]
    pmdot_plus = stateHist2.primers[4:6, 1]
    H_minus = pmdot_minus' * stateHist1.state[4:6, end]
    H_plus = pmdot_plus' * stateHist2.state[4:6, 1]

    dx = [H_minus - H_plus; pmdot_plus - pmdot_minus]
    return dx
end
# function _primer_gradient_descent(node0, nodef, xm, tm, drm; verbose=false)
function _primer_gradient_descent(nodes, baseInd, xm, tm, drm; verbose=false)
    node0 = nodes[baseInd]
    nodef = nodes[baseInd+1]
    tof = nodef.time - node0.time

    ktr = 0
    oldbenefit = norm(nodef.outVelocity - nodef.inVelocity) +
                 norm(node0.outVelocity - node0.inVelocity)
    newbenefit = 999.0
    tm2 = tm
    drm2 = copy(drm)
    warned = false
    while ktr < 100

        # dx = _primer_gradient(node0, nodef, xm, tm2, tof, drm2)
        # dx2 = _mid_point_line_search2(x -> _check_cost(node0, nodef, xm, tm2, tof, drm2, x, dvm), dx)

        # verbose && (@info "at ktr $ktr")
        # optSolution = optimize(x -> _check_cost(node0, nodef, xm, tm2, tof, drm2, x), zeros(4), LBFGS())
        optSolution = optimize(x -> _check_cost(nodes, baseInd, xm, tm2, tof, drm2, x), zeros(4), LBFGS(), Optim.Options(iterations=100))
        if !Optim.converged(optSolution) && !warned
            warned = true
            @warn "Sub-optimization problem failed to converge"
        end
        dx2 = Optim.minimizer(optSolution)

        tm2 = tm2 + dx2[1]
        drm2 = drm2 + dx2[2:4]

        newbenefit = Optim.minimum(optSolution)
        if (abs(newbenefit - oldbenefit) < 1e-6) || (abs(dx2[1]) < 0.001 && norm(dx2[2:4]) < 0.001)
            break
        end
        oldbenefit = newbenefit
        ktr += 1
    end
    _, _, vxfer0, vstop0, vxfer1, vstop1, dv_new =
        _get_split_histories(node0, nodef, xm, tm2, tof, drm2)
    return (tm2, drm2, vxfer0, vstop0, vxfer1, vstop1)
end

function _get_split_histories(node0, nodef, xm, tm, tof, drm)
    (vxfer0, vstop0, vxfer1, vstop1) = _connect_intermediate(node0, nodef, xm, tm, tof, drm)
    dv1 = vxfer0 - node0.inVelocity
    dv2 = vxfer1 - vstop0
    dv3 = nodef.outVelocity - vstop1
    dv_new = norm(dv1) + norm(dv2) + norm(dv3)

    p0_new = unit(dv1)
    pm_new = unit(dv2)
    pf_new = unit(dv3)
    x0p = [node0.position; vxfer0]
    p0dot_new = _calculate_p0dot(p0_new, pm_new, tm, x0p)
    stateHist1 = _get_full_state(p0_new, p0dot_new, x0p, tm)

    x0p = [xm[1:3] + drm; vxfer1]
    pmdot_new = _calculate_p0dot(pm_new, pf_new, tof - tm, x0p)
    stateHist2 = _get_full_state(pm_new, pmdot_new, x0p, tof - tm)
    return (stateHist1, stateHist2, vxfer0, vstop0, vxfer1, vstop1, dv_new)
end

function _check_cost(nodes, baseInd, xm, tm, tof, drm, dx)
    tm2 = clamp(tm + dx[1], 0, tof)
    drm2 = drm + dx[2:4]
    node0 = nodes[baseInd]
    nodef = nodes[baseInd+1]
    (vxfer0, vstop0, vxfer1, vstop1) =
        _connect_intermediate(node0, nodef, xm, tm2, tof, drm2)
    dv_new = 0.0
    for nodeind in eachindex(nodes)
        if nodeind == baseInd
            dv_new += norm(vxfer0 - nodes[nodeind].inVelocity)
            dv_new += norm(vxfer1 - vstop0)
        elseif nodeind == baseInd + 1
            dv_new += norm(nodes[nodeind].outVelocity - vstop1)
        else
            dv_new += norm(nodes[nodeind].outVelocity - nodes[nodeind].inVelocity)
        end
    end
    return dv_new
end
# function _check_cost(node0, nodef, xm, tm, tof, drm, dx)
#     tm2 = tm + dx[1]
#     drm2 = drm + dx[2:4]
#     (vxfer0, vstop0, vxfer1, vstop1) =
#         _connect_intermediate(node0, nodef, xm, tm2, tof, drm2)
#     dv1 = vxfer0 - node0.inVelocity
#     dv2 = vxfer1 - vstop0
#     dv3 = nodef.outVelocity - vstop1
#     dv_new = norm(dv1) + norm(dv2) + norm(dv3)
#     return dv_new
# end

# This function appears to be deprecated
function _mid_point_line_search(checkCost, dxInit)
    gamma = 1.0
    dx = copy(dxInit)
    E0 = checkCost(zeros(length(dx)))
    E1 = checkCost(-gamma * dx)

    if E0 > E1
        # Going in the right direction, add to gamma
        # Idea, keep a log of alpha and E values
        # Fit a polynomial
        # use the argmin value of the polynomial to find the next alpha value
        x_polynomial = [0.0, gamma]
        y_polynomial = [E0, E1]
        gamma *= 1.1
        itrs = 0
        consecutiveWOImprovement = 0
        while itrs < 100 && consecutiveWOImprovement < 5
            E2 = checkCost(-gamma * dx)
            push!(x_polynomial, gamma)
            push!(y_polynomial, E2)
            inds = sortperm(x_polynomial)
            x_polynomial = x_polynomial[inds]
            y_polynomial = y_polynomial[inds]
            p = fit(x_polynomial, y_polynomial, 2)
            range = gamma < 100000 ? (0:50*gamma) : (0:10:50*gamma)
            gamma = argmin(p, range)

            @info "New gamma value: $gamma"

            if gamma in x_polynomial
                gamma = 10 * x_polynomial[end] * rand()
            end

            if E2 < E1
                consecutiveWOImprovement = 0
                E1 = E2
            else
                consecutiveWOImprovement += 1
            end
            itrs += 1
        end
        gamma = x_polynomial[argmin(y_polynomial)]
        return (-gamma * dx)
    end
    # Overshot, come back
    gammaMult = 0.9
    while gamma > 0.0001
        gamma *= gammaMult
        E2 = checkCost(-gamma * dx)
        if E2 < E0
            # display(gamma)
            return (-gamma * dx)
        end
    end

    return zeros(length(dx))
end

function _get_initial_mid_point(stateHist)
    x0p = stateHist.state[:, 1]
    primermag = norm.(eachcol(stateHist.primers[1:3, :]))
    tof = stateHist.t[end]

    # 1) Designate a tm = argmax(p) (if p > 1)
    ind = argmax(primermag)
    tm = stateHist.t[ind]
    xm = stateHist.state[:, ind]
    pm = stateHist.primers[1:3, ind]

    # 2) Build the STM on the unperturbed path between the boundary points (either 
    # t0 and tf or ti and tj if adding fourth, fifth, etc burns) and tm
    stm_0m = _build_STM(x0p, tm)

    N_0m = stm_0m[1:3, 4:6]
    T_0m = stm_0m[4:6, 4:6]

    stm_mf = _build_STM(xm, tof - tm)

    M_mf = stm_mf[1:3, 1:3]
    N_mf = stm_mf[1:3, 4:6]

    # 3) Find \delta r_m from eq 5.56 (with the right ϵ)

    Q = -(M_mf' * inv(N_mf') + T_0m * inv(N_0m))
    dJ = 1.0
    ctr = 0
    drm = zeros(3)
    dvm = zeros(3)
    β = 0.05
    while dJ > 0 && ctr < 50
        temp = Q \ pm
        ϵ = β * norm(xm) / norm(temp)
        drm = ϵ * temp
        dvm = Q * drm
        ndvm = norm(dvm)
        dJ = ndvm * (1 - pm' * unit(dvm)) # if dJ > 0, decrease \beta and try again
        β *= 0.9
        ctr += 1
    end

    return (tm, xm, drm, dvm)
end

function _get_default_half_revs(r1::Float64, tof; mu=3.986e5)
    # Assume r1 is the semi-major axis, convert to period
    halfperiod = pi * sqrt(r1^3 / mu)
    return Int(floor(tof / halfperiod))
end

function _initial_coast_cost(dt0, x0, x1, tof, n, mu, verbose)
    usetof = tof - dt0
    r, v = universalkepler(SA[x0[1:3]...], SA[x0[4:6]...], dt0, mu)
    vxfer, vstop = basic_lambert(r, v, x1[1:3], usetof, n, mu=mu, verbose=false, v2=x1[4:6])
    dv0new = vxfer - v
    x0p = [r; vxfer]

    dv1new = x1[4:6] - vstop

    p0new = unit(dv0new)
    pfnew = unit(dv1new)

    p0dotnew = _calculate_p0dot(p0new, pfnew, usetof, x0p)
    stateHist = _get_full_state(p0new, p0dotnew, x0p, usetof)

    grad = -norm(dv0new) * p0dotnew' * p0new
    return (grad, stateHist)
end
function _initial_coast(x0, x1, tof, p0, p0dot, dv0, n, mu, verbose)
    dt0 = 1.0
    gamma = 2
    grad = -norm(dv0) * p0dot' * p0
    leftbound = (0.0, grad)
    rightbound = (0.0, 0.0)

    # Double check that this initial coast is indeed making it less expensive
    vxfer, vstop = basic_lambert(x0[1:3], x0[4:6], x1[1:3], tof, n, mu=mu, verbose=false, v2=x1[4:6])
    dv0init = vxfer - x0[4:6]
    dv1init = x1[4:6] - vstop
    initcost = norm(dv0init) + norm(dv1init)

    usetof = tof - dt0
    r, v = universalkepler(SA[x0[1:3]...], SA[x0[4:6]...], dt0, mu)
    vxfer, vstop = basic_lambert(r, v, x1[1:3], usetof, n, mu=mu, verbose=false, v2=x1[4:6])
    dv0new = vxfer - v
    dv1new = x1[4:6] - vstop
    newcost = norm(dv0new) + norm(dv1new)
    if initcost < newcost
        return 0.0
    end

    # Try to bracket the zero-value of the gradient.
    # Keep updating `left` until the value goes positive, which becomes `right`
    # then narrow in on a zero value
    zeroFound = false
    while !zeroFound
        while sign(grad) == sign(leftbound[2])

            dt0 *= gamma
            grad, _ = _initial_coast_cost(dt0, x0, x1, tof, n, mu, verbose)
            if sign(grad) != sign(leftbound[2])
                rightbound = (dt0, grad)
            else
                leftbound = (dt0, grad)
            end
        end

        # Then go into golden search or similar to get actual zero value
        # Need to make sure it doesn't converge on a singularity
        ktr = 1
        a, fa = leftbound
        baseline = 10 * abs(fa)
        b, fb = rightbound
        zeroFound = true
        c = 0.0
        while ktr < 1000 && abs(grad) > 1e-8
            c = 0.5 * (a + b)
            grad, _ = _initial_coast_cost(c, x0, x1, tof, n, mu, verbose)
            if sign(grad) == sign(fa)
                a = c
                fa = grad
            else
                b = c
                fb = grad
            end
            # if abs(grad) > 10 * baseline
            if abs(grad) > 0.1
                return c
                zeroFound = false
                leftbound = rightbound
                dt0, grad = leftbound
                break
            end
            @debug "$ktr: Grad is $(abs(grad)), $a : $b"
            ktr += 1
        end
        if zeroFound
            return c
        end
    end
end
function _final_coast_cost(dtf, x0, x1, tof, n, mu, verbose)
    usetof = tof - dtf
    r, v = universalkepler(SA[x1[1:3]...], SA[x1[4:6]...], -dtf, mu)
    vxfer, vstop = basic_lambert(x0[1:3], x0[4:6], r, usetof, n, mu=mu, verbose=false, v2=v)
    # vxfer, vstop = basic_lambert(x0[1:3], x0[4:6], r, usetof, n, mu=mu, verbose=verbose, v2=v)
    dv0new = vxfer - x0[4:6]
    x0p = [x0[1:3]; vxfer]

    dv1new = v - vstop

    p0new = unit(dv0new)
    pfnew = unit(dv1new)

    p0dotnew = _calculate_p0dot(p0new, pfnew, usetof, x0p)
    stateHist = _get_full_state(p0new, p0dotnew, x0p, usetof)

    grad = -norm(dv1new) * stateHist.primers[4:6, end]' * stateHist.primers[1:3, end]
    return (grad, stateHist)
end
function _final_coast(x0, x1, tof, n, mu, verbose)
    dtf = 1.0
    gamma = 2
    grad, _ = _final_coast_cost(0.0, x0, x1, tof, n, mu, verbose)
    if grad < 0
        error("Searching for a final coast when a final coast is not needed")
    end
    leftbound = (0.0, grad)
    rightbound = (0.0, 0.0)
    # Try to bracket the zero-value of the gradient.
    # Keep updating `left` until the value goes negative, which becomes `right`
    # then narrow in on a zero value
    zeroFound = false
    while !zeroFound
        while sign(grad) == sign(leftbound[2])

            dtf *= gamma
            grad, _ = _final_coast_cost(dtf, x0, x1, tof, n, mu, verbose)
            if sign(grad) != sign(leftbound[2])
                rightbound = (dtf, grad)
            else
                leftbound = (dtf, grad)
            end
        end

        # Then go into golden search or similar to get actual zero value
        # Need to make sure it doesn't converge on a singularity
        ktr = 1
        a, fa = leftbound
        baseline = 10 * abs(fa)
        b, fb = rightbound
        zeroFound = true
        c = 0.0
        while ktr < 1000 && abs(grad) > 1e-8
            c = 0.5 * (a + b)
            grad, _ = _final_coast_cost(c, x0, x1, tof, n, mu, verbose)
            if sign(grad) == sign(fa)
                a = c
                fa = grad
            else
                b = c
                fb = grad
            end
            if abs(grad) > 10 * baseline
                zeroFound = false
                leftbound = rightbound
                dtf, grad = leftbound
                break
            end
            @debug "$ktr: Grad is $(abs(grad)), $a : $b"
            ktr += 1
        end
        if zeroFound
            return c
        end
    end
end
"""
Outputs a vector of delta-v and times to connect the initial and final orbits
"""
function primer_lambert(
    x0::AbstractVector, #Starting state vector (position and velocity)
    x1::AbstractVector, #Final state vector (position or position and velocity)
    tof::AbstractFloat; #time of flight
    n::Int=-1, # Number of half revolutions between r1 and r2
    mu::AbstractFloat=3.986e5, #grav parameter for units used
    verbose::Bool=false, #whether or not to print debug statements
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

    nodes = Vector{PrimerNode}()

    # Get seed value from normal Lambert solve
    if verbose
        @info "Running basic Lambert solver to get seed values."
    end
    if positionOnly
        vxfer, vstop = basic_lambert(x0[1:3], x0[4:6], x1[1:3],
            tof, n, mu=mu, verbose=false)
    else
        vxfer, vstop = basic_lambert(x0[1:3], x0[4:6], x1[1:3],
            tof, n, mu=mu, verbose=false, v2=x1[4:6])
    end

    push!(nodes, PrimerNode(0.0, x0[1:3], x0[4:6], vxfer))
    push!(nodes, PrimerNode(tof, x1[1:3], vstop, x1[4:6]))

    dv0 = vxfer - x0[4:6]
    dv1 = !positionOnly ? x1[4:6] - vstop : zeros(3)
    verbose && (@info "Initial Δv cost: $(norm(dv0) + norm(dv1)) km/s")

    # Build initial primer history
    verbose && (@info "Creating initial primer vectors")
    p0 = unit(dv0)
    pf = positionOnly ? zeros(3) : unit(dv1)
    x0p = [x0[1:3]; vxfer]
    p0dot = _calculate_p0dot(p0, pf, tof, x0p)
    stateHist = _get_full_state(p0, p0dot, x0p, tof)

    # Look for initial or terminal coast periods
    if verbose
        @info "Shifting initial or final burn times"
    end

    # if primermag[2]>primermag[1] or negative slope at the end, use the 
    # gradient information in eq 5.36/37 to line search in time towards the center 
    # until the gradient reaches zero
    if p0dot' * p0 > 0
        verbose && (@info "Initial Coast search")
        dt0 = _initial_coast(x0, x1, tof, p0, p0dot, dv0, n, mu, verbose)
        _, stateHist = _initial_coast_cost(dt0, x0, x1, tof, n, mu, verbose)
        r, v = universalkepler(SA[x0[1:3]...], SA[x0[4:6]...], dt0, mu)
        push!(nodes, PrimerNode(dt0, r, v, stateHist.state[4:6, 1]))
        nodes[1].outVelocity = nodes[1].inVelocity
        nodes[2].inVelocity = stateHist.state[4:6, end]
        sort!(nodes)
        p0 = stateHist.primers[1:3, 1]
        p0dot = stateHist.primers[4:6, 1]
        verbose && (@info "Moved initial maneuver $dt0 seconds later")
    end
    # reset primer history with new burn times
    if stateHist.primers[4:6, end]' * stateHist.primers[1:3, end] < 0 && !positionOnly
        verbose && (@info "Terminal Coast search")
        indf = length(nodes)
        indi = indf - 1
        x0new = [nodes[indi].position...; nodes[indi].inVelocity...]
        x1new = [nodes[indf].position...; nodes[indf].outVelocity...]
        tof = nodes[indf].time - nodes[indi].time
        dtf = _final_coast(x0new, x1new, tof, n, mu, verbose)
        if dtf > 0
            _, stateHist = _final_coast_cost(dtf, x0new, x1new, tof, n, mu, verbose)
            r, v = universalkepler(SA[x1[1:3]...], SA[x1[4:6]...], -dtf, mu)
            push!(nodes, PrimerNode(nodes[indf].time - dtf, r, stateHist.state[4:6, end], v))
            nodes[indf].inVelocity = nodes[indf].outVelocity
            nodes[indi].outVelocity = stateHist.state[4:6, 1]
            sort!(nodes)
        end
        verbose && (@info "Moved final maneuver $dtf seconds earlier")
    end

    if verbose
        dv_new = 0.0
        for i in 1:length(nodes)
            dv_new += norm(nodes[i].outVelocity - nodes[i].inVelocity)
        end
        @info "Current solution Δv: $dv_new"
    end
    # Search segments for primermag > 1, add intermediate maneuvers as necessary
    verbose && (@info "Intermediate Maneuver Search")
    baseInd = 1
    numburns = 2
    while numburns < 6
        # Build the intermediate segment primer vector history
        @info "Currently searching for burn $(numburns+1)"
        tfull, pmagfull = _get_full_primer_history(nodes::Vector{PrimerNode})
        ind = argmax(pmagfull)
        pm = pmagfull[ind]
        if pm <= 1.01
            break
        end
        numburns += 1 # find the node it's in
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

        # plotFullPrimerHistory(nodes)
        # _primer_gradient_descent does the search
        # tm, drm, vxfer0, vstop0, vxfer1, vstop1 = _primer_gradient_descent(nodes[baseInd], nodes[baseInd+1], xm, tm, drm, verbose=verbose)
        tm, drm, vxfer0, vstop0, vxfer1, vstop1 = _primer_gradient_descent(nodes, baseInd, xm, tm, drm, verbose=verbose)

        if verbose
            dv_new = 0.0
            for i in 1:length(nodes)
                dv_new += norm(nodes[i].outVelocity - nodes[i].inVelocity)
            end
            @info "Current solution Δv: $dv_new"
        end
        # Once optimal mid-point is found, add as a node, then loop
        nodes[baseInd].outVelocity = vxfer0
        nodes[baseInd+1].inVelocity = vstop1
        push!(nodes, PrimerNode(nodes[baseInd].time + tm, xm[1:3] + drm, vstop0, vxfer1))
        sort!(nodes)
    end
    return nodes
end
