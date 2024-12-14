function lineSearch(x0, J, errFn)
    gamma = 1.0
    gammaDecay = .9
    E0 = errFn(x0)

    n = length(E0)
    E1 = zeros(n)
    newx0 = zeros(n)

    while gamma > .0001
        newx0 = x0 + gamma * (J * E0)
        E1 = errFn(newx0)

        if norm(E0) > norm(E1)
            return (newx0, E1)
        end

        gamma *= gammaDecay
    end

    return (x0, E0)
end

# errFn is errs = errFn(x_current)
# for now, size(errs) must equal size(x0)
function shootMain(
    x0::AbstractVector,
    errFn::Function;
    threshold::Float64=0.0001,
    Δx = 0.0001,
    maxIters=100,
)

    newx0 = copy(x0)
    errs = errFn(x0)
    errsOld = [999999.0, 999999.0, 9999999.0]
    n = length(x0)
    itr = 0

    while norm(errs) > threshold && norm(errsOld - errs) > threshold && itr < maxIters
        # display(norm(errs))
        errsOld = copy(errs)
        itr += 1

        J = zeros(n,n)

        for i in 1:n
            pertx = copy(newx0)
            pertx[i] += Δx
            errsPrime = errFn(pertx)

            for j in 1:n
                J[j,i] = (errs[j] - errsPrime[j]) / Δx
            end
        end

        newx0, errs = lineSearch(newx0, inv(J), errFn)
    end

    return newx0
end

function lambertPropErrs(r0, v0, rf, tof)
    t = (0.0, tof)
    prob = ODEProblem(dstate, [r0; v0], t, 0)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
    newxf = sol.u[end][1:3]

    return newxf - rf

end