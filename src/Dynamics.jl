function primerA(xIn)
    mu = 398600.4418

    x = xIn[1]
    y = xIn[2]
    z = xIn[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z

    denom = (x2 + y2 + z2)^(-5 / 2)

    G = zeros(3, 3)
    G[1, 1] = mu * (2 * x2 - y2 - z2) * denom
    G[1, 2] = 3 * mu * x * y * denom
    G[1, 3] = 3 * mu * x * z * denom

    G[2, 1] = G[1, 2]
    G[2, 2] = -mu * (x2 - 2 * y2 + z2) * denom
    G[2, 3] = 3 * mu * y * z * denom

    G[3, 1] = G[1, 3]
    G[3, 2] = G[2, 3]
    G[3, 3] = -mu * (x2 + y2 - 2 * z2) * denom

    A = zeros(6, 6)
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    A[3, 6] = 1.0
    A[4:6, 1:3] = G

    return A
end

function dPrimer(x, p, t)
    A = primerA(x)
    return A * x
end

function dstate(x, p, t)
    dx = zeros(6)
    r = norm(x[1:3])
    dx[1:3] = x[4:6]
    dx[4:6] = -398600.4418 / r^3 * x[1:3]

    return dx
end

function dstate_canonical(x, p, t)
    dx = zeros(6)
    r = norm(x[1:3])
    dx[1:3] = x[4:6]
    dx[4:6] = -1.0 / r^3 * x[1:3]

    return dx
end

function dxStm(x, p, t)
    state = x[1:6]
    stm = reshape(x[7:end], 6, 6)

    dx = zeros(size(x))
    dx[1:6] = dstate(state, p, t)
    A = primerA(state)
    dstm = A * stm
    dx[7:end] = reshape(dstm, 36, 1)

    return dx
end

function dxPrimer(x, p, t)
    state = x[1:6]
    primer = x[7:end]
    A = primerA(state)
    return [dstate(state, p, t); A * primer]
end
