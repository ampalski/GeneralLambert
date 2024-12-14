##########################
# 15 deg GEO transfer
# x0 = [-30726.027555, 28902.72485, -3.75426, -2.10596, -2.23904, -0.001]
# xf0 = [-37160.167933, 19965.265616, -7.178301, -1.454693, -2.707808, -0.000895]
# tof = 60.0 * 60.0 * 52.0

#60* LEO - > GEO case
x0 = [-4927.10914, 6032.53541, -1828.1225, -1.9319808, -3.36505695, -5.89586078]
xf0 = [-30726.027555, 28902.72485, -3.75426, -2.10596, -2.23904, -0.001]
tof = 60.0 * 60.0 * 6 # nominal
# tof = 60.0 * 60.0 * 12 # forces initial and terminal coasts

# 60 deg GEO Transfer
# x0 = [-30726.027555, 28902.72485, -3.75426, -2.10596, -2.23904, -0.001]
# xf0 = [-40390.96505, -12156.950122, -13.761129, 0.886233, -2.94353082, -.00026304]
# tof = 60.0 * 60.0 * 76.0

# GTO Transfer
# x0 = [-30726.027555, 28902.72485, -3.75426, -2.10596, -2.23904, -0.001]
# xf0 = [-15379.05905, 35948.64615, 2997.88073, -1.877714, -.195777, -.90602355]
# tof = 60.0 * 60.0 * 21.5

r, v = universalkepler(SA[xf0[1:3]...], SA[xf0[4:6]...], tof, 3.986e5)
xf = [r; v]
x1 = copy(xf)
n = 1
mu = 3.986e5
verbose = true

# Get seed Lambert value

vxfer, vstop = basic_lambert(x0[1:3], x0[4:6], xf[1:3], tof, 1, verbose=true, v2=xf[4:6])

dv1 = vxfer - x0[4:6]
x0p = [x0[1:3]; vxfer]

dv2 = xf[4:6] - vstop

p0 = unit(dv1)
# pf = zeros(3)
pf = unit(dv2)

# Build STM 

p0dot = _calculate_p0dot(p0, pf, tof, x0p)

#primers = getFullPrimers(p0, p0dot, x0p, tof)
stateHist = _get_full_state(p0, p0dot, x0p, tof)
primermag = norm.(eachcol(stateHist.primers[1:3, :]))

set_theme!(theme_black())
fig = Figure(size=(1600, 800))

ax = Axis(fig[1, 1])
lines!(ax, stateHist.t, primermag)
fig

nodes = primer_lambert(x0, x1, tof, verbose=true)
model, controlTimes, control, states = collocation_lambert(x0, x1, tof, verbose=true)

# See what shooting method for primer vector rate p0dot looks like.
# Can i start with a zeros p0dot and still land at right answer? If so, 
# don't need stm

p0dot2 = shootMain(p0dot, x -> primerPropErrs(p0, x, pf, x0p, tof))
t = (0.0, tof)
prob = ODEProblem(dxPrimer, [x0p; p0; p0dot2], t, 0)
sol2 = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
fullsol2 = stack(sol2.u, dims=2)

primers2 = fullsol2[7:9, :]
primermag2 = norm.(eachcol(primers2))

# set_theme!(theme_black())
# fig = Figure(size=(1600, 800))

# ax = Axis(fig[1, 1])
# lines!(ax, stateHist.t, primermag)
# lines!(ax, sol2.t, primermag2)
# display(GLMakie.Screen(), fig)

# 1) Designate a tm = argmax(p) (if p > 1)
ind = argmax(primermag)
tm = stateHist.t[ind]
xm = stateHist.state[:, ind]
pm = stateHist.primers[1:3, ind]

# 2) Build the STM on the unperturbed path between the boundary points (either 
# t0 and tf or ti and tj if adding fourth, fifth, etc burns) and tm
stm_0m = buildSTM(x0p, tm)

N_0m = stm_0m[1:3, 4:6]
T_0m = stm_0m[4:6, 4:6]

stm_mf = buildSTM(xm, tof - tm)

M_mf = stm_mf[1:3, 1:3]
N_mf = stm_mf[1:3, 4:6]
T_mf = stm_mf[4:6, 4:6]


# 3) Find \delta r_m from eq 5.56 (with the right \epsilon)
J = [zeros(3, 3) I(3); -I(3) zeros(3, 3)]
stm_fm = -J * stm_mf' * J
M_fm = stm_fm[1:3, 1:3]
T_fm = stm_fm[4:6, 4:6]
N_fm = stm_fm[1:3, 4:6]

Q = -(M_mf' * inv(N_mf') + T_0m * inv(N_0m))
β = 0.05
temp = Q \ pm
ϵ = β * norm(xm) / norm(temp)
drm = ϵ * temp
dvm = Q * drm
ndvm = norm(dvm)
dJ = ndvm * (1 - pm' * unit(dvm)) # if dJ > 0, decrease \beta and try again

# 4) Find the velocity from r0 to rm + \delta rm (paper suggest lambert solver, otherwise 5.51 + shooting to make sure it matches)


# dvxfer = N_0m \ drm #See if this meets up with xm
# new_vxfer = vxfer + dvxfer
# new_vxfer2 = shootMain(new_vxfer, x -> lambertPropErrs(x0[1:3], x, xm[1:3] + drm, tm))

# x0p_new = [x0[1:3]; new_vxfer2]

# t = (0.0, tm)
# prob = ODEProblem(dstate, x0p_new, t, zeros(3))
# sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
# test_xm = sol.u[end]
# display(test_xm[1:3] - (xm[1:3] + drm))
# test_vm_minus = test_xm[4:6]

# # 5) Find the velocity at rf from rm + \delta rm (paper suggests lambert solver, otherwise 5.52 + backwards shooting to make sure it matches)

# test_vm_plus = test_xm[4:6] + dvm #just use this as the seed to shoot
# test_vm_plus = shootMain(test_vm_plus, x -> lambertPropErrs(test_xm[1:3], x, xf[1:3], tof-tm))
# t = (tm, tof)
# prob = ODEProblem(dstate, [test_xm[1:3]; test_vm_plus], t, zeros(3))
# sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
# test_xf = sol.u[end]
# display(test_xf[1:3] - (xf[1:3]))


# # 6) Make sure the new total delta-v is smaller than the original

# dv1_new = new_vxfer2 - x0[4:6]
# dv2_new = test_vm_plus - test_vm_minus

checkInds = findall(primermag .> 1)
costs = zeros(length(primermag))

for checkInd in checkInds
    tm2 = stateHist.t[checkInd]
    xm2 = stateHist.state[:, checkInd]
    pm2 = stateHist.primers[1:3, checkInd]
    costs[checkInd] = checkCost(x0, new_vxfer, xm2, xf, tm2, tof, drm, zeros(4))
    # allow each one to go through a round or two of line search first
end

function checkCost(x0, new_vxfer, xm, xf, tm, tof, drm, dx)
    tm2 = tm + dx[1]
    drm2 = drm + dx[2:4]
    (dv1_new, dv2_new, x0p_new, test_xm, test_vm_plus) = connectIntermediate(x0, new_vxfer, xm, xf, tm2, tof, drm2)
    return norm(dv1_new) + norm(dv2_new)
end

deltasPSO = ParticleSwarmOptimizer([[-2000.0, 2000.0], [-5000.0, 5000.0], [-5000.0, 5000.0], [-5000.0, 5000.0]],
    x -> checkCost(x0, new_vxfer, xm, xf, tm, tof, drm, x), verbose=true)
deltasPSO2 = ParticleSwarmOptimizer([[-2000.0, 2000.0], [-5000.0, 5000.0], [-5000.0, 5000.0], [-5000.0, 5000.0]],
    x -> checkCost(x0, new_vxfer, xm, xf, tm, tof, drm, x), verbose=true, maxIter=5)

function midPointLineSearch(checkCost, dxInit)
    gamma = 1.0
    dx = copy(dxInit)
    E0 = checkCost(zeros(length(dx)))
    E1 = checkCost(-gamma * dx)

    if E0 > E1
        # Going in the right direction, add to gamma
        gammaMult = 1.1
        while gamma < 10000000
            gamma *= gammaMult
            E2 = checkCost(-gamma * dx)
            if E2 > E1
                gamma /= gammaMult
                # display(gamma)
                return (-gamma * dx)
            end
            E1 = E2
        end

    else
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

    end
    return zeros(length(dx))
end

ctr = 0
new_vxfer = copy(vxfer)
# while ctr < 100
(dv1_new, dv2_new, x0p_new, test_xm, test_vm_plus) = connectIntermediate(x0, new_vxfer, xm, xf, tm, tof, drm, dvm)
new_vxfer = x0p_new[4:6]
dv_new = norm(dv1_new) + norm(dv2_new)

display("Change in DV, should be negative")
display(dv_new - norm(dv1))

# 7) Then eqns 5.72 and 5.73 to iterate on tm and \delta rm

# write the function:
# function getPrimerHistory(x0p, p0, pf, tof)
# that gives the full state history, should just be a call of 
# p0dot = calcp0dot(p0,pf,tof, x0p)
#stateHist = getFullState(p0, p0dot, x0p, tof)

p0_new = unit(dv1_new)
pm_new = unit(dv2_new)
p0dot_new = calcp0dot(p0_new, pm_new, tm, x0p_new)
stateHist1 = getFullState(p0_new, p0dot_new, x0p_new, tm)

pmdot_new = calcp0dot(pm_new, pf, tof - tm, [test_xm[1:3]; test_vm_plus])
stateHist2 = getFullState(pm_new, pmdot_new, [test_xm[1:3]; test_vm_plus], tof - tm)

primermag1 = norm.(eachcol(stateHist1.primers[1:3, :]))
primermag2 = norm.(eachcol(stateHist2.primers[1:3, :]))
display(maximum(primermag2))


#check for convergence
pmdot_minus = stateHist1.primers[4:6, end]
pmdot_plus = stateHist2.primers[4:6, 1]
H_minus = pmdot_minus' * stateHist1.state[4:6, end]
H_plus = pmdot_plus' * stateHist2.state[4:6, 1]

dx = [H_minus - H_plus; pmdot_plus - pmdot_minus]
dx2 = midPointLineSearch(x -> checkCost(x0, new_vxfer, xm, xf, tm, tof, drm, x, dvm), dx)
dx3 = midPointLineSearch2(x -> checkCost(x0, new_vxfer, xm, xf, tm, tof, drm, x, dvm), dx)
tm = tm + dx2[1]
drm = drm + dx2[2:4]
if abs(dx2[1]) < 1e-15 && norm(dx2[2:4]) < 1e-15
    break
end
ctr += 1
# end

gamma = 1.0
dx4 = copy(dx)
E0 = checkCost(x0, new_vxfer, xm, xf, tm, tof, drm, zeros(length(dx4)), dvm)
E1 = checkCost(x0, new_vxfer, xm, xf, tm, tof, drm, -gamma * dx4, dvm)

x_polynomial = [0.0, gamma]
y_polynomial = [E0, E1]


# Going in the right direction, add to gamma
# Idea, keep a log of alpha and E values
# Fit a polynomial
# use the argmin value of the polynomial to find the next alpha value
gamma = gamma * 1.1
itrs = 0
consWOImprovement = 0
Emin = E1
while itrs < 100 && consWOImprovement < 3
    E2 = checkCost(x0, new_vxfer, xm, xf, tm, tof, drm, -gamma * dx4, dvm)
    push!(x_polynomial, gamma)
    push!(y_polynomial, E2)
    inds = sortperm(x_polynomial)
    x_polynomial = x_polynomial[inds]
    y_polynomial = y_polynomial[inds]
    p = fit(x_polynomial, y_polynomial, 2)
    gamma = argmin(p, 0:10*gamma)

    if E2 < E1
        consWOImprovement = 0
        E1 = E2
    else
        consWOImprovement += 1
    end
    itrs += 1
end

if E2 > E1
    gamma /= gammaMult
    # display(gamma)
    return (-gamma * dx)
end
E1 = E2
# end

xplot = collect(0:maximum(x_polynomial))
yplot = p.(xplot)
set_theme!(theme_black())
fig = Figure(size=(1600, 800))

ax = Axis(fig[1, 1])
n = length(primermag)
lines!(ax, xplot, yplot)
scatter!(ax, x_polynomial, y_polynomial)
display(GLMakie.Screen(), fig)




set_theme!(theme_black())
fig = Figure(size=(1600, 800))

ax = Axis(fig[1, 1])
n = length(primermag)
lines!(ax, stateHist1.t, primermag1)
lines!(ax, stateHist2.t .+ tm, primermag2)
fig

# gradient descent is new_x = x - alpha * gradient(f)
# where x is [tm;rm]
# derivs are provided in 5.72/73
# probably need to fwd/backwards produce new p/pdot values for each segment (5.9)
#    using the STM_0m and stm_mf components
# need to find pdot and H immediately before and after tm
# then go back to step 4 with the new values, and iterate until the gradient is 0
