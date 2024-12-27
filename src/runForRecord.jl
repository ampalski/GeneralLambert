##########################
# 1: 15 deg GEO transfer
x0 = [-30726.027555, 28902.72485, -3.75426, -2.10596, -2.23904, -0.001]
xf0 = [-37160.167933, 19965.265616, -7.178301, -1.454693, -2.707808, -0.000895]
tof = 60.0 * 60.0 * 52.0
# Currently errors on primer_lambert. Need to deep dive that.
# Couple issues popped up, primarily on the initial and terminal coast, where 
# things can hit singularities. Also changed intermediate burn to always look for 
# maximum primer magnitude, and cut off after 6 maneuvers. Finally made the 
# intermediate burn look at the full dv cost, not just the single section
# primer_lambert done for this
# collocation isn't converging, also need to check on that. might be related to above
# also need to look at number of collocation points, etc.
# Had to constrain the control input, the solver was given too much freedom to
# make wild choices. Getting a weird result where the more supports that are given,
# the more dv it takes. This is true for quadgk and simple rectangular quadrature

# 2: 60 deg GEO Transfer
x0 = [-30726.027555, 28902.72485, -3.75426, -2.10596, -2.23904, -0.001]
xf0 = [-40390.96505, -12156.950122, -13.761129, 0.886233, -2.94353082, -0.00026304]
tof = 60.0 * 60.0 * 76.0
# Can't find an initial solution, need to iterate on tof probably, maybe n
# n was wrong, fixed that.
# Primer works now
# Hanging on collocation, similar to GTO transfer
# Looks like pseudospectral continuous is trying to add an extra orbit

#3: 60* LEO - > GEO case
x0 = [-4927.10914, 6032.53541, -1828.1225, -1.9319808, -3.36505695, -5.89586078]
xf0 = [-30726.027555, 28902.72485, -3.75426, -2.10596, -2.23904, -0.001]
tof = 60.0 * 60.0 * 6 # nominal
# tof = 60.0 * 60.0 * 12 # forces initial and terminal coasts
# Both primer and collocation worked, but only with unconstrained control inputs

# 4: GTO Transfer
xf0 = [-30726.027555, 28902.72485, -3.75426, -2.10596, -2.23904, -0.001]
x0 = [-15379.05905, 35948.64615, 2997.88073, -1.877714, -0.195777, -0.90602355]
tof = 60.0 * 60.0 * 21.5
# Primer hangs in the intermediate maneuver search.
# After limiting it to 100 iterations per search, this goes away. Ends up 
# finding a bunch of practically 0 maneuvers
# collocation also failed to converge -- works with constrained u and N=100

r, v = universalkepler(SA[xf0[1:3]...], SA[xf0[4:6]...], tof, 3.986e5)
xf = [r; v]
x1 = copy(xf)
mu = 3.986e5
n = _get_default_half_revs(0.5 * (norm(x0[1:3]) + norm(x1[1:3])), tof, mu=mu)
verbose = true
vxfer, vstop = basic_lambert(x0[1:3], x0[4:6], x1[1:3], tof, n, mu=mu, verbose=true, v2=x1[4:6])
N = 20 #default 20
tN, wN = _get_nodes_and_weights(N)

# Get the dv numbers from this block
nodes = primer_lambert(x0, x1, tof, verbose=true)
# model, controlTimes, control, states = collocation_lambert(x0, x1, tof, num_supports=100, verbose=true)
model, controlTimes, control, states, velocities = collocation_lambert(x0, x1, tof, num_supports=100, verbose=true, constrain_u=true, mu=1.0)
model2, states2, control2 = pseudospectral_continuous_lambert(x0, x1, tof, poly_order=N, verbose=true, constrain_u=true, mu=1.0)
# model3, states3, control3 = pseudospectral_continuous_lambert_canonical(x0, x1, tof, poly_order=N, verbose=true, constrain_u=true)
dvtotal2 = 0.0
for i in 1:N+1
    dvtotal2 += wN[i] * (norm(control2[i, :]))
end
dvtotal2 *= tof / 2
# dvtotal3 = 0.0
# for i in 1:N+1
#     dvtotal3 += wN[i] * (norm(control3[i, :]))
# end
# dvtotal3 *= tof / 2
# model2, states2, control2 = pseudospectral_impulsive_lambert(x0, x1, tof, poly_order=100, verbose=true, constrain_u=true)
model2, nodes2 = pseudospectral_impulsive_lambert(x0, x1, tof, poly_order=N, verbose=true, constrain_u=true, mu=1.0)

# Then get time numbers from this block, after having been precompiled
@time basic_lambert(x0[1:3], x0[4:6], x1[1:3], tof, n, mu=mu, verbose=false, v2=x1[4:6])
@time nodes = primer_lambert(x0, x1, tof, verbose=false)
# @time model, controlTimes, control, states = collocation_lambert(x0, x1, tof, verbose=false)
@time model, controlTimes, control, states = collocation_lambert(x0, x1, tof, num_supports=100, verbose=false, constrain_u=true, mu=1.0)
@time model2, states2, control2 = pseudospectral_continuous_lambert(x0, x1, tof, poly_order=N, verbose=false, constrain_u=true, mu=1.0)
