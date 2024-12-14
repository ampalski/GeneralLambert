x0 = [-30726.027555, 28902.72485, -3.75426, -2.10596, -2.23904, -0.001]
xf0 = [-37160.167933, 19965.265616, -7.178301, -1.454693, -2.707808, -0.000895]
tof = 60.0 * 60.0 * 26.0

r, v = UnivKepler(xf0[1:3], xf0[4:6], tof, 3.986e5)
xf = [r;v]

# Get seed Lambert value
function getOptimalLambert(x0, xf, tof)
    vxfer, vstop = Lambert(x0[1:3], x0[4:6], xf[1:3], tof, 2, verbose=true)

    dv1 = vxfer - x0[4:6]
    x0p = [x0[1:3]; vxfer]

    p0 = unit(dv1)
    pf = zeros(3)

    # Build initial primer history

    p0dot = calcp0dot(p0,pf,tof, x0p)

    stateHist = getFullState(p0, p0dot, x0p, tof)

    # primermag = norm.(eachcol(stateHist.primers[1:3,:]))
    # set_theme!(theme_black())
    # fig = Figure(size=(1600, 800))

    # ax = Axis(fig[1, 1])
    # lines!(ax, stateHist.t, primermag)
    # display(GLMakie.Screen(), fig)

    # Find initial candidate intermediate point

    tm, xm, drm, dvm = getInitialMidPoint(stateHist)

    # Gradient descent to find the solution

    # deltasPSO = ParticleSwarmOptimizer([[-2000.0, 2000.0], [-5000.0, 5000.0], [-5000.0, 5000.0], [-5000.0, 5000.0]], 
    #     x -> checkCost(x0, new_vxfer, xm, xf, tm, tof, drm, x, dvm), verbose=true)
    # deltasPSO = ParticleSwarmOptimizer([[1200.0, 1600.0], [-5000.0, 5000.0], [-5000.0, 5000.0], [-5000.0, 5000.0]], 
    #     x -> checkCost(x0, new_vxfer, xm, xf, tm, tof, drm, x, dvm), verbose=true)


    new_vxfer, tm2, drm2 = primerIntermediateGradDescent(x0, vxfer, xm, xf, tm, tof, drm, dvm)

    return (new_vxfer, xm, tm2, drm2, dvm)
end
new_vxfer, xm, tm2, drm2, dvm = getOptimalLambert(x0, xf, tof)

stateHist1, stateHist2, new_vxfer, dv_new = 
    getSplitHistories(x0, new_vxfer, xm, xf, tm2, tof, drm2, dvm)

primermag1 = norm.(eachcol(stateHist1.primers[1:3,:]))
primermag2 = norm.(eachcol(stateHist2.primers[1:3,:]))

set_theme!(theme_black())
fig = Figure(size=(1600, 800))

ax = Axis(fig[1, 1])
lines!(ax, stateHist1.t, primermag1)
lines!(ax, stateHist2.t .+ tm2, primermag2)
display(GLMakie.Screen(), fig)