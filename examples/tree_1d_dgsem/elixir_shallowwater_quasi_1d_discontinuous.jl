using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiShallowWater

###############################################################################
# Semidiscretization of the quasi 1d shallow water equations
# See Chan et al.  https://doi.org/10.48550/arXiv.2307.12089 for details

equations = ShallowWaterEquationsQuasi1D(gravity = 9.81)

function initial_condition_discontinuity(x, t, equations::ShallowWaterEquationsQuasi1D)
    RealT = eltype(x)
    H = 2 + convert(RealT, 0.1) * exp(-25 * x[1]^2)
    v = 0

    if x[1] > 0
        b = convert(RealT, 0.1)
        a = one(RealT)
    else
        b = zero(RealT)
        a = convert(RealT, 1.1)
    end

    return prim2cons(SVector(H, v, b, a), equations)
end

initial_condition = initial_condition_discontinuity

###############################################################################
# Get the DG approximation space

volume_flux = (flux_chan_etal, flux_nonconservative_chan_etal)
surface_flux = (flux_lax_friedrichs, flux_nonconservative_chan_etal)
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = -0.5
coordinates_max = 0.5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 10_000,
                periodicity = true)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 200,
                                     save_initial_solution = true,
                                     save_final_solution = true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-8, reltol = 1.0e-8,
            ode_default_options()..., callback = callbacks);
