
using OrdinaryDiffEq
using Trixi
using TrixiShallowWater

###############################################################################
# Semidiscretization of the multilayer shallow water equations to test well-balancedness

equations = ShallowWaterMultiLayerEquations1D(gravity_constant = 1.0, H0 = 0.7,
                                              rhos = (0.8, 0.9, 1.0))

"""
    initial_condition_fjordholm_well_balanced(x, t, equations::ShallowWaterMultiLayerEquations1D)

Initial condition to test well balanced with a bottom topography adapted from Fjordholm
- Ulrik Skre Fjordholm (2012)
  Energy conservative and stable schemes for the two-layer shallow water equations.
  [DOI: 10.1142/9789814417099_0039](https://doi.org/10.1142/9789814417099_0039)
"""
function initial_condition_fjordholm_well_balanced(x, t,
                                                   equations::ShallowWaterMultiLayerEquations1D)
    inicenter = 0.5
    x_norm = x[1] - inicenter
    r = abs(x_norm)

    H = [0.7, 0.6, 0.5]
    v = [0.0, 0.0, 0.0]
    b = r <= 0.1 ? 0.2 * (cos(10 * pi * (x[1] - 0.5)) + 1) : 0.0

    return prim2cons(SVector(H..., v..., b), equations)
end

initial_condition = initial_condition_fjordholm_well_balanced

###############################################################################
# Get the DG approximation space

volume_flux = (flux_ersing_etal, flux_nonconservative_ersing_etal)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_ersing_etal, flux_nonconservative_ersing_etal),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = 0.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000,
                periodicity = true)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = false,
                                     extra_analysis_integrals = (lake_at_rest_error,))

stepsize_callback = StepsizeCallback(cfl = 1.0)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
summary_callback() # print the timer summary
