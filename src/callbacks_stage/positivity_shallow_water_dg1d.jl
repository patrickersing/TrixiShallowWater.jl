# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function limiter_shallow_water!(u, threshold::Real, variable,
                                mesh::Trixi.AbstractMesh{1},
                                equations::ShallowWaterEquations1D,
                                dg::DGSEM, cache)
    @unpack weights = dg.basis

    Trixi.@threaded for element in eachelement(dg, cache)
        # determine minimum value
        value_min = typemax(eltype(u))
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            value_min = min(value_min, variable(u_node, equations))
        end

        # detect if limiting is necessary
        value_min < threshold || continue

        # compute mean value
        u_mean = zero(get_node_vars(u, equations, dg, 1, element))
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            u_mean += u_node * weights[i]
        end
        # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
        u_mean = u_mean / 2^ndims(mesh)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)

            # Cut off velocity in case that the waterheight is smaller than the threshold

            h_node, h_v_node, b_node = u_node
            h_mean, h_v_mean, _ = u_mean # b_mean is not used as b_node must not be overwritten

            # Set them both to zero to apply linear combination correctly
            if h_node <= threshold
                h_v_node = zero(eltype(u))
                h_v_mean = zero(eltype(u))
            end

            u_node = SVector(h_node, h_v_node, b_node)
            u_mean = SVector(h_mean, h_v_mean, b_node)

            # When velocity is cut off, the only averaged value is the waterheight,
            # because the velocity is set to zero and this value is passed.
            # Otherwise, the velocity is averaged, as well.
            # Note that the auxiliary bottom topography variable `b` is never limited.
            set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
                           equations, dg, i, element)
        end
    end

    # "Safety" application of the wet/dry thresholds over all the DG nodes
    # on the current `element` after the limiting above in order to avoid dry nodes.
    # If the value_mean < threshold before applying limiter, there
    # could still be dry nodes afterwards due to logic of the limiting
    Trixi.@threaded for element in eachelement(dg, cache)
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)

            h, hv, b = u_node

            # Apply velocity desingularization
            hv = h * (2 * h * hv) /
                 (h^2 + max(h^2, equations.threshold_desingularization))

            if h <= threshold
                h = threshold
                hv = zero(eltype(u))
            end

            u_node = SVector(h, hv, b)

            set_node_vars!(u, u_node, equations, dg, i, element)
        end
    end

    return nothing
end

# Note that for the `ShallowWaterMultiLayerEquations1D` only the waterheight `h` is limited in
# each layer. Furthermore, a velocity desingularization is applied after the limiting to avoid
# numerical problems near dry states.
function limiter_shallow_water!(u, threshold::Real, variable,
                                mesh::Trixi.AbstractMesh{1},
                                equations::ShallowWaterMultiLayerEquations1D,
                                dg::DGSEM, cache)
    @unpack weights = dg.basis

    Trixi.@threaded for element in eachelement(dg, cache)
        # Limit layerwise
        for m in eachlayer(equations)
            # determine minimum value
            value_min = typemax(eltype(u))
            for i in eachnode(dg)
                u_node = get_node_vars(u, equations, dg, i, element)
                value_min = min(value_min, variable(u_node, equations)[m])
            end

            # detect if limiting is necessary
            value_min < threshold - eps() || continue

            # compute mean value
            u_mean = zero(get_node_vars(u, equations, dg, 1, element))
            for i in eachnode(dg)
                u_node = get_node_vars(u, equations, dg, i, element)
                u_mean += u_node * weights[i]
            end

            # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
            u_mean = u_mean / 2^ndims(mesh)

            # We compute the value directly with the mean values.
            # The waterheight `h` is limited independently in each layer.
            value_mean = variable(u_mean, equations)[m]
            theta = (value_mean - threshold) / (value_mean - value_min)

            for i in eachnode(dg)
                u_node = get_node_vars(u, equations, dg, i, element)
                h_node = waterheight(u_node, equations)[m]
                h_mean = waterheight(u_mean, equations)[m]

                u[m, i, element] = theta * h_node + (1 - theta) * h_mean
            end
        end
    end

    # "Safety" application of the wet/dry thresholds over all the DG nodes
    # on the current `element` after the limiting above in order to avoid dry nodes.
    # If the value_mean < threshold before applying limiter, there
    # could still be dry nodes afterwards due to logic of the limiting.
    # Additionally, a velocity desingularization is applied to avoid numerical 
    # problems near dry states.
    Trixi.@threaded for element in eachelement(dg, cache)
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)

            h = MVector(waterheight(u_node, equations))
            hv = MVector(momentum(u_node, equations))
            b = u_node[end]

            # Apply velocity desingularization
            hv = h .* (2 * h .* hv) ./
                 (h .^ 2 .+ max.(h .^ 2, equations.threshold_desingularization))

            for i in eachlayer(equations)
                # Ensure positivity and zero velocity at dry states
                if h[i] <= threshold
                    h[i] = threshold
                    hv[i] = zero(eltype(u))
                end
            end

            u_node = SVector(h..., hv..., b)

            set_node_vars!(u, u_node, equations, dg, i, element)
        end
    end

    return nothing
end
end # @muladd
