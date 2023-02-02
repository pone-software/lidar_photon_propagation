
module Optics

using PhysicsTools
using Rotations
using StaticArrays
using LinearAlgebra

export Ray, RayStatus, SNULL, SREFRACTED, SABSORBED
export refract_ray, advance_ray, trace_ray
export PlanoConvexLens, RectScreen
export World


@enum RayStatus SNULL SREFRACTED SABSORBED SERROR

struct Ray{T}
    position::SVector{3, T}
    direction::SVector{3, T}
    status::RayStatus

    # Ensure direction is normalized
    function Ray(pos::SVector{3, T}, dir::SVector{3, T}, status::RayStatus) where {T}
        dir = dir ./ norm(dir)
        return new{T}(pos, dir, status)
    end
end

function Ray(pos::AbstractVector, dir::AbstractVector, status::RayStatus)

    T = promote_type(eltype(pos), eltype(dir))
    return Ray(SVector{3, T}(pos), SVector{3, T}(dir), status)

end


# TODO: This is basically a copy from PropagatePhotons.jl
# Could be refactored to PhysicsTools
function line_sphere_intersection(ray::Ray{T}, sphere_center, sphere_radius) where {T}
    target_pos = sphere_center
    target_rsq = sphere_radius^2
    position = ray.position
    direction = ray.direction
    
    dpos = position .- target_pos

    a = dot(direction, dpos)
    pp_norm_sq = sum(dpos .^ 2)

    b = a^2 - (pp_norm_sq - target_rsq)

    isec = b >= 0

    if !isec
        return false, zero(T), zero(T)
    end
  
    d1 = -a - sqrt(b)
    d2 = -a + sqrt(b) 

    return true, d1, d2

end

function line_circle_intersection(ray::Ray{T}, circle_position, circle_radius) where {T}
    pos = ray.position
    dir = ray.direction

    dir_normal = dir[3]

    if dir_normal == 0
        return false, zero(T)
    end

    d = (circle_position[3] - pos[3]) / dir_normal

    if (d < 0) 
        return false, zero(T)
    end

    isec = ((pos[1] + dir[1] * d - circle_position[1])^2 + (pos[2] + dir[2] * d - circle_position[2])^2) < circle_radius^2

    if isec
        return true, d
    else
        return false, zero(T)
    end
end

function line_rectangle_intersection(ray::Ray{T}, rectangle_position, length_x, length_y) where {T}
    
    pos = ray.position
    dir = ray.direction
    dir_normal = dir[3]

    if dir_normal == 0
        return false, zero(T)
    end

    d = (rectangle_position[3] - pos[3]) / dir_normal

    isec = (abs((pos[1] + dir[1] * d) - rectangle_position[1]) < length_x/2) & (abs((pos[2] + dir[2] * d) - rectangle_position[2]) < length_y/2)

    if isec
        return true, d
    else
        return false, zero(T)
    end
end


abstract type Interface end
abstract type Lens <: Interface end
get_orientation(iface::Interface) = iface.orientation

struct PlanoConvexLens <: Lens
    sphere_center::SVector{3, Float64}
    sphere_radius::Float64
    thickness::Float64
    ref_index::Float64
    diameter::Float64
    orientation::SVector{3, Float64}
end

struct RectScreen <: Interface
    screen_center::SVector{3, Float64}
    screen_dx::Float64
    screen_dy::Float64
    orientation::SVector{3, Float64}
end

struct World
    ref_index::Float64
    interfaces::Vector{Interface}
end

get_interfaces(world::World) = world.interfaces



"""
    is_in_spherical_segment(pos, sphere_radius, segment_length)
Check whether `pos` is on a spherical segment of width `segment_length`
Assumes sphere is upright and position is on sphere.
"""
function is_on_spherical_segment(pos, sphere_center, sphere_radius, segment_length)
     # Assume local coordinates where orientation is parallel to e_z

    # Check whether pos is in spherical segment defined by lens thickness
    theta_bound = acos((sphere_radius - segment_length) / sphere_radius)

    rpos = pos .-sphere_center
    rpos_normed = (rpos)  ./ norm(rpos)

    ang = acos(dot(rpos_normed, [0, 0, 1]))

    return ang < theta_bound
end

# Check whether pos is in spherical segment defined by lens thickness
is_on_spherical_segment(pos, lens::PlanoConvexLens) = is_on_spherical_segment(pos, lens.sphere_center, lens.sphere_radius, lens.thickness)

"""
    calculate_surface_normal(::Interface, position)
Calculate the normal at `position`. Normal is always facing inwards.
"""
calculate_surface_normal(::Interface, position) = error("Not implemented")

function calculate_surface_normal(lens::PlanoConvexLens, position)
    vec = lens.sphere_center .- position
    return vec / norm(vec)
end


rotate_ray(ray::Ray, R::RotMatrix3) = Ray(R * ray.position, R * ray.direction, ray.status)
ray_position_at(ray::Ray, d) = ray.position .+ d .* ray.direction
advance_ray(ray::Ray, d) = Ray(ray_position_at(ray, d), ray.direction, ray.status)
propagate_ray_interface(::Ray, ::Interface) = error("Not implemented")



"""
    intersection(::Ray, ::Interface)
Calculate the intersection point of a ray at an Interface.
Intersection point is relative to the ray position.
"""
intersection(::Ray, ::Interface) = error("Not implemented")

"""
    intersection(ray::Ray, lens::PlanoConvexLens)

Calculate the intersection point of a ray with a plano-convex lens.
Currently only intersection with the sperical part of the lens are supported.
Intersections at any other point will return false.
Assumes that ray is givin in local coordinates, where lens is oriented parallel to e_z.
"""
function intersection(ray::Ray{T}, lens::PlanoConvexLens) where {T}

    isec, d1, d2 = line_sphere_intersection(ray, lens.sphere_center, lens.sphere_radius)
    
    # order intersection points
    d1, d2 = sort([d1, d2])

    if !isec || (d1 < 0 && d2 < 0)
        # No intersection or both points are in the rays past
        return false, zero(T)
    end

    if d1 < 0 && d2 >= 0
        # Ray starts inside lens, probably already refracted
        return false, zero(T)
    end
    
    # Select first intersection point. If second intersection point would be
    # on the spherical segment, then we enter the lens from the wrong side
    isec_pos = ray_position_at(ray, d1)

    if !is_on_spherical_segment(isec_pos, lens)
        return false, zero(T)
    end

    # We've passed all checks
    return true, d1
end


"""
    intersection(ray::Ray, screen::RectScreen)

Calculate the intersection point of a ray with a rectangular screen.
Assumes that ray is givin in local coordinates, where screen is oriented parallel to e_z.
"""
function intersection(ray::Ray{T}, screen::RectScreen) where {T}
    return line_rectangle_intersection(ray, screen.screen_center, screen.screen_dx, screen.screen_dy)
end


"""
    snells_law(direction, surface_normal, n_1, n_2)
Vector version of Snells Law. Calculates refracted direction at boundary
between medium with refractive index `n_1` and `n_2`` at a surface with surface normal `surface_normal`.
See https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
"""
function snells_law(direction, surface_normal, n_1, n_2)
    mu = n_1 / n_2
    
    sqrt_temp = 1 - mu^2 * (1 - (dot(surface_normal, direction))^2)
    @assert sqrt_temp >=0
  
    return sqrt(sqrt_temp) .* surface_normal .+ mu * (
            direction .- dot(surface_normal, direction) .* surface_normal)
end


snells_law(ray::Ray, surface_normal, n_1, n_2) = Ray(ray.position, snells_law(ray.direction, surface_normal, n_1, n_2), SREFRACTED)

"""
    propagate_ray_interface(ray::Ray, lens::PlanoConvexLens)

Refract ray at a plano-convex lens.
Assumes that ray is already propagated to the interface.
Currently only rays entering the spherical part of the lens are supported.
"""
function propagate_ray_interface(ray::Ray, lens::PlanoConvexLens, n_ext)
    
    normal_vector = calculate_surface_normal(lens, ray.position)
    backplane_position = lens.sphere_center .+ (lens.sphere_radius - lens.thickness) .* SA[0., 0., 1]
    
    # refract ray at spherical surface
    refracted_ray = snells_law(ray, normal_vector, n_ext, lens.ref_index)

    # calculate intersection with backplane
    isec, backplane_d = line_circle_intersection(refracted_ray, backplane_position, lens.diameter / 2)

    if !isec
        # no intersection with backplane
        println("No intersection with backplane")
        return Ray(ray.position, ray.direction, SERROR)
    end


    # Advance ray to backplane
    refracted_ray_bp = advance_ray(refracted_ray, backplane_d)

    # refract ray at backplane
    refracted_ray_out = snells_law(refracted_ray_bp, SA[0., 0., -1], lens.ref_index, n_ext)
    return refracted_ray_out
end


"""
    propagate_ray_interface(ray::Ray, screen::RectScreen)

Absorb a ray at a screen.
Assumes that ray is already propagated to the interface.
"""
function propagate_ray_interface(ray::Ray, ::RectScreen, ::Number)
    return Ray(ray.position, ray.direction, SABSORBED)
end


function trace_ray(ray::Ray, world::World)

    if ray.status == SABSORBED || ray.status == SERROR
        return ray
    end

    intersections = Dict()
    interfaces = get_interfaces(world)
    for iface in interfaces
        orientation = get_orientation(iface)
        # Rotate ray into local coordinates where lens is oriented parallel to e_z
        R = RotMatrix3(calc_rot_matrix(orientation, [0, 0, 1.]))
        ray_r = rotate_ray(ray, R)
        
        isec, ray_d = intersection(ray_r, iface)
        @show iface, isec
        if isec
            intersections[ray_d] = (iface, R)
        end    
    end

    if length(intersections) == 0    
        return ray
    end

    skeys = sort(collect(keys(intersections)))
    
    ray_d = first(skeys)
    interface, R = intersections[ray_d]
    ray = rotate_ray(advance_ray(ray, ray_d), R)
    refracted = propagate_ray_interface(ray, interface, world.ref_index)
    out_ray = rotate_ray(refracted, inv(R))

    # Recurse until no intersections left
    return trace_ray(out_ray, world)

end

end # module