using PhotonPropagation
using LidarPhotonProp

using StaticArrays
using DataFrames

using Rotations
using LinearAlgebra
using Parquet
using Unitful
using TerminalLoggers
using Logging
using ArgParse
using ProgressLogging


global_logger(TerminalLogger(right_justify=120))


"""
    world_setup_strawb_lidar(;
        lidar_tilt,
        mean_cos_sca_ang,
        seed)

Create setups for CUDA & geometric photon prop
"""
function world_setup_strawb_lidar(;
    lidar_tilt,
    mean_cos_sca_ang,
    seed,
    n_photons=Int(1E11))

    # Setup for CUDA photon prop
    medium = make_cascadia_medium_properties(Float32(mean_cos_sca_ang))
    mono_spectrum = Monochromatic(450.0f0)

    pmt_z = 50u"mm"
    cylinder_end_z = pmt_z + 72.65u"mm"
    cylinder_end_position = SA_F32[0, 0, ustrip(u"m", cylinder_end_z)]
    filter_radius = 21.1u"mm" / 2

    # Entry window
    target = CircularDetector(
        cylinder_end_position,
        ustrip(Float32, u"m", filter_radius),
        UInt16(1),
    )

    laser_x = 44.5u"mm"
    laser_z = 125.4u"mm"
    laser_pos = SA_F32[ustrip(u"m", laser_x), 0, ustrip(u"m", laser_z)]
    laser_dir = SA_F32[sin(lidar_tilt), 0, cos(lidar_tilt)]
    beam_divergence = Float32(0.25E-3)

    prop_source_pencil_beam = PencilEmitter(
        laser_pos,
        laser_dir,
        beam_divergence,
        0.0f0,
        n_photons
    )
    setup = PhotonPropSetup(prop_source_pencil_beam, target, medium, mono_spectrum, seed)


    #Setup for geometric ray tracing

    # Lens properties
    #TODO: Interpolate to 450nm
    n_435 = 1.52668

    lens_radius = 30.9u"mm"
    lens_thickness = 4.7u"mm"
    lens_backplane_z = 63u"mm" + pmt_z
    lens_sphere_center_z = lens_backplane_z + lens_thickness - lens_radius
    lens_diameter = 1u"inch"

    pmt_length_x = 3u"mm"
    pmt_length_y = 1u"mm"


    lens = PlanoConvexLens(
        SA[0, 0, ustrip(u"m", lens_sphere_center_z)],
        ustrip(u"m", lens_radius),
        ustrip(u"m", lens_thickness),
        n_435,
        ustrip(u"m", lens_diameter),
        SA_F64[0, 0, 1])

    pmt = RectScreen(
        SA_F64[0, 0, ustrip(u"m", pmt_z)],
        ustrip(u"m", pmt_length_x),
        ustrip(u"m", pmt_length_y),
        SA_F64[0, 0, 1]
    )

    world = World(1.0, [lens, pmt])

    return setup, world
end




"""
    run_photon_sim(; g, tilt_angles, n_sims)

Run the photon propagation using a mean scattering angle `g` for different
laser tilts `tilt_angles`. For each angle, the simulation is repeated `n_sims` times using
different seeds.
Returns a `DataFrame` with photons that have hit the
"""
function run_photon_sim(; g, tilt_angles, n_sims)

    tilt_angles = deg2rad.(tilt_angles)
    all_photons = []

    @progress for t in tilt_angles
        @progress for i in 1:n_sims
            setup, world = world_setup_strawb_lidar(mean_cos_sca_ang=g, lidar_tilt=t, seed=Int64(i))
            photons = propagate_photons(setup)
            calc_total_weight!(photons, setup)
            photons[:, :laser_tilt] .= t

            #apply_fov!(photons, 1.43)
            rays = Ray.(photons[:, :position], photons[:, :direction], SNULL)
            traced_rays = trace_ray.(rays, Ref(world))
            rays_hit_screen = [ray.status == SABSORBED for ray in traced_rays]
            photons[:, :hit_pmt] .= rays_hit_screen
            push!(all_photons, photons)
        end
    end

    photons = reduce(vcat, all_photons)

    return photons
end

function apply_fov!(photons, fov)
    incident_angle = acos.(dot.(Ref(@SVector[0, 0, 1]), -photons[:, :direction]))

    photons[:, :incident_angle] = incident_angle
    mask = incident_angle .< deg2rad(fov)
    photons[:, :fov] = mask
    return photons
end


function run_as_script()

    s = ArgParseSettings()
    @add_arg_table s begin
        "--output"
        help = "Output filename"
        arg_type = String
        required = true
        "--n_sims"
        help = "Number of simulations"
        arg_type = Int
        required = true
        "--g"
        help = "Mean scattering angle"
        arg_type = Float32
        default = 0.99f0
        "--tilt_angles"
        help = "Laser tilt (deg)"
        arg_type = Float32
        nargs = '+'
    end


    parsed_args = parse_args(ARGS, s; as_symbols=true)
    photons = run_photon_sim(; parsed_args...)

    apply_fov!(photons, 1.43)
    write_parquet(args.output, photons[:, [:time, :laser_tilt, :abs_weight, :hit_pmt]])
end


if !isinteractive()
    run_as_script()
end

photons = run_photon_sim(g=0.99, tilt_angles=[-2.5], n_sims=1)
#photons = run_photon_sim(g=0.99, tilt_angles=(-5:1:1), n_sims=7)
