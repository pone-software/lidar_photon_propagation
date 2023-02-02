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
using Format
using CairoMakie

global_logger(TerminalLogger(right_justify=120))

"""
    run_photon_sim(; g, tilt_angles, n_sims)

Run the photon propagation using a mean scattering angle `g` for different
laser tilts `tilt_angles`. For each angle, the simulation is repeated `n_sims` times using 
different seeds.
Returns a `DataFrame` with photons that have hit the
"""



function run_photon_sim(; g, tilt_angles, n_sims)
    medium = make_cascadia_medium_properties(Float32(g))
    mono_spectrum = Monochromatic(450.0f0)

    pmt_z = 50u"mm"
    cylinder_end_z = pmt_z + 72.65u"mm"
    cylinder_end_position = SA_F32[0, 0, ustrip(u"m", cylinder_end_z)]

    target = CircularDetector(
        cylinder_end_position,
        ustrip(Float32, u"m", 1u"inch"),
        UInt16(1),
    )

    laser_pos = SA_F32[44.5E-3, 0, 125.4E-3]

    tilt_angles = deg2rad.(tilt_angles)
    all_photons = []
    @progress for t in tilt_angles

        # Tilt in -x-direction
        beam_dir = SA_F32[sin(t), 0, cos(t)]

        beam_divergence = Float32(0.25E-3)

        # Emmit from slightly outside the module
        prop_source_pencil_beam = PencilEmitter(
            laser_pos,
            beam_dir,
            beam_divergence,
            0.0f0,
            Int64(1E11)
        )

        @progress for i in 1:n_sims
            setup = PhotonPropSetup(prop_source_pencil_beam, target, medium, mono_spectrum, Int64(i))
            photons = propagate_photons(setup)
            calc_total_weight!(photons, setup)
            photons[:, :laser_tilt] .= t
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
    write_parquet(args.output, photons[:, [:time, :laser_tilt, :abs_weight, :incident_angle]])
end


if !isinteractive()
    run_as_script()
end

#photons = run_photon_sim(g=0.99, tilt_angles=[-2.5], n_sims=1)
photons = run_photon_sim(g=0.99, tilt_angles=(-7.5:2.5:5), n_sims=5)
apply_fov!(photons, 1.43)
rays = Ray.(photons[:, :position], photons[:, :direction], SNULL)


# Lens properties
n_435 = 1.52668
lens_radius = 30.9E-3
lens_thickness = 4.7E-3
lens_backplane_z = 63E-3

lens = PlanoConvexLens(
    SA[0, 0, lens_backplane_z+lens_thickness-lens_radius],
    lens_radius,
    lens_thickness,
    n_435,
    0.0254,
    SA_F64[0, 0, 1])

pmt = RectScreen(
    SA_F64[0, 0, 0],
    3E-3,
    1E-3,
    SA_F64[0, 0, 1]
)

world = World(1., [lens, pmt])

traced_rays = trace_ray.(rays, Ref(world))
rays_screen = [ray.status == SABSORBED for ray in traced_rays]

fig = Figure()
ax = Axis3(fig[1, 1])

for (r1, r2) in zip(traced_rays[rays_screen], traced_rays[rays_screen])

    r1_bt = r1.position .- 0.1 * r1.direction

    r2_ft = r2.position .+ 0.1 * r2.direction

    #@show hcat(r1_bt, r1.position, r2.position)

    lines!(ax, hcat(r1_bt, r1.position, r2.position))
end
fig

photons[:, :screen] .= rays_screen
photons[:, :screen_and_fov] .= photons[:, :screen] .&& photons[:, :fov]  

groups = groupby(photons, :laser_tilt)

length(groups)
fig = Figure()
for (i, (gname, group)) in enumerate(pairs(groups))
    row, col = divrem(i-1, 3)
    ax = Axis(fig[row+1, col+1], title=format("Laser tilt: {:.1f} (deg)", rad2deg(gname[:laser_tilt])),
    xlabel="Angle(Lens Axis, Photon Direction)")
    hist!(group[:, :fov_angles], normalization=:density, label=gname)
end
fig

acc_rates_fov = combine(groups, [:screen_and_fov, :abs_weight] => ((fov, weight) -> sum(weight[fov]) / sum(weight)) => :acc_rate)
lines(rad2deg.(acc_rates[:, :laser_tilt]), acc_rates[:, :acc_rate],
    axis=(; xlabel="Laser Tilt (deg)", ylabel="Acceptance Rate (FOV)"))


acc_rates = combine(groups, [:screen_and_fov, :abs_weight] => ((fov, weight) -> sum(weight[fov]) / (5*1E11)) => :acc_rate)

lines(rad2deg.(acc_rates[:, :laser_tilt]), acc_rates[:, :acc_rate],
    axis=(; xlabel="Laser Tilt (deg)", ylabel="Acceptance Rate"))

fig = Figure()
time_bins = 0:1:50
for (i, (gname, group)) in enumerate(pairs(groups))
    row, col = divrem(i-1, 3)
    ax = Axis(fig[row+1, col+1], title=format("Laser tilt: {:.1f} (deg)", rad2deg(gname[:laser_tilt])),
    xlabel="Time (ns)", yscale=log10, limits=(-5, 55, 1E-1, 1E4))
    mask = group[:, :fov]
    if sum(mask) > 0
        hist!(group[mask, :time], weight=group[mask, :abs_weight], bins=time_bins, fillto=1E-1)
    end
end
fig
