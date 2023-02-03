using Format
using CairoMakie
using Parquet
using DataFrames

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

sum(photons[:, :screen])
sum(photons[:, :screen_and_fov])
groups = groupby(photons, :laser_tilt)

length(groups)
fig = Figure()
for (i, (gname, group)) in enumerate(pairs(groups))
    row, col = divrem(i - 1, 3)
    ax = Axis(fig[row+1, col+1], title=format("Laser tilt: {:.1f} (deg)", rad2deg(gname[:laser_tilt])),
        xlabel="Angle(Lens Axis, Photon Direction)")
    hist!(group[:, :fov_angles], normalization=:density, label=gname)
end
fig

acc_rates_fov = combine(groups, [:screen_and_fov, :abs_weight] => ((fov, weight) -> sum(weight[fov]) / sum(weight)) => :acc_rate)
lines(rad2deg.(acc_rates_fov[:, :laser_tilt]), acc_rates_fov[:, :acc_rate],
    axis=(; xlabel="Laser Tilt (deg)", ylabel="Acceptance Rate (FOV)"))


acc_rates = combine(groups, [:screen_and_fov, :abs_weight] => ((fov, weight) -> sum(weight[fov]) / (5 * 1E11)) => :acc_rate)

lines(rad2deg.(acc_rates[:, :laser_tilt]), acc_rates[:, :acc_rate],
    axis=(; xlabel="Laser Tilt (deg)", ylabel="Acceptance Rate"))

fig = Figure()
time_bins = 0:1:50
for (i, (gname, group)) in enumerate(pairs(groups))
    row, col = divrem(i - 1, 3)
    ax = Axis(fig[row+1, col+1], title=format("Laser tilt: {:.1f} (deg)", rad2deg(gname[:laser_tilt])),
        xlabel="Time (ns)", yscale=log10, limits=(-5, 55, 1E-1, 1E4))
    mask = group[:, :screen_and_fov]
    if sum(mask) > 0
        hist!(group[mask, :time], weight=group[mask, :abs_weight], bins=time_bins, fillto=1E-1)
    end
end
fig
