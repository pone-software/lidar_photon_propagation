# MC Simulation of the STRAW-B LIDAR

## Installation
Check out the repository and run julia with the environment provided by the repo.
```bash
julia --project=.
```

Next, instantiate the environment to download dependencies:
```julia
using Pkg
Pkg.instantiate()
```

## Running photon propagation
To simulate the lidar, you can run the convience script `scripts/simulate_lidar.jl`.
This will run photon propagation + optics simulation for a list of lidar tilt angles. Note: This requires a working CUDA installation.
```bash
julia --project=. scripts/simulate_lidar.jl --output outfile.parquet --n_sims 10 --g 0.99 --tilt_angles -5 -4 -3 -2 -1 0
```
