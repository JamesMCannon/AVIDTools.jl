"""
    AVIDTools

Convenience and utility functions for LongwaveModePropagator.jl interacting with the AVID network of VLF receivers.
"""
module LMPTools

using PyCall, Conda
using Rotations, StaticArrays
using HDF5, Dates
using GeographicLib, SatelliteToolboxGeomagneticField
using LongwaveModePropagator
using VLF
using ArchGDAL, Statistics

project_path(parts...) = normpath(@__DIR__, "..", parts...)

const GROUND_DATA = h5read(project_path("data", "conductivity_data.h5"), "/")

const TRANSMITTER = Dict(
    :NAA => Transmitter{VerticalDipole}("NAA", 44.646394, -67.281069, VerticalDipole(), Frequency(24e3), 1e6),
    :NAU => Transmitter{VerticalDipole}("NAU", 18.39875, -67.177433, VerticalDipole(), Frequency(40.75e3), 1e5),
    :NML => Transmitter{VerticalDipole}("NML", 46.366014, -98.335544, VerticalDipole(), Frequency(25.2e3), 233000),
    :NLK => Transmitter{VerticalDipole}("NLK", 48.203611, -121.917056, VerticalDipole(), Frequency(24.8e3), 142.77e3), #ARL measurement comparisons with LMP suggest 142.77. Uncertaintity from 2 std of ARL measurments is 2.8% or 3.49e3 W. FWM dictionary of transmitter powers suggests 250e3. Quick look at CAL data supports this power - deeper dive needed both here and NPM
    :NPM => Transmitter{VerticalDipole}("NPM", 21.420, -158.151, VerticalDipole(), Frequency(21.4e3), 424000)
)

const RECEIVER = Dict(#make consistent precision in lat/long
    :ARL => Receiver("ARL", 48.16630, -122.11487, 0.0, VerticalDipole()),
    :CAL => Receiver("CAL", 51.653775, -128.13274, 0.0, VerticalDipole()),
    :PRU => Receiver("PRU", 54.31045, -130.32571, 0.0, VerticalDipole()),
    :JU => Receiver("JU", 58.59062, -134.904145833, 0.0, VerticalDipole()),
    :WHI => Receiver("WHI", 60.75127, -135.10105, 0.0, VerticalDipole()),
    :FSI => Receiver("FSI", 61.756622, -121.22904, 0.0, VerticalDipole()),
    :FSM => Receiver("FSM", 60.02608, -111.93164, 0.0, VerticalDipole()),
    :RAB => Receiver("RAB", 58.2197, -103.6964, 0.0, VerticalDipole()), #UPDATE after deployment
    :CH => Receiver("CH", 58.7686, -94.1725, 0.0, VerticalDipole()), #UPDATE after looking at data
    :GIL => Receiver("GIL", 56.377043, -94.64403, 0.0, VerticalDipole()),
    :ISL => Receiver("ISL", 53.85517, -94.65967, 0.0, VerticalDipole()),
    :PIN => Receiver("PIN", 50.25812, -95.86516, 0.0, VerticalDipole()),
    :LAM => Receiver("LAM", 46.3522, -98.2919, 0.0, VerticalDipole()) #UPDATE after deployment
)

#structures useful for saving LMP outputs to be used in estimations
struct SimulationMultipleSample
    h_prime::Float64
    beta::Float64
    E::Vector{Complex{Float64}}
    amp::Vector{Float64}
    pha::Vector{Float64}
end

struct SimulationSingleSample
    h_prime::Float64
    beta::Float64
    E::Complex{Float64}
    amp::Float64
    pha::Float64
end

struct SimulationInputs
    dists::Vector{Float64}
    grnds::Vector{Ground}
    mags::Vector{BField}
    tx::Transmitter
    rx::Receiver
    params::LMPParams
end

const CHAOS = PyNULL()
const CHAOS_MODEL = PyNULL()

export TRANSMITTER
export RECEIVER
export range
export get_ground, get_groundcode, get_epsilon, get_sigma, groundsegments
export igrf, chaos, load_CHAOS_matfile
export zenithangle, isday, ferguson, flatlinearterminator, smoothterminator, fourierperturbation
export gen_terminator_raster, get_subsolar_point, gcp_percent_day

function __init__()
    cp = try
        pyimport("chaosmagpy")
    catch
        # Install chaosmagpy
        Conda.add(["numpy", "scipy", "pandas", "cython", "matplotlib"],
            Conda.ROOTENV)

        Conda.pip_interop(true, Conda.ROOTENV)
        Conda.pip("install", ["cdflib", "hdf5storage"])
        Conda.pip("install", "chaosmagpy")

        pyimport("chaosmagpy")
    end

    chaosfile = newestchaos()
    @info "Loading $chaosfile"

    model = cp.load_CHAOS_matfile(joinpath(project_path("data"), chaosfile))
    copy!(CHAOS, cp)
    copy!(CHAOS_MODEL, model)
end

function newestchaos()
    datafiles = readdir(project_path("data"); sort=true)
    pattern = r"CHAOS-([0-9]+\.?[0-9]*)\.mat"

    newestidx = 1
    newestversion = v"0.0"
    for (i, f) in enumerate(datafiles)
        m = match(pattern, f)
        isnothing(m) && continue
        versionnumber = parse(VersionNumber, m[1])

        if versionnumber > newestversion
            newestversion = versionnumber
            newestidx = i
        end
    end
    return datafiles[newestidx]
end

"""
    load_CHAOS_matfile(file)

Load a CHAOS model coefficient MATLAB `.mat` file for use by [`chaos`](@ref).

These files can be found at https://www.spacecenter.dk/files/magnetic-models/CHAOS-7/.

By default, LMPTools loads $(newestchaos()).

The model coefficients are loaded by the Python package
[chaosmagpy](https://pypi.org/project/chaosmagpy/), which is under MIT license with
Copyright (c) 2022 Clemens Kloss.
"""
function load_CHAOS_matfile(file)
    cp = pyimport("chaosmagpy")
    @info "Loading $file"
    model = cp.load_CHAOS_matfile(file)
    copy!(CHAOS, cp)
    copy!(CHAOS_MODEL, model)

    return nothing
end

"""
    load_CHAOS_matfile(version::VersionNumber)

Load CHAOS model `version`.

Only versions `v7.8` through `v7.11` and `v7.18` are provided with this package. Users can download
other versions from https://www.spacecenter.dk/files/magnetic-models/CHAOS-7/ onto their
local machine and specify the file path to this function instead.

# Example
```julia
load_CHAOS_matfile(v"7.11")
```
"""
function load_CHAOS_matfile(version::VersionNumber)
    if version == v"7.8"
        load_CHAOS_matfile(joinpath(project_path("data"), "CHAOS-7.8.mat"))
    elseif version == v"7.9"
        load_CHAOS_matfile(joinpath(project_path("data"), "CHAOS-7.9.mat"))
    elseif version == v"7.10"
        load_CHAOS_matfile(joinpath(project_path("data"), "CHAOS-7.10.mat"))
    elseif version == v"7.11"
        load_CHAOS_matfile(joinpath(project_path("data"), "CHAOS-7.11.mat"))
    elseif version == v"7.18"
        load_CHAOS_matfile(joinpath(project_path("data"), "CHAOS-7.18.mat"))
    else
        @warn "CHAOS $version not found. Model coefficients can be found here: https://www.spacecenter.dk/files/magnetic-models/CHAOS-7/"
    end

    return nothing
end

###
# Extend library functions

function GeographicLib.GeodesicLine(tx::Transmitter, rx::Receiver)
    return GeodesicLine(tx.longitude, tx.latitude; lon2=rx.longitude, lat2=rx.latitude)
end

"""
    range(tx::Transmitter, rx::Receiver)

Return the range (great-circle distance, but on the WGS-84 ellipsoid) in meters between
`tx` and `rx`.
"""
function Base.range(tx::Transmitter, rx::Receiver)
    return inverse(tx.longitude, tx.latitude, rx.longitude, rx.latitude).dist
end


###
# Ground

function searchsortednearest(a, x)
    @assert issorted(a) "`a` must be sorted!"

    idx = searchsortedfirst(a, x)

    idx == 1 && return idx
    idx > length(a) && return length(a)
    a[idx] == x && return idx
    abs(a[idx]-x) < abs(a[idx-1]-x) ? (return idx) : (return idx-1)
end

"""
    get_ground(param, lat, lon)

Return the ground `param` "codemap", "epsilonmap", or "sigmamap" at position `lat`, `lon`
in degrees.

Convenience functions are available for the most common fields with arguments `lat`, `lon`
only:
    
    * [`get_groundcode`](@ref)
    * [`get_sigma`](@ref)
    * [`get_epsilon`](@ref)

# References

[Ferguson1998]: J. A. Ferguson, “Computer programs for assessment of long-wavelength radio
    communications, version 2.0: User’s guide and source files,” Space and Naval Warfare
    Systems Center, San Diego, CA, Technical Document 3030, May 1998. Accessed: Feb. 14, 2017.
    [Online]. Available: http://www.dtic.mil/docs/citations/ADA350375

[Morgan1968]: R. R. Morgan, “World-wide VLF effective-conductivity map,” Westinghouse
    Electric Corporation, Westinghouse Electric Corporation Report WEST-80133F-1, Jan. 1968.
    Accessed: Dec. 14, 2017. [Online]. Available: http://www.dtic.mil/docs/citations/AD0675771
"""
function get_ground(param, lat, lon)
    latidx = searchsortednearest(GROUND_DATA["lat"], lat)
    lonidx = searchsortednearest(GROUND_DATA["lon"], lon)

    return GROUND_DATA[param][latidx, lonidx]
end
get_groundcode(lat, lon) = get_ground("codemap", lat, lon)
get_sigma(lat, lon) = get_ground("sigmamap", lat, lon)
get_epsilon(lat, lon) = get_ground("epsilonmap", lat, lon)


"""
    groundsegments(tx::Transmitter, rx::Receiver; resolution=20e3, require_permittivity_change=false)

Return a `Vector` of `Ground`'s and a `Vector` of distances at which the ground changes
between `tx` and `rx` using path `resolution` in meters.

If `require_permittivity_change == true`, then if the relative permittivity doesn't change
we don't consider it a ground change. This helps reduce the number of short small changes
that occur during a long path which have the same permittivity but a small change in
conductivity that has a small influence on the electric field in the EIWG but comes a the
cost of additional (potentially nearly identical) segments in the waveguide model.
"""
function groundsegments(tx::Transmitter, rx::Receiver; resolution=20e3, require_permittivity_change=false)
    line = GeodesicLine(tx, rx)
    pts = waypoints(line, dist=resolution)

    lat, lon, dist = first(pts).lat, first(pts).lon, first(pts).dist
    lastground = GROUND[get_groundcode(lat, lon)]

    # initialize
    grounds = [lastground]
    distances = [dist]  # == 0.0

    for pt in pts
        lat, lon, dist = pt.lat, pt.lon, pt.dist
        ground = GROUND[get_groundcode(lat, lon)]

        if require_permittivity_change
            if ground.ϵᵣ != lastground.ϵᵣ
                push!(grounds, ground)
                push!(distances, dist)
                lastground = ground
            end
        elseif ground != lastground
            push!(grounds, ground)
            push!(distances, dist)
            lastground = ground
        end
    end

    return grounds, distances
end


###
# Magnetic field

"""
    igrf(geoaz, lat, lon, year; alt=60e3)

Return a `BField` from IGRF-13 for position (`lat`, `lon`)°  at fractional
`year`. `geoaz`° is the geodetic azimuth of the path from transmitter to receiver.
By default, the magnetic field at an altitude of 60,000 meters is returned,
but this can be overridden with the `alt` keyword argument.
"""
function igrf(geoaz, lat, lon, year; alt=60e3)
    n, e, u = igrfd(year, alt, lat, lon, Val(:geodetic))
    t = hypot(n, e, u)

    # Rotate the neu frame to the propagation path xyz frame
    # negate az to correct rotation direction for downward pointing v
    R = RotXZ(π, -deg2rad(geoaz))
    Rd = R*SVector(n, e, u)

    return BField(t*1e-9, Rd[1]/t, Rd[2]/t, Rd[3]/t)
end

"""
    igrf(tx::Transmitter, rx::Receiver, year, dists; alt=60e3)

Return a `Vector{BField}` at each distance in `dists` in meters along the path from `tx`
to `rx` in fractional `year`.

If applying `igrf` to a single distance, it is recommended to use the `geoaz`, `lat`, `lon`
form of [`igrf`](@ref). 
"""
function igrf(tx::Transmitter, rx::Receiver, year, dists; alt=60e3)
    line = GeodesicLine(tx, rx)
    geoaz = inverse(tx.longitude, tx.latitude, rx.longitude, rx.latitude).azi

    bfields = Vector{BField}(undef, length(dists))
    for (i, d) in enumerate(dists)
        lo, la, _ = forward(line, d)
        bfields[i] = igrf(geoaz, la, lo, year; alt=alt)
    end
    return bfields
end

"""
    chaos(geoaz, lat::Real, lon:Real, year; alt=60e3)

Return a `BField` from CHAOS-7 using internal and external sources for position
(`lat`, `lon`)°  at fractional `year`.

`geoaz`° is the geodetic azimuth of the path from transmitter to receiver.
By default, the magnetic field at an altitude of 60,000 meters is returned,
but this can be overridden with the `alt` keyword argument.

See also: [`load_CHAOS_matfile`](@ref)

# References

[Finlay2019]: Finlay, C.C., Kloss, C., Olsen, N., Hammer, M. and Toeffner-Clausen, L.,
    (2019) DTU Candidate models for IGRF-13. Technical Note submitted to IGRF-13 task force,
    1st October 2019
[Finlay2020]: Finlay, C.C., Kloss, C., Olsen, N., Hammer, M. Toeffner-Clausen, L.,
    Grayver, A and Kuvshinov, A. (2020), The CHAOS-7 geomagnetic field model and observed
    changes in the South Atlantic Anomaly, Earth Planets and Space 72,
    doi:10.1186/s40623-020-01252-9
"""
function chaos(geoaz, lat::Number, lon::Number, year; alt=60e3)
    Re = 6371.2
    r = Re + alt/1000
    t = CHAOS.data_utils.dyear_to_mjd(year)  # MJD2000

    theta = 90 - lat  # geocentric co-lat (deg)
    phi = lon  # geocentric lon (deg)

    Br, Bt, Bp = CHAOS_MODEL(t, r, theta, phi)
    u, s, e = only(Br), only(Bt), only(Bp)

    # Rotate the "use" frame to the propagation path xyz frame
    # negate az to correct rotation direction for downward pointing v
    R = RotXZ(π, -deg2rad(geoaz))
    Rd = R*SVector(-s,e,-u)

    mag = hypot(u, s, e)
    return BField(mag*1e-9, Rd[1]/mag, Rd[2]/mag, Rd[3]/mag)
end

"""
    chaos(tx::Transmitter, rx::Receiver, year, dists; alt=60e3)

Return a `Vector{BField}` at each distance in `dists` in meters along the path from `tx`
to `rx` in fractional `year`.

If applying `chaos` to a single distance, it is recommended to use the `geoaz`, `lat`, `lon`
form of [`chaos`](@ref).
"""
function chaos(tx::Transmitter, rx::Receiver, year, dists; alt=60e3)
    line = GeodesicLine(tx, rx)
    geoaz = inverse(tx.longitude, tx.latitude, rx.longitude, rx.latitude).azi

    lats = Vector{Float64}(undef, length(dists))
    lons = similar(lats)
    for (i, d) in enumerate(dists)
        lo, la, _ = forward(line, d)
        lats[i] = la
        lons[i] = lo
    end

    bfields = chaos(geoaz, lats, lons, year; alt)

    return bfields
end

"""
    chaos(geoaz, lats, lons, year; alt=60e3)

Return a `Vector{BField}` at each latitude-longitude pair formed from each element of `lats`
and `lons` in fractional `year`.
"""
function chaos(geoaz, lats, lons, year; alt=60e3)
    length(lats) == length(lons) || throw(ArgumentError("length(lats) must equal length(lons)"))

    Re = 6371.2
    r = Re + alt/1000
    t = CHAOS.data_utils.dyear_to_mjd(year)  # MJD2000

    theta = 90 .- lats  # geocentric co-lat (deg)
    phi = lons  # geocentric lon (deg)

    u, s, e = CHAOS_MODEL(t, r, theta, phi)

    # Rotate the "use" frame to the propagation path xyz frame
    # negate az to correct rotation direction for downward pointing v
    R = RotXZ(π, -deg2rad(geoaz))

    bfields = Vector{BField}(undef, length(lats))
    for i in eachindex(bfields)
        Rd = R*SVector(-s[i],e[i],-u[i])

        mag = hypot(u[i], s[i], e[i])
        bfields[i] = BField(mag*1e-9, Rd[1]/mag, Rd[2]/mag, Rd[3]/mag)
    end

    return bfields
end


###
# Sun position

"""
    zenithangle(lat, lon, y, m, d, h, Δτ=nothing)

Return the solar zenith angle in degrees for year `y`, month `m`, day `d`, and hour `h` in
universal time (UT1). `Δτ` is the difference between terrestrial time and universal time.
The position in `lat`, `lon` is in degrees.

If `Δτ` is `nothing`, it will be computed by a linear extrapolation. Although this is not
accurate because `Δτ` is irregular, an error up to 30 seconds does not significantly affect
the result.

This function uses the calculation-optimized formulation of Algorithm 5 from

    Grena, R., “Five new algorithms for the computation of sun position from 2010 to 2110,”
    Solar Energy, 86, 2012, pp. 1323-1337. doi: 10.1016/j.solener.2012.01.024

which has a maximum error of 0.0027° (~10 arcsec) between year 2010 and 2110.

Refraction is not considered in this implementation.
"""
function zenithangle(lat, lon, yr::Int, mo::Int, dy::Int, hr::Real, Δτ=nothing)
    θ, ϕ = deg2rad(lon), deg2rad(lat)

    # Time scale computation
    if mo <= 2
        mo += 12
        yr -= 1
    end

    t = trunc(Int, 365.25*(yr - 2000)) + trunc(Int, 30.6001*(mo + 1)) -
        trunc(Int, 0.01*yr) + dy + 0.0416667*hr - 21958

    if isnothing(Δτ)
        Δτ = 96.4 + 0.00158*t
    end

    te = t + 1.1574e-5*Δτ

    # Heliocentric ecliptic longitude L
    ωa = 0.0172019715
    a = SVector(3.33024e-2, 3.512e-4, 5.2e-6)
    b = SVector(-2.0582e-3, -4.07e-5, -9e-7)
    s1, c1 = sincos(ωa*te)
    s2, c2 = 2*s1*c1, (c1 + s1)*(c1 - s1)
    s3, c3 = s2*c1 + c2*s1, c2*c1 - s2*s1
    s, c = SVector(s1, s2, s2), SVector(c1, c2, c3)

    β = 2.92e-5
    dβ = -8.23e-5
    ω = SVector(1.49e-3, 4.31e-3, 1.076e-2, 1.575e-2, 2.152e-2, 3.152e-2, 2.1277e-1)
    d = SVector(1.27e-5, 1.21e-5, 2.33e-5, 3.49e-5, 2.67e-5, 1.28e-5, 3.14e-5)
    φ = SVector(-2.337, 3.065, -1.533, -2.358, 0.074, 1.547, -0.488)

    L = 1.7527901 + 1.7202792159e-2*te
    for k = 1:3
        L += a[k]*s[k] + b[k]*c[k]
    end
    L += dβ*s1*sin(β*te)
    for i = 1:7
        L += d[i]*sin(ω[i]*te + φ[i])
    end

    # Nutation correction
    ωn = 9.282e-4
    ν = ωn*te - 0.8
    sν, cν = sincos(ν)
    Δλ = 8.34e-5*sν
    λ = L + π + Δλ
    ϵ = 4.089567e-1 - 6.19e-9*te + 4.46e-5*cν
    sλ, cλ = sincos(λ)
    sϵ, cϵ = sincos(ϵ)

    α = atan(sλ*cϵ, cλ)
    α < 0 && (α += 2π)  # [-π/2, π/2] → [0, 2π]

    δ = asin(sλ*sϵ)
    H = 1.7528311 + 6.300388099*t + θ - α + 0.92*Δλ
    H = mod2pi(H + π) - π  # hour angle

    # Calculate zenith and azimuth angle with parallax correction for topocentric coords
    sϕ, cϕ = sincos(ϕ)
    sδ, cδ = sincos(δ)
    cH = cos(H)
    # sH, cH = sincos(H)

    se0 = sϕ*sδ + cϕ*cδ*cH
    ep = asin(se0) - 4.26e-5*sqrt(1 - se0^2)

    # Γ = atan(sH, cH*sϕ - sδ*cϕ/cδ)  # azimuth

    # Refraction correction to elevation
    # Δre = 0.08422*P/(273 + T)/tan(ep + 0.003138/(ep + 0.08919))

    z = π/2 - ep  # - Δre  # zenith angle

    return rad2deg(z)
end

"""
    zenithangle(lat, lon, dt::DateTime, Δτ=nothing)

Return the solar zenith angle in degrees for universal `dt` with `Δτ` difference in seconds
between terrestrial time and universal time. The position `lat`, `lon` should be in degrees.
"""
function zenithangle(lat, lon, dt::DateTime, Δτ=nothing)
    y, m, d = yearmonthday(dt)
    h = hour(dt) + minute(dt)/60 + second(dt)/3600 + millisecond(dt)/3600_000

    return zenithangle(lat, lon, y, m, d, h, Δτ)
end

"""
    isday(sza)

Return `true` if solar zenith angle `sza` is less than 98°, otherwise return `false`.
"""
function isday(sza)
    abs(sza) < 98 ? true : false
end

"""
    ferguson(lat, sza, month_num::Int) → (h′, β)
    ferguson(lat, sza, dt::DateTime)

Ferguson ionosphere for geographic latitude `lat` in degrees, solar zenith angle `sza` in
degrees, and month number `month_num` or `DateTime` `dt`.

If the month term of the ionosphere fit is not desired, specify `month_num = 0`.

The higher order terms in [^Ferguson1980] (sunspot number and magnetic absorption) are ignored.

# References

[Ferguson1980]: J. A. Ferguson, "Ionospheric profiles for predicting nighttime VLF/LF
    propagation," NOSC/TR-530, 1980.
"""
function ferguson(lat, sza, month_num::Integer)
    lat = deg2rad(lat)
    sza = deg2rad(sza)

    cos_lat = cos(lat)
    cos_sza = cos(sza)

    month_scalar = cospi(2*(month_num-0.5)/12)

    hprime = 74.37 - 8.097*cos_sza + 5.779*cos_lat - 1.213*month_scalar
    beta = 0.3849 - 0.1658*cos_sza - 0.08584*cos_lat + 0.1296*month_scalar

    return hprime, beta
end
ferguson(lat, sza, dt::DateTime) = ferguson(lat, sza, month(dt))

"""
    flatlinearterminator(sza, hp_day=74, hp_night=86, b_day=0.3, b_night=0.5) → (h′, β)

A simple terminator model with flat day and night ionospheres and a linear transition
between `sza_min` and `sza_max`. Solar zenith angle `sza` is in degrees.
"""
function flatlinearterminator(sza, hp_day=74, hp_night=86, b_day=0.3, b_night=0.5;
    sza_min=90, sza_max=100)

    if sza_min < sza < sza_max
        # Linear trend between day and night
        m_hprime = (hp_night - hp_day) / (sza_max - sza_min)
        m_beta = (b_night - b_day) / (sza_max - sza_min)

        hprime = m_hprime*(sza - sza_min) + hp_day
        beta = m_beta*(sza - sza_min) + b_day
    elseif sza < sza_min
        hprime = hp_day
        beta = b_day
    else  # sza > sza_max
        hprime = hp_night
        beta = b_night
    end

    return hprime, beta
end

"""
    flatlinearterminator(sza::AbstractArray; hp_day=74, hp_night=86, b_day=0.3, b_night=0.5) → (h′, β)

Given an array of `sza`, return the corresponding arrays of h′s and βs.
"""
function flatlinearterminator(sza::AbstractArray, hp_day=74, hp_night=86, b_day=0.3, b_night=0.5;
    sza_min=90, sza_max=100)

    # Linear trend between day and night
    m_hprime = (hp_night - hp_day) / (sza_max - sza_min)
    m_beta = (b_night - b_day) / (sza_max - sza_min)

    hprimes = similar(sza, Float64)
    betas = similar(hprimes)
    for i in eachindex(sza)
        if sza_min < sza[i] < sza_max  
            hprimes[i] = m_hprime*(sza[i] - sza_min) + hp_day
            betas[i] = m_beta*(sza[i] - sza_min) + b_day
        elseif sza[i] < sza_min
            hprimes[i] = hp_day
            betas[i] = b_day
        else  # sza > sza_max
            hprimes[i] = hp_night
            betas[i] = b_night
        end
    end

    return hprimes, betas
end

"""
    smoothterminator(sza, hp_day=74, hp_night=86, b_day=0.3, b_night=0.5;
        sza_min=90, sza_max=100, steepness=1) → (h′, β)
    
A smooth terminator model using a logistic curve from day to night ionospheres as a function
of solar zenith angle `sza`.

The `steepness` parameter `1` with the default values of `sza_min` and `sza_max`
approximately asymptotes to the day and night values by `sza_min` and `sza_max` and is
centered on their midpoint.
"""
function smoothterminator(sza, hp_day=74, hp_night=86, b_day=0.3, b_night=0.5;
    sza_min=90, sza_max=100, steepness=1)

    hp_night > hp_day || throw(ArgumentError("`hp_night` should be greater than `hp_day`."))
    b_night > b_day || throw(ArgumentError("`b_night` should be greater than `b_day`."))
    sza_max > sza_min || throw(ArgumentError("`sza_max` should be greater than `sza_min`."))

    hprime_range = hp_night - hp_day
    beta_range = b_night - b_day
    sza_mid = (sza_max - sza_min)/2 + sza_min

    # logistic curve
    hprime = hprime_range/(1 + exp(-steepness*(sza - sza_mid))) + hp_day
    beta = beta_range/(1 + exp(-steepness*(sza - sza_mid))) + b_day

    return hprime, beta
end

"""
    fourierperturbation(sza, coeff; a=deg2rad(45)/2)

Return the perturbation produced by the Fourier series:
```
pert = coeff[1] + coeff[2]*cos(sza*π/a) + coeff[3]*cos(sza*2π/a) + coeff[4]*sin(sza*π/a) + coeff[5]*sin(sza*2π/a)
```
for solar zenith angle `sza` in degrees.

This can be used to produce simulated ionospheres in conjunction with [`ferguson`](@ref).

# Examples
```julia-repl
julia> lat, lon = 45, -105
julia> dt = DateTime(2020, 1, 15);
julia> sza = zenithangle(lat, lon, dt);
julia> h, b = ferguson(lat, sza, dt);

julia> coeffH = (1.345, 0.668, -0.177, -0.248, 0.096);
julia> hpert = fourierperturbation(sza, coeffH)
1.7991484480844766

julia> h + hpert
79.5769471288202
```
"""
function fourierperturbation(sza, coeff; a=deg2rad(45)/2)
    sza = deg2rad(sza)

    pert = coeff[1] +
        coeff[2]*cospi(sza/a) +
        coeff[4]*sinpi(sza/a) +
        coeff[3]*cospi(sza*2/a) +
        coeff[5]*sinpi(sza*2/a)
    
    return pert
end

#Functions below added by AVID team, additional documentation needed 

"""
    get_sunrise_sunset(latitude, longitude, year, day_of_year, alt) -> (Float64,Float64)

Return times of the next sunrise and sunset in seconds UTC for a provided position 
    and date.

# Arguments
- Position is provided in `longitude`, `latitude` (decimal degrees), and 
    `alt` (meters)
- Time is provided in `year`, and `day_of_year` (1 is Jan 1)

# Example
```jldoctest
julia> sunrise,sunset = get_sunrise_sunset(40.014984,-105.270546,2024,275,0)
(46568.12530180309, 89095.38700982607)
```
"""
function get_sunrise_sunset(latitude::Number,longitude::Number,year::Int,
    day_of_year::Number,alt::Number)
    if mod(year,4) == 0 && mod(year,100) != 0 #Looking for leap years
        #fraction of year (radians), assume time is midnight UTC
        frac_year = 2*pi/366 * (day_of_year-1)
    else
        frac_year = 2*pi/365 * (day_of_year-1)
    end
    #eqn of time (minutes)
    eqtime = 229.18*(0.000075 + 0.001868*cos(frac_year) - 0.032077*sin(frac_year) - 
        0.014615*cos(2*frac_year) - 0.040849*sin(2*frac_year))
    #solar declination angle (radians)
    decl = 0.006918-0.39912*cos(frac_year)+0.070257*sin(frac_year)-
        0.006758*cos(2*frac_year)+0.000907*sin(2*frac_year)-
        0.002697*cos(3*frac_year)+0.00148*sin(3*frac_year)
    #hour angle (degrees)
    ha = acosd((sind(-0.833-0.0347*sqrt(alt))/(cosd(latitude)*cos(decl)))-
        tand(latitude)*tan(decl))
    #Sunrise (seconds)
    sunrise = (720 - 4*(longitude + ha) - eqtime)*60
    #Sunset (seconds)
    sunset = (720 - 4*(longitude - ha) - eqtime)*60
    # Convert to Seconds since timing data is in seconds.
    #If Sunset > 86400, that means local sunset is the next day in UTC
    return sunrise,sunset
end

"""
    gen_terminator_raster(degree_res,year::Int,day_of_year::Int; <keyword arguments>)
"""
function gen_terminator_raster(degree_res::Number,year::Int,day_of_year::Int;
    alt::Number=60e3,time_of_day::Number=0,
    isFilled::Bool=true,exportMap::Bool=false,mapName::String="terminator")

    resolution = 1/degree_res
    long_mat = -180:degree_res:180
    day_of_year = day_of_year + (time_of_day/86400)
    
    # Draw a straight line down the prime meridian
    lat_mat = 90:(degree_res*-1):-90

    llmat = zeros(length(lat_mat),length(long_mat))

    pointsmat = zeros(2*length(lat_mat),2)

    for i=axes(lat_mat,1)
        # Try to get sunrise and sunset times for each point
        time_tuple = try
            get_sunrise_sunset(lat_mat[i],0,year,day_of_year,alt)
        catch zeroOrTwoSunsets # If error, then point has either two or zero sunset points during the day. 
            NaN # This error can be ignored, as these points lie outside of the terminator line.
        end

        sunrise_long = NaN
        sunset_long = NaN

        # Turn time into longitude coordinates
        if !isnan(time_tuple[1])
            sunrise_long = mod((((time_tuple[1]-43200)/86400) * 360),360) - 180
            sunset_long = mod((((time_tuple[2]-43200)/86400) * 360),360) - 180
        end

        pointsmat[i,1] = lat_mat[i]
        pointsmat[i,2] = sunrise_long

        pointsmat[(i+length(lat_mat)),1] = lat_mat[i]
        pointsmat[(i+length(lat_mat)),2] = sunset_long

        # Represent coordinates in matrix
        if !isnan(sunrise_long)
            j_rise = round(Int,(sunrise_long+180)*resolution)+1
            j_set = round(Int,(sunset_long+180)*resolution)+1
            llmat[i,j_rise] = 1
            llmat[i,j_set] = 1
            
            # Optionally fill in area where it is "daytime"
            if isFilled == true
                llmat[i,j_rise:length(long_mat)] = ones(1,length(j_rise:length(long_mat)))
                llmat[i,1:j_set] = ones(1,length(1:j_set))
            end
        end

        
    end

    pointsmat = sortslices(pointsmat, dims=1, lt=(x,y)->isless(x[2],y[2]))

    points_longs = filter(!isnan,pointsmat[:,2])
    new_pointsmat = zeros(length(points_longs),2)
    new_pointsmat[:,2] = points_longs
    new_pointsmat[:,1] = pointsmat[1:length(points_longs),1]
    pointsmat = new_pointsmat

    if isFilled == true
        # Fills the top and bottom sections of the map accordingly
        topIndex = 1
        while (iszero(llmat[topIndex,:]))
            topIndex += 1
        end

        if mean(llmat[topIndex,:]) >= mean(llmat[topIndex+1,:])
            llmat[1:topIndex-1,:] = ones(1:topIndex-1,length(long_mat));
        end

        bottomIndex = length(llmat[:,1])
        while (iszero(llmat[bottomIndex,:]))
            bottomIndex -= 1
        end

        if mean(llmat[bottomIndex,:]) >= mean(llmat[bottomIndex-1,:])
            llmat[bottomIndex+1:length(llmat[:,1]),:] = ones(bottomIndex+1:length(llmat[:,1]),length(long_mat))
        end

    end

    if time_of_day != 0
        degree_shift = mod((((time_of_day)/86400) * 360),360) - 180
        j_shift = round(Int,(degree_shift+180)*resolution)+1
        right_end = length(long_mat)
        
        llmat_shifted = zeros(length(lat_mat),length(long_mat))
        llmat_shifted[:,1:(right_end - j_shift + 1)] = llmat[:,j_shift:right_end]
        llmat_shifted[:,(right_end - j_shift + 2):right_end] = llmat[:,1:(j_shift - 1)]
        llmat = llmat_shifted

        points_shift = - mod((((time_of_day+43200)/86400) * 360),360) + 180
        @. pointsmat[:,2] = mod(pointsmat[:,2] + points_shift + 180,360) - 180
        pointsmat = sortslices(pointsmat, dims=1, lt=(x,y)->isless(x[2],y[2]))
    end

    if exportMap == true
        AG = ArchGDAL
        datadir = normpath(@__DIR__,"..", "raster_data") # Folder "raster_data" needs to be added in order to export GeoTIFFs
        
        proj = AG.toWKT(AG.importPROJ4("+proj=longlat +datum=WGS84 +no_defs"))
        transform = [-180.0, degree_res, 0, 90.0, 0, -1*degree_res]

        llmat_map = convert(Array{Float32},transpose(llmat))

        AG.create(
		normpath(datadir, mapName * ".tif"),
		driver = AG.getdriver("GTiff"),
		width = size(llmat_map,1),
		height = size(llmat_map,2),
		nbands = 1,
		dtype=Float32,
	) do dataset
		AG.write!(dataset, llmat_map, 1)

		AG.setgeotransform!(dataset, transform)
		AG.setproj!(dataset, proj)
	    end
        
    end

    return llmat,pointsmat
end

"""
    gen_terminator_raster(degree_res,datetime::DateTime; <keyword arguments>)

Generate a matrix which depicts the ionospheric terminator at a provided resolution 
    (`degree_res` in degrees per pixel), and datetime UTC. 
    
# Outputs
- A matrix that depicts the terminator. A value of `1` is day, and `0` is night.
- A matrix that contains all computed latitudes and longitudes of the computed
terminator line

# Arguments
- `alt::Number=60e3`: the height at which the terminator is computed (meters).
- `time_of_day::Number=0`: only when using `year` and `day_of_year` (1 being Jan 1). 
Time of day in seconds UTC at which to compute the terminator.
- `exportMap::Bool=false`: whether or not to export a GeoTIFF map of the
computed terminator.
- `isFilled::Bool=true`: whether or not to make all night values in the matrix `0`. 
A value of `true` would only make the terminator `0`.
- `mapName::String="terminator"`: if exporting the matrix as a GeoTIFF raster,
this would be the name of the image.

# Example
```jldoctest

```
"""
function gen_terminator_raster(degree_res::Number,datetime::DateTime;
    alt::Number=60e3,
    isFilled::Bool=false,exportMap::Bool=false,mapName="terminator")
    # Generates a raster which depicts the terminator. Utilizes the DateTime() type, with time assumed to be in UTC
    year = Dates.year(datetime)
    day_of_year = Dates.dayofyear(datetime)
    time_of_day = (Dates.hour(datetime)*3600) + (Dates.minute(datetime)*60) + 
        Dates.second(datetime)

    (llmat,pointsmat) = gen_terminator_raster(degree_res,year,day_of_year,
    alt=alt,time_of_day=time_of_day,
    isFilled=isFilled,exportMap=exportMap,mapName=mapName)

    return llmat,pointsmat

end

function get_subsolar_point(year::Int,day_of_year::Int,time_of_day::Number)
    # Generates the coordinates of the subsolar point at a given date and time
    if mod(year,4) == 0 && mod(year,100) != 0 #Looking for leap years
        #fraction of year (radians)
        frac_year = 2*pi/366 * (day_of_year-1/2)
    else
        frac_year = 2*pi/365 * (day_of_year-1/2)
    end
    #eqn of time (minutes)
    eqtime = 229.18*(0.000075 + 0.001868*cos(frac_year) - 0.032077*sin(frac_year) - 
        0.014615*cos(2*frac_year) - 0.040849*sin(2*frac_year))
    #solar declination angle(radians)
    decl = 0.006918-0.39912*cos(frac_year)+0.070257*sin(frac_year)-
        0.006758*cos(2*frac_year)+0.000907*sin(2*frac_year)-0.002697*cos(3*frac_year)+
        0.00148*sin(3*frac_year)
    
    subsolar_lat = decl*(360/2pi)
    subsolar_long = -15*((time_of_day/3600)-12+(eqtime/60))

    return subsolar_lat,subsolar_long
end

function get_subsolar_point(datetime::DateTime)
    # Generates a raster which depicts the terminator. Utilizes the DateTime() type, with time assumed to be in UTC
    year = Dates.year(datetime)
    day_of_year = Dates.dayofyear(datetime)
    time_of_day = (Dates.hour(datetime)*3600) + (Dates.minute(datetime)*60) + 
        Dates.second(datetime)

    (subsolar_lat,subsolar_long) = get_subsolar_point(year,day_of_year,time_of_day)

    return subsolar_lat,subsolar_long

end

function is_day(lat,long,datetime::DateTime;alt=60e3)
    # Returns a bool indicating if the provided point in space and time is in day or night
    # Will return nothing near the poles, at a latitude of roughly ±65 for alt=60e3
    year = Dates.year(datetime)
    day_of_year = Dates.dayofyear(datetime)
    time_of_day = (Dates.hour(datetime)*3600) + (Dates.minute(datetime)*60) + 
        Dates.second(datetime)

    # Get sunrise and sunset times
    (sunrise,sunset) = try get_sunrise_sunset(lat,long,year,day_of_year,alt)
        catch whoops
            # Sun either does not rise or set during this day, at this location
            return
        end

    if time_of_day > sunrise && time_of_day < sunset
        return true
    else
        return false
    end
end

function gcp_percent_day(start_lat,start_long,end_lat,end_long,datetime;
    resolution=20e3,alt=60e3)
    # Using provided start and end coordinates, returns the percentage of the Great Circle path within the terminator 

    line = GeodesicLine(start_long,start_lat,lon2=end_long,lat2=end_lat)
    pts = waypoints(line, dist=resolution)
    gcp_bool = zeros(length(pts))
    index = 1

    # check each point along the path to see if it is during the day
    for pt in pts
        lat, lon = pt.lat, pt.lon
        gcp_bool[index] = is_day(lat,lon,datetime,alt=alt)
        index += 1
    end
    
    # take an average
    GCP_percent_day = mean(gcp_bool)

    # return the GCP itself

    return GCP_percent_day,gcp_bool
end

function readSims(h,beta,filebase)
    pha= fill(NaN, size(beta,1), size(h,1)) #h, Beta, amp, pha
    amp= fill(NaN, size(beta,1), size(h,1)) #h, Beta, amp, pha
    
    for b in eachindex(beta)
        x = beta[b]
        temp=load(filebase*"$x.jld2","sorted")
        for i in eachindex(temp)
            pha[b,i] = temp[i].pha
            amp[b,i] = temp[i].amp
        end
    end
    return amp,pha
end

function circle_distance_deg(a,b)
    diff1 = abs(a-b)
    if a > 0
        diff2=abs((a-360)-b)
    elseif a<0
        diff2=abs((a+360-b))
    else
        diff2=diff1
    end
    
    if diff1<diff2
        return diff1
    else
        return diff2
    end
end

function ununwrapPhase(elem)
    if elem>180
        elem=ununwrapPhase(elem-360)
    elseif elem<-180
        elem=ununwrapPhase(elem+360)
    end
    return elem
end
#TODO refactor to while loops^&v
function recursive_dists_segmentation(current,next,max_diff,curr_gnd,next_gnd)
    diff_x = next-current
    if diff_x>max_diff
        dists,associated_gnds=recursive_dists_segmentation(current+max_diff,next,max_diff,curr_gnd,next_gnd)
        list = vcat(current,dists)
        gnds = vcat(curr_gnd,associated_gnds)
    else
        list = vcat(current,next)
        gnds = vcat(curr_gnd,next_gnd)
    end
    return list, gnds
end


function WGSegmentation(tx,rx;x_res=20e3,dec_year=2024,gs_res=50e3,max_WG_length=400e3)
    first_gnds, first_dists = groundsegments(tx,rx,resolution=x_res,require_permittivity_change=false)
    max_dist=range(tx,rx)

    append!(first_dists,max_dist)
    append!(first_gnds,first_gnds[[end]])

    max_idx = length(first_dists)
    dists=[]
    gnds=[]
    
    for i in 1:1:max_idx-1
        d, g = recursive_dists_segmentation(first_dists[i],first_dists[i+1],max_WG_length,first_gnds[i],first_gnds[i+1])
        if i > 1
            if dists[end]==d[1]
                    append!(dists,d[2:end])
                    append!(gnds,g[2:end])
            else
                    append!(dists,d)
                    append!(gnds,g)
            end
        else
            append!(dists,d)
            append!(gnds,g)
        end
    end

    pop!(dists)
    pop!(gnds)
    mag = igrf(tx,rx,dec_year,dists)
    num_segments = length(dists)

    sample_dists = collect(0:gs_res:max_dist)
    push!(sample_dists,max_dist)

    gs = GroundSampler(sample_dists,Fields.Ez)

    return num_segments, dists, gnds, mag, gs
end

function fromdB(dbuV)
    uV = 10 ^(dbuV/20) #uV/m
    V = uV*(10^-6)   #V/m
    return V
end

function todB(V)
    uV = V*10^6 #uV/m
    dbuV = 20*log10(uV) #dBuV/m
    return dbuV
end

function plotphamap(h,beta,sim_pha,rx,)
    figure()
    phamap=heatmap(h,beta,sim_pha,xlabel="h'",ylabel="\$\\beta\$",title="NLK phase at "*rx.name,color =cgrad(:viridis),clims=(-180,180),colorbar_title="Phase (\$ \\degree \$)",dpi=400)
    show()
    Plots.savefig(phamap,rx.name*"_PhaMap.png")
end

function plotampmap(h,beta,sim_amp,tv,rx;secondary_title="")
    figure()
    mi = minimum(tv)
    ma = maximum(tv)
    clims = (mi, ma)
    tl = ["$v" for v in tv]
    ampmap=heatmap(h,beta,(sim_amp),color =cgrad(:viridis),colorbar_ticks=(tv,tl),clims=clims,xlabel="h'",ylabel="\$\\beta\$",title="NLK amplitude at "*rx.name*secondary_title,colorbar_title="dB (uV/m)",dpi=400)
    show()
    
    Plots.savefig(ampmap,rx.name*"_AmpMap"*secondary_title*".png")
end

end  # module
