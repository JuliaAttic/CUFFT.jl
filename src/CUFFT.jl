module CUFFT

import Base: fft!, ifft!
export Plan, compatibility, destroy, plan, stream
# "Public" but not exported: version()

include("libcufft.jl")
const lib = LibCUFFT

type Plan{From,To,N}
    p::Cint
end
convert(::Type{lib.cufftHandle}, p::Plan) = p.p

destroy(p::Plan) = lib.cufftDestroy(p)

plan_dict = [
    (Float32,Complex64) => lib.CUFFT_R2C,
    (Complex64,Float32) => lib.CUFFT_C2R,
    (Complex64,Complex64) => lib.CUFFT_C2C,
    (Float64,Complex128) => lib.CUFFT_D2Z,
    (Complex128,Float64) => lib.CUFFT_Z2D,
    (Complex128,Complex128) => lib.CUFFT_Z2Z
]

function plan{F,T}(dims::Dims, from::Type{F}, to::Type{T})
    n = length(dims)
    p = Cint[0]
    c = plan_dict[(from,to)]
    if n == 1
        lib.cufftPlan1d(p, dims..., c)
    elseif n == 2
        lib.cufftPlan2d(p, dims..., c)
    elseif n == 3
        lib.cufftPlan3d(p, dims..., c)
    else
        error("Must be 1, 2, or 3-dimensional")
    end
    Plan{from,to,n}(p[1])
end

direction(forward::Bool) = forward ? lib.CUFFT_FORWARD : lib.CUFFT_INVERSE

exec!(p::Plan{Complex64,Complex64}, input, output, forward::Bool = true) =
    lib.cufftExecC2C(p, input, output, direction(forward))
exec!(p::Plan{Float32,Complex64}, input, output, forward::Bool = true) =
    lib.cufftExecR2C(p, input, output)
exec!(p::Plan{Complex64,Float32}, input, output, forward::Bool = true) =
    lib.cufftExecC2R(p, input, output)
exec!(p::Plan{Complex128,Complex128}, input, output, forward::Bool = true) =
    lib.cufftExecZ2Z(p, input, output, direction(forward))
exec!(p::Plan{Float64,Complex128}, input, output, forward::Bool = true) =
    lib.cufftExecD2Z(p, input, output)
exec!(p::Plan{Complex128,Float64}, input, output, forward::Bool = true) =
    lib.cufftExecZ2D(p, input, output)
    
version() = (ret = Cint[0]; lib.cufftGetVersion(ret); ret[1])

modedict = [
    :native     => lib.CUFFT_COMPATIBILITY_NATIVE,
    :padding    => lib.CUFFT_COMPATIBILITY_FFTW_PADDING,
    :asymmetric => lib.CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC,
    :all        => lib.CUFFT_COMPATIBILITY_FFTW_ALL
]

compatibility(p::Plan, mode::Symbol) = lib.cufftSetCompatibilityMode(p, modedict[mode])

# TODO: stream

end
