module CUFFT
using CUDArt

import Base: convert
export Plan, compatibility, exec!, plan, stream
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

function plan_size(dest, src)
    nd = ndims(dest)
    ndims(src) == nd || throw(DimensionMismatch())
    sz = Array(Cint, nd)
    for i = 2:nd
        if size(dest,i) != size(src,i)
            throw(DimensionMismatch())
        end
        sz[i] = size(dest,i)
    end
    local plantype
    if eltype(dest) == eltype(src)
        size(dest,1) == size(src,1) || throw(DimensionMismatch())
        sz[1] = size(dest,1)
    else
        szbig = max(size(dest,1), size(src,1))
        szsmall = min(size(dest,1), size(src,1))
        szsmall == div(szbig,2)+1 || throw(DimensionMismatch())
        sz[1] = szbig
    end
    reverse(sz)
end

function plan(dest::AbstractCudaArray, src::AbstractCudaArray)
    p = Cint[0]
    sz = plan_size(dest, src)
    inembed = reverse(Cint[size(src)...])
    onembed = reverse(Cint[size(dest)...])
    inembed[end] = div(CUDArt.pitch(src), sizeof(eltype(src)))
    onembed[end] = div(CUDArt.pitch(dest), sizeof(eltype(dest)))
    plantype = plan_dict[(eltype(src),eltype(dest))]
    lib.cufftPlanMany(p, ndims(dest), sz, inembed, 1, 1, onembed, 1, 1, plantype, 1)
    pl = Plan{eltype(src),eltype(dest),ndims(dest)}(p[1])
    cudafinalizer(pl, destroy)
    pl
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
