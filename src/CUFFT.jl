module CUFFT
using CUDArt
using Compat

import Base: convert
import CUDArt: destroy
export Plan, compatibility, plan, tie, RCpair, RCfree
# "Public" but not exported: version()

include("libcufft.jl")
const lib = LibCUFFT

type Plan{From,To,N}
    p::Cint
end
convert(::Type{lib.cufftHandle}, p::Plan) = p.p

function destroy(p::Plan)
#     println("Destroying ", p)
#     Base.show_backtrace(STDOUT, backtrace())
    lib.cufftDestroy(p)
end

plan_dict = @compat Dict(
    (Float32,Complex64) => lib.CUFFT_R2C,
    (Complex64,Float32) => lib.CUFFT_C2R,
    (Complex64,Complex64) => lib.CUFFT_C2C,
    (Float64,Complex128) => lib.CUFFT_D2Z,
    (Complex128,Float64) => lib.CUFFT_Z2D,
    (Complex128,Complex128) => lib.CUFFT_Z2Z
)

# For in-place R2C and C2R transforms
function RCpair{T<:AbstractFloat}(realtype::Type{T}, realsize) # TODO?: add dims
    csize = [realsize...]
    csize[1] = div(realsize[1],2) + 1
    C = CudaPitchedArray(Complex{T}, csize...)
    R = reinterpret(T, C, tuple(realsize...))
    R, C
end

function RCpair{T<:AbstractFloat}(A::Array{T}) # TODO?: add dims
    R, C = RCpair(eltype(A), size(A))
    copy!(R, A)
    R, C
end

RCfree{T<:AbstractFloat}(R::CudaPitchedArray{T}, C::CudaPitchedArray{Complex{T}}) = free(C)

function plan_size(dest, src)
    nd = ndims(dest)
    ndims(src) == nd || throw(DimensionMismatch())
    sz = Vector{Cint}(nd)
    for i = 2:nd
        if size(dest,i) != size(src,i)
            throw(DimensionMismatch())
        end
        sz[i] = size(dest, i)
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

# Returns a function dofft!(dest, src, forward).
# Unlike FFTW's plan, this does not destroy the inputs
# TODO?: add dims
function plan(dest::AbstractCudaArray, src::AbstractCudaArray; compat::Symbol = :padding, stream=null_stream)
    p = Cint[0]
    sz = plan_size(dest, src)
    inembed = reverse(Cint[size(src)...])
    onembed = reverse(Cint[size(dest)...])
    inembed[end] = pitchel(src)
    onembed[end] = pitchel(dest)
    plantype = plan_dict[(eltype(src),eltype(dest))]
    lib.cufftPlanMany(p, ndims(dest), sz, inembed, 1, 1, onembed, 1, 1, plantype, 1)
    pl = Plan{eltype(src),eltype(dest),ndims(dest)}(p[1])
    #compatibility(pl, compat)
    tie(pl, stream)
    (dest,src,forward) -> exec!(pl, src, dest, forward)
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

modedict = Dict(
    :native     => lib.CUFFT_COMPATIBILITY_NATIVE,
    :padding    => lib.CUFFT_COMPATIBILITY_FFTW_PADDING,
    :asymmetric => lib.CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC,
    :all        => lib.CUFFT_COMPATIBILITY_FFTW_ALL
)

compatibility(p::Plan, mode::Symbol) = lib.cufftSetCompatibilityMode(p, modedict[mode])

tie(p::Plan, s::Stream) = lib.cufftSetStream(p, s)

end
