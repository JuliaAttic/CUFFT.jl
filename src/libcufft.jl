module LibCUFFT
using Compat

import Base.Sys: WORD_SIZE

include("cufft_h.jl")
import CUDArt.rt.cudaStream_t

if is_windows()
    const libcufft = Libdl.find_library([string("cufft", WORD_SIZE, "_75"),
      string("cufft", WORD_SIZE, "_70"), string("cufft", WORD_SIZE, "_65")],
      [joinpath(ENV["CUDA_PATH"], "bin")])
else
    const libcufft = Libdl.find_library(["libcufft", "cufft"], ["/usr/lib", "/usr/local/cuda"])
end

isempty(libcufft) && error("Cannot load libcufft")

function checkerror(code::cufftResult)
    if code == CUFFT_SUCCESS
        return nothing
    end
    println("An error was triggered:")
    Base.show_backtrace(STDOUT, backtrace())
    throw(string(Symbol(code)))
end


function cufftPlan1d(plan, nx, _type, batch)
  checkerror(ccall( (:cufftPlan1d, libcufft), cufftResult, (Ptr{cufftHandle}, Cint, cufftType, Cint), plan, nx, _type, batch))
end
function cufftPlan2d(plan, nx, ny, _type)
  checkerror(ccall( (:cufftPlan2d, libcufft), cufftResult, (Ptr{cufftHandle}, Cint, Cint, cufftType), plan, nx, ny, _type))
end
function cufftPlan3d(plan, nx, ny, nz, _type)
  checkerror(ccall( (:cufftPlan3d, libcufft), cufftResult, (Ptr{cufftHandle}, Cint, Cint, Cint, cufftType), plan, nx, ny, nz, _type))
end
function cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, _type, batch)
  checkerror(ccall( (:cufftPlanMany, libcufft), cufftResult, (Ptr{cufftHandle}, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Cint, cufftType, Cint), plan, rank, n, inembed, istride, idist, onembed, ostride, odist, _type, batch))
end
function cufftDestroy(plan)
  checkerror(ccall( (:cufftDestroy, libcufft), cufftResult, (cufftHandle,), plan))
end
function cufftExecC2C(plan, idata, odata, direction)
  checkerror(ccall( (:cufftExecC2C, libcufft), cufftResult, (cufftHandle, Ptr{cufftComplex}, Ptr{cufftComplex}, Cint), plan, idata, odata, direction))
end
function cufftExecR2C(plan, idata, odata)
  checkerror(ccall( (:cufftExecR2C, libcufft), cufftResult, (cufftHandle, Ptr{cufftReal}, Ptr{cufftComplex}), plan, idata, odata))
end
function cufftExecC2R(plan, idata, odata)
  checkerror(ccall( (:cufftExecC2R, libcufft), cufftResult, (cufftHandle, Ptr{cufftComplex}, Ptr{cufftReal}), plan, idata, odata))
end
function cufftExecZ2Z(plan, idata, odata, direction)
  checkerror(ccall( (:cufftExecZ2Z, libcufft), cufftResult, (cufftHandle, Ptr{cufftDoubleComplex}, Ptr{cufftDoubleComplex}, Cint), plan, idata, odata, direction))
end
function cufftExecD2Z(plan, idata, odata)
  checkerror(ccall( (:cufftExecD2Z, libcufft), cufftResult, (cufftHandle, Ptr{cufftDoubleReal}, Ptr{cufftDoubleComplex}), plan, idata, odata))
end
function cufftExecZ2D(plan, idata, odata)
  checkerror(ccall( (:cufftExecZ2D, libcufft), cufftResult, (cufftHandle, Ptr{cufftDoubleComplex}, Ptr{cufftDoubleReal}), plan, idata, odata))
end
function cufftSetStream(plan, stream)
  checkerror(ccall( (:cufftSetStream, libcufft), cufftResult, (cufftHandle, cudaStream_t), plan, stream))
end
function cufftSetCompatibilityMode(plan, mode)
  checkerror(ccall( (:cufftSetCompatibilityMode, libcufft), cufftResult, (cufftHandle, cufftCompatibility), plan, mode))
end
function cufftGetVersion(version)
  checkerror(ccall( (:cufftGetVersion, libcufft), cufftResult, (Ptr{Cint},), version))
end

end
