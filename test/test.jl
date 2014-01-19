include("../src/CUFFT.jl")
using CUFFT

NX, NY, NZ = 64, 64, 128

# I get segfaults when trying to use cuFFT with the Driver API
 using CUDA
 dev = CuDevice(0)
 ctx = create_context(dev)
 p = plan((NX, NY, NZ), Complex64, Complex64)

# include("/home/tim/julia/CUDArt/src/CUDArt.jl")
# using CUDArt
# 
# devices(dev->capability(dev)[1] >= 2) do devlist
#     p = plan((NX, NY, NZ), Complex64, Complex64)
# end
