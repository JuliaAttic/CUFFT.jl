import CUDArt
import CUFFT
using Base.Test

CUDArt.devices(dev->CUDArt.capability(dev)[1] >= 2, nmax=1) do devlist
    CUDArt.device(devlist[1])
    # A simple 1d transform
    n = 64
    nc = div(n,2)+1
    # Create an array and move to the GPU
    a = randn(n)
    g = CUDArt.CudaArray(a)
    # Allocate space for the FFT
    gfft = CUDArt.CudaArray(Complex128, nc)
    # Perform the FFT
    pl = CUFFT.plan((n,), Float64, Complex128)
    CUFFT.exec!(pl, g, gfft)
    afftg = CUDArt.to_host(gfft)
    afft = rfft(a)
    @test_approx_eq afftg afft
    # Perform the IFFT
    pli = CUFFT.plan((n,), Complex128, Float64)
    CUFFT.exec!(pli, gfft, g)
    # Move back to host and compare
    a2 = CUDArt.to_host(g)
    @test_approx_eq a a2/n
    # 2D transform
    NX, NY = 8, 5
    A = randn(NX, NY)
    G = CUDArt.CudaArray(A)
    GFFT = CUDArt.CudaArray(Complex128, div(NX,2)+1, NY)
    pl = CUFFT.plan(size(A), Float64, Complex128)
    CUFFT.compatibility(pl, :all)
    CUFFT.exec!(pl, G, GFFT)
    AFFTG = CUDArt.to_host(GFFT)
    AFFT = rfft(A)  # Note: these two aren't equal! Figure this out. There are combinations of transposes that get closer
#     @test_approx_eq AFFTG AFFT
    pli = CUFFT.plan((NX,NY), Complex128, Float64)
    CUFFT.exec!(pli, GFFT, G)
    A2 = CUDArt.to_host(G)
    @test_approx_eq A A2/length(A)
    # 3D transform using CudaArray
    NX, NY, NZ = 38, 75, 124
    A = randn(NX, NY, NZ)
    G = CUDArt.CudaArray(A)
    GFFT = CUDArt.CudaArray(Complex128, div(NX,2)+1, NY, NZ)
    pl = CUFFT.plan((NX,NY,NZ), Float64, Complex128)
    CUFFT.exec!(pl, G, GFFT)
    AFFTG = CUDArt.to_host(GFFT)
    AFFT = rfft(A)
#     @test_approx_eq AFFTG AFFT
    pli = CUFFT.plan((NX,NY,NZ), Complex128, Float64)
    CUFFT.exec!(pli, GFFT, G)
    A2 = CUDArt.to_host(G)
    @test_approx_eq A A2/length(A)
    # CudaPitchedArray
end
