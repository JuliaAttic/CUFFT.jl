## This package is deprecated.

The same functionality is available in [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl).

# CUFFT

**Build status**: [![][buildbot-julia05-img]][buildbot-julia05-url] [![][buildbot-julia06-img]][buildbot-julia06-url]

[buildbot-julia05-img]: http://ci.maleadt.net/shields/build.php?builder=CUFFT-julia05-x86-64bit&name=julia%200.5
[buildbot-julia05-url]: http://ci.maleadt.net/shields/url.php?builder=CUFFT-julia05-x86-64bit
[buildbot-julia06-img]: http://ci.maleadt.net/shields/build.php?builder=CUFFT-julia06-x86-64bit&name=julia%200.6
[buildbot-julia06-url]: http://ci.maleadt.net/shields/url.php?builder=CUFFT-julia06-x86-64bit

This is a wrapper of the CUFFT library. It works in conjunction with the [CUDArt](https://github.com/JuliaGPU/CUDArt.jl) package.

## Usage example

Here's an example of taking a 2D real transform, and then it's inverse, and comparing against Julia's CPU-based 

```julia
using CUDArt, CUFFT, Base.Test

CUDArt.devices(dev->capability(dev)[1] >= 2, nmax=1) do devlist
    A = rand(7,6)
    # Move data to GPU
    G = CudaArray(A)
    # Allocate space for the output (transformed array)
    GFFT = CudaArray(Complex{eltype(A)}, div(size(G,1),2)+1, size(G,2))
    # Compute the FFT
    pl! = plan(GFFT, G)
    pl!(GFFT, G, true)
    # Copy the result to main memory
    AFFTG = to_host(GFFT)
    # Compare against Julia's rfft
    AFFT = rfft(A)
    @test_approx_eq AFFTG AFFT
    # Now compute the inverse transform
    pli! = plan(G,GFFT)
    pli!(G, GFFT, false)
    A2 = to_host(G)
    @test_approx_eq A A2/length(A)
end
```

#### Notes on memory

For those who dive into the internals, one potentially-confusing point is that C's (or FFTW's) convention for representing array dimensions is opposite that of Julia. C's convention stems from the static representation of arrays,

```
const NX = 3
const NY = 5
double *myarray[NX][NY] = {
  {1.0, 2.0, 3.0, 4.0, 5.0},
  {6.0, 7.0, 8.0, 9.0, 10.0},
  {11.0, 12.0, 13.0, 14.0, 15.0}};
```

Consequently, `NX` represents the number of rows, and `NY` the number of columns (even though visually `x` is the horizontal axis and `y` the vertical axis). The first dimension therefore does _not_ correspond to the "fast" dimension in linear-memory layout.
