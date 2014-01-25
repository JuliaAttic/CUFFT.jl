# CUFFT API function return values 
typealias cufftResult Cint
const CUFFT_SUCCESS        = 0x0
const CUFFT_INVALID_PLAN   = 0x1
const CUFFT_ALLOC_FAILED   = 0x2
const CUFFT_INVALID_TYPE   = 0x3
const CUFFT_INVALID_VALUE  = 0x4
const CUFFT_INTERNAL_ERROR = 0x5
const CUFFT_EXEC_FAILED    = 0x6
const CUFFT_SETUP_FAILED   = 0x7
const CUFFT_INVALID_SIZE   = 0x8
const CUFFT_UNALIGNED_DATA = 0x9
    
typealias cufftReal Float32
typealias cufftDoubleReal Float64

typealias cufftComplex Complex64
typealias cufftDoubleComplex Complex128

# CUFFT transform directions 
const CUFFT_FORWARD = -1 # Forward FFT
const CUFFT_INVERSE =  1 # Inverse FFT

# CUFFT supports the following transform types 
typealias cufftType Cint
const CUFFT_R2C = 0x2a     # Real to Complex
const CUFFT_C2R = 0x2c     # Complex to Real
const CUFFT_C2C = 0x29     # Complex to Complex
const CUFFT_D2Z = 0x6a     # Double to Double-Complex
const CUFFT_Z2D = 0x6c     # Double-Complex to Double
const CUFFT_Z2Z = 0x69     # Double-Complex to Double-Complex

typealias cufftCompatibility Cint
const   CUFFT_COMPATIBILITY_NATIVE          = 0x00
const   CUFFT_COMPATIBILITY_FFTW_PADDING    = 0x01
const   CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC = 0x02
const   CUFFT_COMPATIBILITY_FFTW_ALL        = 0x03

typealias cufftHandle Cint
