import DFT, FFTW
import DFT: fft_kernel_sizes
using Test

# kernel test
@testset "FFT kernel codelet" begin
    for i in fft_kernel_sizes
        data = rand(i)
        @test DFT.fft(data) ≈ FFTW.fft(data)
    end
end
