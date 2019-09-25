import FourierTransforms
import FourierTransforms: fft_kernel_sizes
using Test, LinearAlgebra, OffsetArrays

include("symbolic.jl")

function testdft(x::AbstractVector{T}) where T<:Union{Real,Complex}
    Tr = real(float(T))
    Tc = complex(float(T))
    y = similar(x, Tc)
    N = length(x)
    for (k1, yind) in enumerate(eachindex(y))
        fct = zero(Tr)
        yk  = zero(Tc)
        for xi in x
            yk  += xi*complex(cos(fct), -sin(fct))
            fct += one(Tr)/N*(k1 - 1)*2*π
        end
        y[yind] = yk
    end
    return y
end

@testset "Multiplication of size: $sz" for sz in [collect(fft_kernel_sizes); 511; 512; 513]
  @testset "Real plan" begin
      xr = rand(sz)
      @test FourierTransforms.fft(xr) ≈ testdft(xr)
      p = FourierTransforms.plan_fft(xr)
      @test p*xr == FourierTransforms.fft(xr)
      @test p\(p*xr) ≈ xr
  end

  @testset "Complex plan applied to $atype" for (atype, x) in (
      ("Vector", rand(Complex{Float64}, sz)),
      ("OffsetVector", OffsetArray(rand(Complex{Float64}, sz), 0:(sz - 1)))
  )
      p = FourierTransforms.plan_fft(x)
      @test mul!(similar(x), p, x) == p*x
      @test p*x ≈ testdft(x)
      @test p\(p*x) ≈ x
  end
end
