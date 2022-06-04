import FourierTransforms
import FourierTransforms: fft_kernel_sizes
using Test, LinearAlgebra, OffsetArrays

include("symbolic.jl")

function testdft(x::AbstractVector{T},forward=true) where T<:Union{Real,Complex}
    Tr = real(float(T))
    Tc = Complex{BigFloat}
    y = similar(x, Tc)
    N = length(x)
    d = big(π)*2/big(0.0+N)
    ifac = (-1)^forward
    for (k1, yind) in enumerate(eachindex(y))
        fct = zero(Tr)
        yk  = zero(Tc)
        for xi in x
            yk  += xi*complex(cos(fct), ifac * sin(fct))
            fct += d*(k1 - 1)
        end
        y[yind] = yk
    end
    return y
end

@testset "Type $T" for T in [Float64, BigFloat]
    @testset "Multiplication of size: $sz" for sz in [collect(2:16)...,18,30,105,121]
        @testset "Real plan" begin
            xr = rand(sz)
            @test FourierTransforms.fft(xr) ≈ testdft(xr)
            p = FourierTransforms.plan_fft(xr)
            @test p*xr == FourierTransforms.fft(xr)
            @test p\(p*xr) ≈ xr
        end

        @testset "Complex plan applied to $atype" for (atype, offset) in (
            ("Vector", false),
            ("OffsetVector", true)
            )
            # if T=BigFloat, we want to check against a better standard
            setprecision(BigFloat, 192)
            if offset
                x = OffsetArray(rand(Complex{T}, sz), 0:(sz - 1))
            else
                x = rand(Complex{T}, sz)
            end
            small = eps(T)
            for forward in (true, false)
                if forward
                    p = FourierTransforms.plan_fft(x)
                else
                    p = FourierTransforms.plan_bfft(x)
                end
                y = p*x
                @test mul!(similar(x), p, x) == y
                setprecision(BigFloat, 256)
                z = testdft(x, forward)
                u = Float64(maximum(abs.(y-z)) / (small*sz*log2(sz)))
                @test u < 2
                @test p\(p*x) ≈ x
            end
        end
    end
end
