using FourierTransforms
using OffsetArrays
using Test

include("util/symbolic.jl")

using FourierTransforms: @nontwiddle, @twiddle, CTPlan
FourierTransforms.ωpow(::Type{SymbolicExpr}, n, i) = RootOfUnity(n)^i

@eval @nontwiddle($SymbolicExpr, true, 4)
@eval @nontwiddle($SymbolicExpr, true, 2)
@eval @twiddle($SymbolicExpr, true, 4)
@eval @twiddle($SymbolicExpr, true, 2)

# Input length 8
let inputs = [Var(Symbol(string('x', subscripts[i]))) for i = 0:7]
    plan = CTPlan(SymbolicExpr, true, 8)
    y = plan * inputs
    y = map(simplify, map(expand, y))

    naive = map(simplify, [sum(inputs[i+1]*RootOfUnity(8)^(i*j) for i = 0:7) for j = 0:7])

    # Test that the fft is symbolically equal to the naive expression.
    @test all(map(simplify, map(expand, map(-, y, naive)))) do x
        x == Plus(())
    end
end

# Input length 7 (excercising bluestein fft)
# TODO: This requires more support than the SymbolicExpr type currently has
#=
let inputs = [Var(Symbol(string('x', subscripts[i]))) for i = 0:5]
    plan = CTPlan(SymbolicExpr, true, length(inputs))
    y = plan * inputs
    y = map(simplify, map(expand, y))

    naive = map(simplify, [sum(inputs[i+1]*RootOfUnity(length(inputs))^(i*j) for i = 0:length(inputs)-1) for j = 0:length(inputs)-1])

    # Test that the fft is symbolically equal to the naive expression.
    @test all(map(simplify, map(expand, map(-, y, naive)))) do x
        x == Plus(())
    end
end
=#
