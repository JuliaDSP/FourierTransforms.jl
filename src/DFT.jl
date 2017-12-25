# This file is a part of Julia. License is MIT: http://julialang.org/license
__precompile__()

module DFT

# DFT plan where the inputs are an array of eltype T
abstract type Plan{T} end

import Base: show, summary, size, ndims, length, eltype,
             *, A_mul_B!, inv, \, A_ldiv_B!

eltype(::Plan{T}) where T = T
eltype(T::Type{<:Plan}) = T.parameters[1]

# size(p) should return the size of the input array for p
size(p::Plan, d) = size(p)[d]
ndims(p::Plan) = length(size(p))
length(p::Plan) = prod(size(p))::Int

##############################################################################
export fft, ifft, bfft, fft!, ifft!, bfft!,
       plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
       rfft, irfft, brfft, plan_rfft, plan_irfft, plan_brfft

complexfloat(x::AbstractArray{Complex{<:AbstractFloat}}) = x

# return an Array, rather than similar(x), to avoid an extra copy for FFTW
# (which only works on StridedArray types).
@static if VERSION <= v"0.7.0-DEV.3180"
    complexfloat(x::AbstractArray{T}) where T<:Complex = copy!(Array{typeof(float(one(T)))}(size(x)), x)
    complexfloat(x::AbstractArray{T}) where T<:AbstractFloat = copy!(Array{typeof(complex(one(T)))}(size(x)), x)
    complexfloat(x::AbstractArray{T}) where T<:Real = copy!(Array{typeof(complex(float(one(T))))}(size(x)), x)
else
    complexfloat(x::AbstractArray{T}) where T<:Complex = copyto!(Array{typeof(float(one(T)))}(uninitialized, size(x)), x)
    complexfloat(x::AbstractArray{T}) where T<:AbstractFloat = copyto!(Array{typeof(complex(one(T)))}(uninitialized, size(x)), x)
    complexfloat(x::AbstractArray{T}) where T<:Real = copyto!(Array{typeof(complex(float(one(T))))}(uninitialized, size(x)), x)
end

# implementations only need to provide plan_X(x, region)
# for X in (:fft, :bfft, ...):
for f in (:fft, :bfft, :ifft, :fft!, :bfft!, :ifft!, :rfft)
    pf = Symbol(:plan_, f)
    @eval begin
        $f(x::AbstractArray) = $pf(x) * x
        $f(x::AbstractArray, region) = $pf(x, region) * x
        $pf(x::AbstractArray; kws...) = $pf(x, 1:ndims(x); kws...)
    end
end

# promote to a complex floating-point type (out-of-place only),
# so implementations only need Complex{Float} methods
for f in (:fft, :bfft, :ifft)
    pf = Symbol(:plan_, f)
    @eval begin
        $f(x::AbstractArray{<:Real}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::AbstractArray{<:Real}, region; kws...) = $pf(complexfloat(x), region; kws...)
        $f(x::AbstractArray{Complex{<:Union{Integer,Rational}}}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::AbstractArray{Complex{<:Union{Integer,Rational}}}, region; kws...) = $pf(complexfloat(x), region; kws...)
    end
end
rfft(x::AbstractArray{<:Union{Integer,Rational}}, region=1:ndims(x)) = rfft(float(x), region)
plan_rfft(x::AbstractArray{<:Union{Integer,Rational}}, region; kws...) = plan_rfft(float(x), region; kws...)

# only require implementation to provide *(::Plan{T}, ::Array{T})
@static if VERSION <= v"0.7.0-DEV.3180"
    *(p::Plan{T}, x::AbstractArray) where T = p * copy!(Array{T}(size(x)), x)
else
    *(p::Plan{T}, x::AbstractArray) where T = p * copy!(Array{T}(uninitialized, size(x)), x)
end

# Implementations should also implement A_mul_B!(Y, plan, X) so as to support
# pre-allocated output arrays.  We don't define * in terms of A_mul_B!
# generically here, however, because of subtleties for in-place and rfft plans.

##############################################################################
# To support inv, \, and A_ldiv_B!(y, p, x), we require Plan subtypes
# to have a pinv::Plan field, which caches the inverse plan, and which
# should be initially undefined.  They should also implement
# plan_inv(p) to construct the inverse of a plan p.

# hack from @simonster (in #6193) to compute the return type of plan_inv
# without actually calling it or even constructing the empty arrays.
_pinv_type(p::Plan) = typeof([plan_inv(x) for x in typeof(p)[]])
pinv_type(p::Plan) = eltype(_pinv_type(p))

inv(p::Plan) =
    isdefined(p, :pinv) ? p.pinv::pinv_type(p) : (p.pinv = plan_inv(p))
\(p::Plan, x::AbstractArray) = inv(p) * x
A_ldiv_B!(y::AbstractArray, p::Plan, x::AbstractArray) = A_mul_B!(y, inv(p), x)

##############################################################################
# implementations only need to provide the unnormalized backwards FFT,
# similar to FFTW, and we do the scaling generically to get the ifft:

mutable struct ScaledPlan{T,P,N} <: Plan{T}
    p::P
    scale::N # not T, to avoid unnecessary promotion to Complex
    pinv::Plan
    ScaledPlan{T,P,N}(p, scale) where {T,P,N} = new(p, scale)
end
ScaledPlan(p::P, scale::N) where {P<:Plan,N<:Number} = ScaledPlan{eltype(P),P,N}(p, scale)
ScaledPlan(p::ScaledPlan, α::Number) = ScaledPlan(p.p, p.scale * α)

size(p::ScaledPlan) = size(p.p)

show(io::IO, p::ScaledPlan) = print(io, p.scale, " * ", p.p)
summary(p::ScaledPlan) = string(p.scale, " * ", summary(p.p))

*(p::ScaledPlan, x::AbstractArray) = scale!(p.p * x, p.scale)

*(α::Number, p::Plan) = ScaledPlan(p, α)
*(p::Plan, α::Number) = ScaledPlan(p, α)
*(I::UniformScaling, p::ScaledPlan) = ScaledPlan(p, I.λ)
*(p::ScaledPlan, I::UniformScaling) = ScaledPlan(p, I.λ)
*(I::UniformScaling, p::Plan) = ScaledPlan(p, I.λ)
*(p::Plan, I::UniformScaling) = ScaledPlan(p, I.λ)

# Normalization for ifft, given unscaled bfft, is 1/prod(dimensions)
normalization(T, sz, region) = one(T) / prod([sz...][[region...]])
normalization(X, region) = normalization(real(eltype(X)), size(X), region)

plan_ifft(x::AbstractArray, region; kws...) =
    ScaledPlan(plan_bfft(x, region; kws...), normalization(x, region))
plan_ifft!(x::AbstractArray, region; kws...) =
    ScaledPlan(plan_bfft!(x, region; kws...), normalization(x, region))

plan_inv(p::ScaledPlan) = ScaledPlan(plan_inv(p.p), inv(p.scale))

A_mul_B!(y::AbstractArray, p::ScaledPlan, x::AbstractArray) =
    scale!(p.scale, A_mul_B!(y, p.p, x))

##############################################################################
# Real-input DFTs are annoying because the output has a different size
# than the input if we want to gain the full factor-of-two(ish) savings
# For backward real-data transforms, we must specify the original length
# of the first dimension, since there is no reliable way to detect this
# from the data (we can't detect whether the dimension was originally even
# or odd).

for f in (:brfft, :irfft)
    pf = Symbol(:plan_, f)
    @eval begin
        $f(x::AbstractArray, d::Integer) = $pf(x, d) * x
        $f(x::AbstractArray, d::Integer, region) = $pf(x, d, region) * x
        $pf(x::AbstractArray, d::Integer;kws...) = $pf(x, d, 1:ndims(x);kws...)
    end
end

for f in (:brfft, :irfft)
    @eval begin
        $f(x::AbstractArray{<:Real}, d::Integer, region=1:ndims(x)) = $f(complexfloat(x), d, region)
        $f(x::AbstractArray{Complex{<:Union{Integer,Rational}}}, d::Integer, region=1:ndims(x)) = $f(complexfloat(x), d, region)
    end
end

function rfft_output_size(x::AbstractArray, region)
    d1 = first(region)
    osize = [size(x)...]
    osize[d1] = osize[d1]>>1 + 1
    return osize
end

function brfft_output_size(x::AbstractArray, d::Integer, region)
    d1 = first(region)
    osize = [size(x)...]
    @assert osize[d1] == d>>1 + 1
    osize[d1] = d
    return osize
end

plan_irfft(x::AbstractArray{Complex{T}}, d::Integer, region; kws...) where T =
    ScaledPlan(plan_brfft(x, d, region; kws...),
               normalization(T, brfft_output_size(x, d, region), region))

##############################################################################

export fftshift, ifftshift

fftshift(x) = circshift(x, div([size(x)...],2))

function fftshift(x,dim)
    s = zeros(Int,ndims(x))
    s[dim] = div(size(x,dim),2)
    circshift(x, s)
end

ifftshift(x) = circshift(x, div([size(x)...],-2))

function ifftshift(x,dim)
    s = zeros(Int,ndims(x))
    s[dim] = -div(size(x,dim),2)
    circshift(x, s)
end

##############################################################################
# Native Julia FFTs:

include("ctfft.jl")
include("fftn.jl")

##############################################################################

end
