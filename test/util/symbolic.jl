using AutoHashEquals
abstract type SymbolicExpr end
import Base: show

struct One <: SymbolicExpr; end
show(io::IO, o::One) = print(io, 1)

struct NegOne <: SymbolicExpr; end
show(io::IO, o::NegOne) = print(io, -1)
Base.:-(a::SymbolicExpr) = *(NegOne(), a)
Base.:-(a::SymbolicExpr, b::SymbolicExpr) = +(a,-(b))

struct Var <: SymbolicExpr
    name::Symbol
end
show(io::IO, v::Var) = print(io, v.name)

const superscripts = OffsetArray(['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹'], 0:9)
struct Pow{T<:SymbolicExpr} <: SymbolicExpr
    s::T
    n::Int
    Pow(p::Pow, n::Integer) = new{typeof(p.s)}(p.s, p.n*n)
    Pow(s, n::Integer) = new{typeof(s)}(s, n)
end
Base.:^(s::SymbolicExpr, n::Integer) = Pow(s, n)
show(io::IO, p::Pow) = print(io, p.s, reverse(map(x->superscripts[x], digits(p.n)))...)

with_parens(x, T) = isa(x, T) ? sprint(print, "(", x, ")") : sprint(print, x)

@auto_hash_equals struct Plus <: SymbolicExpr
    x::NTuple{N, SymbolicExpr} where N
    function Plus(x::NTuple{N, SymbolicExpr}) where {N}
        es = SymbolicExpr[]
        for e in x
            isa(e, Plus) ? append!(es, e.x) : push!(es, e)
        end
        new((es...,))
    end
end
Plus(x::NTuple{1, SymbolicExpr}) = x[1]
function show(io::IO, p::Plus)
    length(p.x) == 0 && return print(io, "0")
    print(io, p.x[1])
    for e in p.x[2:end]
        if isa(e, Times) && e.x[1] == NegOne()
            print(io, " - ")
            print(io, with_parens(Times(e.x[2:end]), Plus))
        else
            print(io, " + ")
            print(io, with_parens(e, Plus))
        end
    end
end
Base.:+(x::SymbolicExpr...) = Plus(x)
Base.zero(::Type{SymbolicExpr}) = Plus(())

@auto_hash_equals struct Times <: SymbolicExpr
    x::NTuple{N, SymbolicExpr} where {N}
    function Times(x::NTuple{N, SymbolicExpr}) where {N}
        es = SymbolicExpr[]
        parity = One()
        function process!(x)
            for e in x
                isa(e, One) && continue
                if isa(e, NegOne)
                    parity = (parity == One() ? NegOne() : One())
                    continue
                end
                isa(e, Times) ? process!(e.x) : push!(es, e)
            end
        end
        process!(x)
        parity == One() || pushfirst!(es, NegOne())
        new((es...,))
    end
end
Times(x::NTuple{1, SymbolicExpr}) = x[1]
show(io::IO, p::Times) = join(io, map(x->with_parens(x, Plus), p.x), " * ")
Base.:*(x::SymbolicExpr...) = Times(x)

const subscripts = OffsetArray(['₀' + i for i = 0:9], 0:9)
struct RootOfUnity <: SymbolicExpr
    n
end

function simplify(p::Pow{RootOfUnity})
    (p.n == 0 || p.n == p.s.n) && return One()
    p.n == 1 && return p.s
    p.n > p.s.n && return simplify(Pow(p.s, mod(p.n, p.s.n)))
    iseven(p.s.n) && p.n >= div(p.s.n, 2) &&
        return simplify(Times((NegOne(), Pow(p.s, mod(p.n, div(p.s.n, 2))))))
    return p
end

function Base.:(==)(a::Union{RootOfUnity, Pow{RootOfUnity}},
         b::Union{RootOfUnity, Pow{RootOfUnity}})
    base = lcm((isa(x, RootOfUnity) ? x.n : x.s.n for x in (a,b))...)
    _d(r::RootOfUnity) = div(base, r.n)
    _d(p::Pow{RootOfUnity}) = _d(p.s) * p.n
    _d(a) == _d(b)
end

function accum(itr)
    base = lcm((isa(x, RootOfUnity) ? x.n : x.s.n for x in itr)...)
    _d(r::RootOfUnity) = div(base, r.n)
    _d(p::Pow{RootOfUnity}) = _d(p.s) * p.n
    n = sum(_d, itr)
    simplify(Pow(RootOfUnity(base), n))
end

function simplify(t::Times)
    t = Times(map(simplify, t.x))
    isa(t, Times) || return t
    for (i,e) in pairs(t.x)
        isa(e, One) && return simplify(Times(t.x[filter(j->j != i, 1:length(t.x))]))
    end
    roots = filter(x->isa(x, Union{RootOfUnity, Pow{RootOfUnity}}), collect(t.x))
    if length(roots) > 1
        others = filter(x->!isa(x, Union{RootOfUnity, Pow{RootOfUnity}}), collect(t.x))
        return Times((others..., accum(roots)))
    end
    t
end

find_var(x) = nothing
find_var(x::Var) = x.name
find_var(t::Times) = something(map(find_var, t.x)...)
find_var(t::Plus) = something(map(find_var, t.x)...)

cancels(a::Times, b) = isa(a.x[1], NegOne) && Times(a.x[2:end]) == b

function simplify(t::Plus)
    t = Plus(map(simplify, t.x))
    # Sort the terms by variable name
    t = Plus(tuple(sort(collect(t.x), by=find_var)...))
    # See if adjacent terms cancel each other out
    for i = 1:length(t.x)-1
        term = t.x[i]
        next_term = t.x[i+1]
        if (isa(term, Times) && cancels(term, next_term)) ||
            (isa(next_term, Times) && cancels(next_term, term))
            return simplify(Plus(t.x[filter(j->!in(j, (i, i+1)), 1:length(t.x))]))
        end
    end
    return t
end
simplify(x) = x

expand(x) = x
expand(p::Plus) = Plus(map(expand, p.x))
function expand(t::Times)
    for (i,e) in pairs(t.x)
        if isa(e, Plus)
            other_factors = Times(t.x[filter(j->j != i, 1:length(t.x))])
            return Plus(((expand(Times((pe, other_factors))) for pe in e.x)...,))
        end
    end
    return t
end

show(io::IO, ω::RootOfUnity) = print(io, string("ω", subscripts[ω.n]))
