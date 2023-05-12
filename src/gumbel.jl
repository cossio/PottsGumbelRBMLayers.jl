"""
    categorical_sample_from_logits_gumbel(logits)

Like categorical_sample_from_logits, but using the Gumbel trick.
"""
function categorical_sample_from_logits_gumbel(logits::AbstractArray)
    z = logits .+ randgumbel.(eltype(logits))
    idx = dropdims(argmax(z; dims=1); dims=1)
    c = first.(Tuple.(idx))
    return c
end

"""
    randgumbel(T = Float64)

Generates a random Gumbel variate.
"""
randgumbel(::Type{T} = Float64) where {T} = -log(randexp(T))
