function trimmed_mean(w, d)
    sorted_vector = sort(vec(w))
    h = d ÷ 2
    mean(sorted_vector[1+h : end-h])
end

function trimmed_filter(img::AbstractMatrix{<:Real}, ksize::Integer, d::Integer)
    isodd(ksize) || throw(ArgumentError("ksize must be odd, got $ksize"))
    iseven(d) && 0 ≤ d < ksize^2 || throw(ArgumentError("d must be even in 0:$(ksize^2-1)"))

    out = mapwindow(w -> trimmed_mean(w, d), img, (ksize, ksize))
    return out
end

median_filter(img, k) = trimmed_filter(img, k, k^2 - 1)
