### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 5dafaf16-ce2a-11f1-9500-1729d89884be
begin
	import Pkg
	# activate the root project - Parent directory
	Pkg.activate(joinpath(@__DIR__, ".."))

	using Revise
	using MachineVisionPipeline
	using Images, FileIO, PlutoUI
	using Statistics
	using Random
	using Plots
end

# ╔═╡ 29b912c5-050e-428c-b43b-f4d0cb864ee8
begin
	data_dir = normpath(@__DIR__, "..", "data")
	image_files = filter(f -> endswith(f, ".png"), readdir(data_dir))
	@bind selected_file Select(image_files)
end

# ╔═╡ 7150a515-01ea-4d3a-a14c-fbfc203a2796
img = load(joinpath(data_dir, selected_file))

# ╔═╡ 12a81b46-569b-4957-90c0-e7619529ac15
# ╠═╡ skip_as_script = true
#=╠═╡
md"threshold"
  ╠═╡ =#

# ╔═╡ 0395677b-0591-4355-839d-3e77073ada88
@bind threshold Slider(0.0:0.01:1.0, default=0.5, show_value=true)

# ╔═╡ 111e8f88-e036-4a2f-94e3-000f586a208b
md"Blur σ"

# ╔═╡ d5fea232-3c63-4f81-a62a-03f65aefadd3
@bind blur_σ Slider(0.5:0.5:10.0, default= 2.0, show_value=true)

# ╔═╡ bf19544a-c1db-46de-82e9-828ed9e9c4e2
begin
	# Test functions defined in src/MachineVisionPipeline.jl
	g = Float64.(Gray.(img))
	blurred = imfilter(g, Kernel.gaussian(blur_σ))
	mask = Gray.(blurred) .> threshold # produces boolean array : must be converted into a RGB or Gray representation

	# display image
	mosaicview(img, blurred, RGB.(mask); nrow=1)
end

# ╔═╡ a963eff1-0966-47c3-8077-1551f908083b
(eltype(img), eltype(g), img[1, 1], g[1, 1])

# ╔═╡ 548d4505-7a45-4e64-8f6e-a08d9b0f556e
let k = Kernel.gaussian(2.0)
	(axes(k), sum(k), k[0, 0])
end

# ╔═╡ 37215914-55d1-481d-a49b-1a8d2fb1da64
Kernel.gaussian(1.0)

# ╔═╡ c89a83d2-66ba-460d-a3dd-6a01441c190d
begin
	gaussian_filter(a, σ) = imfilter(a, Kernel.gaussian(σ))
	mean_filter(a, k) = imfilter(a, centered(fill(1 / k^2, k, k)))
	median_filter(a, k) = mapwindow(median, a, (k, k))

	# functions that return functions, so that fliters can be chained
	G(σ) = a -> gaussian_filter(a, σ)
	Med(k) = a -> median_filter(a, k)
end

# ╔═╡ 963fe972-6aae-4060-9ea5-d78dd35c0bfa
begin 
	function crop(a, cy, cx; r = 40)
		H, W = size(a)
		a[clamp(cy - r, 1, H):clamp(cy + r, 1, H), clamp(cx - r, 1, W): clamp(cx + r, 1, W)]
	end
	zoom(a, s = 5) = repeat(a, inner= (s, s)) # nearest neighbour enlargement
end

# ╔═╡ 405480e5-e57d-4cd1-9ca6-cfaa8559d4c6
md"""
- k $(@bind k Slider(3:2:11, default=5, show_value=true))
- σ $(@bind σ Slider(0.5:0.5:4.0, default=1.0, show_value=true))
- row $(@bind cy Slider(1:size(g, 1), default=size(g, 1) ÷ 2))
- col $(@bind cx Slider(1:size(g,1), default=size(g, 2) ÷ 2))
"""

# ╔═╡ 8e5abe7f-546f-4550-809d-a836aab6c248
@time begin 
	labels = ["Original", "Gaussian", "Mean", "Median"]
	filtered = (g, gaussian_filter(g, σ), mean_filter(g, k), median_filter(g, k))
	Gray.(clamp01.(mosaicview(zoom.(crop.(filtered, cy, cx))...; nrow = 1, npad = 4, fillvalue = 1)))
end

# ╔═╡ faf0e08e-7c93-42d5-9bca-6832f3a20912
function trimmed_mean(w, d)
	sorted_vector = sort(vec(w))
	h = d ÷ 2
	mean(sorted_vector[1+h : end-h])
end

# ╔═╡ 9ed411c6-f2c4-47ba-b470-8b9a658bcdfc
let t = [0.2 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 0.2]
	(trimmed_mean(t, 0), trimmed_mean(t, 8))   # expect (≈0.289, 0.2)
end

# ╔═╡ 376e9b25-bf3f-4ef7-9c40-689fab3e99d6
md"""
- cutoff $(@bind cutoff Slider(0:1:30, default= 12, show_value=true))
- kernel size $(@bind ksize Slider(3:2:15, default= 5, show_value=true))
"""

# ╔═╡ b64fc882-077c-48c9-a483-2d5685777991
trimmed_image = mapwindow(w -> trimmed_mean(w, cutoff), g, (ksize, ksize))

# ╔═╡ 9f4249d8-dae4-4dfc-83a8-5f1d524e5362
Gray.(clamp01.(mosaicview(trimmed_image)))

# ╔═╡ 73e578a6-af2e-417b-ad25-ffdb9d6cff9c
begin
# synthetic image test
	Random.seed!(42)
	clean = [
		((x - 60) / 35)^2 + ((y - 60) / 22)^2 < 1 ? 0.8 : 0.1
		for y in 1:120, x in 1:120
	]
	mosaicview(Gray.(clean))
end

# ╔═╡ a84c9a2c-6b96-489b-bc51-a087f5493db9
begin
	noisy = clean .+ 0.05 .* randn(size(clean))
	hit = rand(size(noisy)...) .< 0.05
	noisy[hit] .= rand([0.0, 1.0], count(hit))
	for s in 1:5
		noisy[10:9+s, 18s:18s+s-1] .= 1.0
	end
	mse(a, b) = sum(abs2, a .- b) / length(a)
	psnr(a, b) = 10 * log10(1/mse(a, b))
	Gray.(clamp01.(noisy))
end

# ╔═╡ d7c60cb6-e188-4ff4-8980-696756c49fe1
results = [
	"noisy" => noisy,
	"gaussian σ=1" => noisy |> G(1.0),
	"mean 3×3" => mean_filter(noisy, 3),
	"median 3×3" => noisy |> Med(3),
	"median 5×5" => noisy |> Med(5),
	"median -> gaussian" => noisy |> Med(3) |> G(1.0),
	"gaussian -> median" => noisy |> G(1.0) |> Med(3),
	"trimmed 5×5 d=12" => mapwindow(w -> trimmed_mean(w, 12), noisy, (5,5)),
]

# ╔═╡ e49d79a0-3f01-4f0b-827e-2a9a39b308b7
md"""
The Δ PSNR for different methods (higher the better)
"""

# ╔═╡ 1113be22-8397-4eb9-a08d-db139f9e576f
[name => round(psnr(clean, out) - psnr(clean, last(first(results))), digits= 2) for (name, out) in results]

# ╔═╡ 08429b15-72d0-4622-9c53-c5c473f31c95
md"""
The figures corresponds top to bottom in zig-zag way
"""

# ╔═╡ d47173e0-2aea-477c-8f3f-937ee1ee6672
Gray.(clamp01.(mosaicview(last.(results)...; nrow = 2, npad = 3, fillvalue = 1)))

# ╔═╡ 2797308a-166d-4560-ab38-75c430667a01
let r = 60
	plot([noisy[r,: ], gaussian_filter(noisy, 1.5)[r,: ], median_filter(noisy, 5)[r, :], clean[r, :]], label= ["noisy" "gaussian" "median" "clean"], xlabel= "column", ylabel="intensity")
end

# ╔═╡ Cell order:
# ╠═5dafaf16-ce2a-11f1-9500-1729d89884be
# ╠═29b912c5-050e-428c-b43b-f4d0cb864ee8
# ╠═7150a515-01ea-4d3a-a14c-fbfc203a2796
# ╠═12a81b46-569b-4957-90c0-e7619529ac15
# ╠═0395677b-0591-4355-839d-3e77073ada88
# ╠═111e8f88-e036-4a2f-94e3-000f586a208b
# ╠═d5fea232-3c63-4f81-a62a-03f65aefadd3
# ╠═bf19544a-c1db-46de-82e9-828ed9e9c4e2
# ╠═a963eff1-0966-47c3-8077-1551f908083b
# ╠═548d4505-7a45-4e64-8f6e-a08d9b0f556e
# ╠═37215914-55d1-481d-a49b-1a8d2fb1da64
# ╠═c89a83d2-66ba-460d-a3dd-6a01441c190d
# ╠═963fe972-6aae-4060-9ea5-d78dd35c0bfa
# ╠═8e5abe7f-546f-4550-809d-a836aab6c248
# ╠═405480e5-e57d-4cd1-9ca6-cfaa8559d4c6
# ╠═faf0e08e-7c93-42d5-9bca-6832f3a20912
# ╠═9ed411c6-f2c4-47ba-b470-8b9a658bcdfc
# ╠═b64fc882-077c-48c9-a483-2d5685777991
# ╠═9f4249d8-dae4-4dfc-83a8-5f1d524e5362
# ╠═376e9b25-bf3f-4ef7-9c40-689fab3e99d6
# ╠═73e578a6-af2e-417b-ad25-ffdb9d6cff9c
# ╠═a84c9a2c-6b96-489b-bc51-a087f5493db9
# ╠═d7c60cb6-e188-4ff4-8980-696756c49fe1
# ╠═e49d79a0-3f01-4f0b-827e-2a9a39b308b7
# ╠═1113be22-8397-4eb9-a08d-db139f9e576f
# ╠═08429b15-72d0-4622-9c53-c5c473f31c95
# ╠═d47173e0-2aea-477c-8f3f-937ee1ee6672
# ╠═2797308a-166d-4560-ab38-75c430667a01
