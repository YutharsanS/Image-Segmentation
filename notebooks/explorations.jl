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

# ╔═╡ f3d0f210-f2c0-4733-8a61-4cf443cd2366
begin
	imgs = load.(joinpath.(data_dir, image_files))
	mosaicview(imgs...; nrow=1)
end

# ╔═╡ 75e4bccb-2f58-4072-8f31-ac6fcdbdd7bd
md"""
# Filtering - Median Filtering
"""

# ╔═╡ 55c736b6-4c8f-4b2f-9f03-e8ea13069ee1
K = 15 # Kernel size (global)

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

	# functions that return functions, so that filters can be chained
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
- col $(@bind cx Slider(1:size(g,2), default=size(g, 2) ÷ 2))
"""

# ╔═╡ 8e5abe7f-546f-4550-809d-a836aab6c248
@time let
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

# ╔═╡ 5afa1ac5-1ad5-4572-90b6-17f470bf718a
function trimmed_filter(g, ksize, d)
	trimmed_image = mapwindow(w -> trimmed_mean(w, d), g, (ksize, ksize))
end

# ╔═╡ 9ed411c6-f2c4-47ba-b470-8b9a658bcdfc
let t = [0.2 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 0.2]
	(trimmed_mean(t, 0), trimmed_mean(t, 8))   # expect (≈0.289, 0.2)
end

# ╔═╡ 376e9b25-bf3f-4ef7-9c40-689fab3e99d6
md"""
- cutoff $(@bind cutoff Slider(0:2:30, default= 12, show_value=true))
- kernel size $(@bind ksize Slider(3:2:15, default= 5, show_value=true))
"""

# ╔═╡ fd5dfee2-39a2-41fd-9b96-1295b49061d4
Δmedian_value = maximum(abs.(trimmed_filter(g, ksize, cutoff) .- median_filter(g, ksize)))

# ╔═╡ b64fc882-077c-48c9-a483-2d5685777991
let
trimmed_image = Gray.(trimmed_filter(g, ksize, cutoff))
median_image = median_filter(g, ksize)
mosaicview(trimmed_image, median_image; nrow=1)
end

# ╔═╡ f4910f5b-944b-4605-b9bc-10d7e6469e50
md"""
- Δ with median filter :  $(Δmedian_value)
"""

# ╔═╡ 138948d9-ef80-45c0-88c2-51c88cf7f4e7
# noisy image generation
function make_test(σn, p; seed = 42)
	Random.seed!(seed)
	clean = [((x - 60) / 35)^2 + ((y - 60) / 22)^2 < 1 ? 0.8 : 0.1 for y in 1:120, x in 1:120]
	# gaussian noise
	noisy = clean .+ σn .* randn(size(clean))
	# salt and pepper noise 
	hit = rand(size(noisy)...) .< p
	noisy[hit] .= rand([0.0, 1.0], count(hit))
	# stars
	for s in 1:5
		noisy[10:9+s, 18s:18s+s-1] .= 1.0
	end
	return clean, noisy
end

# ╔═╡ c345fa28-8196-448c-b209-bc48af5e0ac6
md"""
- σn $(@bind σn Slider(0.0:0.05:0.3, default=0.05, show_value=true))
- p $(@bind p Slider(0.0:0.05:0.2, default=0.05, show_value=true))
"""

# ╔═╡ 6d12bd57-4cea-4575-982c-59bbadf5e15e
clean, noisy = make_test(σn, p)

# ╔═╡ 73e578a6-af2e-417b-ad25-ffdb9d6cff9c
# synthetic image test
mosaicview(Gray.(clean))

# ╔═╡ a84c9a2c-6b96-489b-bc51-a087f5493db9
begin
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
	"trimmed 5×5 d=12" => trimmed_filter(noisy, 5, 12),
]

# ╔═╡ e49d79a0-3f01-4f0b-827e-2a9a39b308b7
md"""
The Δ PSNR for different methods (higher is better)
"""

# ╔═╡ 1113be22-8397-4eb9-a08d-db139f9e576f
psnr_values_controlled = [name => round(psnr(clean, out) - psnr(clean, last(first(results))), digits= 2) for (name, out) in results]

# ╔═╡ 08429b15-72d0-4622-9c53-c5c473f31c95
md"""
The figures correspond top to bottom in zig-zag order
"""

# ╔═╡ d47173e0-2aea-477c-8f3f-937ee1ee6672
Gray.(clamp01.(mosaicview(last.(results)...; nrow = 1, npad = 3, fillvalue = 1)))

# ╔═╡ 2797308a-166d-4560-ab38-75c430667a01
let r = 60
	plot([noisy[r,: ], gaussian_filter(noisy, 1.5)[r,: ], median_filter(noisy, 5)[r, :], clean[r, :]], label= ["noisy" "gaussian" "median" "clean"], xlabel= "column", ylabel="intensity")
end

# ╔═╡ 8f9e7533-f50a-4d1f-9148-13d79d471d27
# step through different d values
let
	base = psnr(clean, noisy)
	[
		d => round(psnr(clean, trimmed_filter(noisy, 5, d)) - base, digits=2)
		for d in 0:4:24
	]
end

# ╔═╡ 90a6d830-e5c5-4ad4-bf92-07ed61054f64
psnr_values_controlled

# ╔═╡ 833568c1-ffb8-4aba-8d0d-99658b638ae9
Gray.(clamp01.(mosaicview([zoom(out[6:18, 14:110], 4) for out in last.(results)]...;
                          ncol = 1, npad = 2, fillvalue = 1)))

# ╔═╡ 59dca33c-874a-4c96-9aff-5fc1f1c447a7
md"""
# Segementation
"""

# ╔═╡ e119d917-8156-4654-b265-f905622d18fa
mosaicview(Gray.(g))

# ╔═╡ 6d88a013-dd93-4209-99af-e64c83a4c83a
md"""
- σseg $(@bind σseg Slider(0.5:0.1:4.0, default=1.0, show_value=true))
- kseg $(@bind kseg Slider(3:2:21, default=5, show_value=true))
"""

# ╔═╡ 458121a4-af60-4a29-9aa2-cff024749d8b
filtered = median_filter(g, kseg)

# ╔═╡ d3f302d4-6569-4835-8457-3ee20c5d3d7c
let
	p1 = histogram(vec(g), bins = 100, yscale = :log10, title = "raw", legend = false)
	p2 = histogram(vec(filtered), bins = 100, yscale = :log10, title = "median $(kseg)×$(kseg)", legend = false)
	plot(p1, p2, layout = (1, 2), size = (800, 300))
end

# ╔═╡ ae2907c2-806a-4940-8f58-72c44316c1db
function intermeans_threshold(f; T0 = mean(f), ϵ= 1e-4, maxiter=100)
	T = T0
	history = [T]
	for _ in 1:maxiter
		μ1 = mean(f[f .≤ T])
		μ2 = mean(f[f .> T])
		T_new = (μ1 + μ2) / 2 

		if abs(T_new - T) < ϵ
			break
		end
		T = T_new
		push!(history, T)
	end
	return T, history
end

# ╔═╡ 9e1b13bc-2d08-40fc-831e-64385f0e1775
function background_threshold(f; k = 5, border = 10, σmin = 0.01)
	H, W = size(f)

	# collect border pixels
	b = vcat(vec(f[1: border, :]), vec(f[H-border+1:H, :]), vec(f[:, 1:border]), vec(f[:, W-border+1:border]))
	μ = median(b)
	σ = 1.4826 * median(abs.(b .- μ))
	return μ + k * max(σ, σmin), μ, σ
end

# ╔═╡ a8781fa6-8059-4f07-bc5a-82fe3625e376
# convergence over multiple values
[T0 => round.(intermeans_threshold(filtered; T0=T0)[2], digits=4) for T0 in (0.1, 0.5, 0.9)]

# ╔═╡ 09a8a6f6-20ba-4466-bb2f-46d1b98fb94d
function largest_component(labels)
	maximum(labels) == 0 && return falses(size(labels))
	sizes = component_lengths(labels)
	return labels .== argmax(sizes[1:end])
end

# ╔═╡ 9551cc52-203c-4efb-b470-5936b2b1bda2
[file => let g = Float64.(Gray.(load(joinpath(data_dir, file))))
            f = median_filter(g, K)
            T, h = intermeans_threshold(f)
            labels = label_components(f .> T)
            (T = round(T, digits = 3), iters = length(h) - 1,
             ncomp = maximum(labels), area = count(largest_component(labels)))
        end
 for file in image_files]

# ╔═╡ d9a5f34f-e1a8-41c2-92d0-9df40346c0db
md"k $(@bind kσ Slider(2:1:15, default = 5, show_value = true))"

# ╔═╡ 2ab11992-2119-44d6-90f9-a097d6115917
let
	Tim, _ = intermeans_threshold(filtered)
	Tbg, μ, σ = background_threshold(filtered; k = kσ)
	a = largest_component(label_components(filtered .> Tim))
	b = largest_component(label_components(filtered .> Tbg))
	md"""
	inter-means T = $(round(Tim, digits = 3)) · background T = $(round(Tbg, digits = 3)) (μ = $(round(μ, digits = 3)), σ = $(round(σ, digits = 3)))

	$(mosaicview(Gray.(a), Gray.(b); nrow = 1, npad = 6, fillvalue = 1))
	"""
end

# ╔═╡ 85ea43ef-391e-4974-9cb9-057823f2cde1
area_curve(f, Ts) = [count(largest_component(label_components(f .> T))) for T in Ts]

# ╔═╡ 62587e2f-e4f3-46ae-b194-a6d3fca31266
let
	Ts = 0.02:0.01:0.90
	A = @time area_curve(filtered, Ts)
	Tim, _ = intermeans_threshold(filtered)
	plot(Ts, A, yscale = :log10, marker = :circle, ms = 2, legend = false,
	     xlabel = "threshold T", ylabel = "largest-component area")
	vline!([Tim])
end

# ╔═╡ 62dc8977-4643-4316-a4fb-b3042acc0eb2
function segment_threshold(f; σ_cut = 0.05, k = 5)
	Tbg, μ, σ = background_threshold(f; k = k)
	σ < σ_cut && return Tbg, :background
	Tim, _ = intermeans_threshold(f)
	return Tim, :intermeans
end

# ╔═╡ cfa29cf0-7962-477e-a24b-0f579dd91acd
# T, hist_T = intermeans_threshold(filtered)
T, _ = segment_threshold(filtered)

# ╔═╡ 1e5d0cc0-6b47-40a2-9059-07df3e5c1da2
let
	hist = histogram(vec(filtered); bins=100, label="median filter K = $kseg", alpha=0.5, xlabel="Intensity", ylabel="Pixel count", title="Intensity histogram")
	histogram!(hist, vec(g); label="original", alpha=0.5)
	# histogram!(hist, vec(gaussian_filter(g, σseg)); label="gaussian", alpha=0.5)
	vline!([T], lw=2, label="Threshold")
	
end

# ╔═╡ 26b3d671-8171-43dc-9463-b1aac2eb63ab
# if  more than half the boarder pixels are white, then the background is white : assumption
# we assume the object doesn't touch the boarders and each image only features one object

begin
	mask_seg = filtered .> T
	boarders_seg = vcat(mask_seg[1, :], mask_seg[end, :], mask_seg[:, 1], mask_seg[:, end])
	if (count(boarders_seg)[1] > (length(boarders_seg) / 2) )
		mask_seg = .!mask_seg # invert the mask
	end
end

# ╔═╡ 8b971093-7bfb-45b8-99e7-d6d83e9eb927
mosaicview(filtered, Gray.(mask_seg); nrow=1)

# ╔═╡ b09e2368-d666-40ba-ba60-53d07cdff0db
labels = label_components(mask_seg, strel_box((3, 3))) # label connected components C8

# ╔═╡ bbc43d93-740e-47d3-a25b-cb087fe4442c
obj = largest_component(labels)

# ╔═╡ 145c34ea-d02d-401e-85d6-8b3f03830170
let
	tint = RGB.(max.(g, 0.7 .* obj), g, g)
	mosaicview(RGB.(Gray.(mask_seg)), RGB.(Gray.(obj)), tint; nrow=1, npad=6, fillvalue=1)
end

# ╔═╡ f7349d08-ccb7-4791-86c9-664363b19966
md"""
# Countour Tracking
"""

# ╔═╡ 6c8a2a0c-78a0-4dcb-99e9-d423b10466e0
const DIRS = (CartesianIndex(-1, 1),  CartesianIndex(0, 1),   # 0 NE, 1 E
              CartesianIndex(1, 1),   CartesianIndex(1, 0),   # 2 SE, 3 S
              CartesianIndex(1, -1),  CartesianIndex(0, -1),  # 4 SW, 5 W
              CartesianIndex(-1, -1), CartesianIndex(-1, 0))  # 6 NW, 7 N

# ╔═╡ 61073048-6b9a-4669-837e-9b396f23c850
function next_pixel(obj, cur, n)
	m = mod(n - 2, 8)
	for _ in 1:8
		q = cur + DIRS[m + 1]
		checkbounds(Bool, obj, q) && obj[q] && return q, m
		m = mod(m + 1, 8)
	end
	return nothing # isolated pixel
end

# ╔═╡ e320ad53-42a2-4459-83cd-9ff968117bba
function trace_boundary(obj)
	start = findfirst(obj)
	start === nothing && throw(ArgumentError("mask is empty"))
	path = [start]
	cur, n, second = start, 1, nothing

	for _ in 1: 4*length(obj)
		nxt = next_pixel(obj, cur, n)
		nxt === nothing && return path
		q, m = nxt

		cur == start && q == second && (pop!(path); return path)
		second === nothing && (second = q)

		push!(path, q)
		cur, n = q, m
	end
	error("tracing did not terminate")
end

# ╔═╡ 0da36c94-d422-4584-bddc-5db1d5019ce0
let
sq = falses(5, 5); sq[2:4, 2:4] .= true
next_pixel(sq, CartesianIndex(2, 2), 1)    # expect (CartesianIndex(2, 3), 1): one step east
trace_boundary(sq)
end

# ╔═╡ 83a21a74-9b67-4e53-9be0-d0d87d5c36ff
lobes = Bool[0 0 0 0 0 0 0 0 0 0
             0 0 0 0 1 1 1 0 0 0
             0 0 0 0 1 1 1 0 0 0
             0 0 0 0 1 1 1 0 0 0
             0 0 0 1 1 1 1 0 0 0
             0 0 1 0 0 0 0 0 0 0
             0 0 1 1 1 1 0 0 0 0
             0 0 1 1 1 1 0 0 0 0
             0 0 1 1 1 1 0 0 0 0
             0 0 0 1 1 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0]

# ╔═╡ a0696c6c-40bd-4777-ac96-875a8a39632c
let p = trace_boundary(lobes)
	(length(p), unique(p))
end

# ╔═╡ 0b8edfb8-2a71-499d-95eb-26e58162ac5c
begin
	const N4 = (CartesianIndex(-1, 0), CartesianIndex(1, 0), CartesianIndex(0, -1), CartesianIndex(0, 1))
	# a boundary pixel has at least one background (or off-image) 4-neighbour
	on_edge(obj, p) = any(d -> !checkbounds(Bool, obj, p + d) || !obj[p + d], N4)
	# the path is closed: its last pixel is next to its first
	is_closed(path) = maximum(abs, Tuple(path[end] - path[1])) <= 1
end

# ╔═╡ eefab908-0593-44f2-a63f-d0b48a930515
let path = trace_boundary(obj)
	(length = length(path), all_on_edge = all(p -> on_edge(obj, p), path), closed = is_closed(path))
end

# ╔═╡ 0d458d93-8e0a-4a36-9d19-5712d6837b0c
let out = RGB.(g), path = trace_boundary(obj)
	for p in path
		out[p] = RGB(1, 0, 0)
	end
	out[first(path)] = RGB(0, 1, 0)          # start pixel in green
	out
end

# ╔═╡ Cell order:
# ╠═5dafaf16-ce2a-11f1-9500-1729d89884be
# ╠═f3d0f210-f2c0-4733-8a61-4cf443cd2366
# ╠═29b912c5-050e-428c-b43b-f4d0cb864ee8
# ╠═75e4bccb-2f58-4072-8f31-ac6fcdbdd7bd
# ╠═55c736b6-4c8f-4b2f-9f03-e8ea13069ee1
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
# ╠═5afa1ac5-1ad5-4572-90b6-17f470bf718a
# ╠═9ed411c6-f2c4-47ba-b470-8b9a658bcdfc
# ╠═fd5dfee2-39a2-41fd-9b96-1295b49061d4
# ╠═b64fc882-077c-48c9-a483-2d5685777991
# ╠═376e9b25-bf3f-4ef7-9c40-689fab3e99d6
# ╠═f4910f5b-944b-4605-b9bc-10d7e6469e50
# ╠═138948d9-ef80-45c0-88c2-51c88cf7f4e7
# ╠═6d12bd57-4cea-4575-982c-59bbadf5e15e
# ╠═c345fa28-8196-448c-b209-bc48af5e0ac6
# ╠═73e578a6-af2e-417b-ad25-ffdb9d6cff9c
# ╠═a84c9a2c-6b96-489b-bc51-a087f5493db9
# ╠═d7c60cb6-e188-4ff4-8980-696756c49fe1
# ╠═e49d79a0-3f01-4f0b-827e-2a9a39b308b7
# ╠═1113be22-8397-4eb9-a08d-db139f9e576f
# ╠═08429b15-72d0-4622-9c53-c5c473f31c95
# ╠═d47173e0-2aea-477c-8f3f-937ee1ee6672
# ╠═2797308a-166d-4560-ab38-75c430667a01
# ╠═8f9e7533-f50a-4d1f-9148-13d79d471d27
# ╠═90a6d830-e5c5-4ad4-bf92-07ed61054f64
# ╠═833568c1-ffb8-4aba-8d0d-99658b638ae9
# ╠═59dca33c-874a-4c96-9aff-5fc1f1c447a7
# ╠═e119d917-8156-4654-b265-f905622d18fa
# ╠═458121a4-af60-4a29-9aa2-cff024749d8b
# ╠═1e5d0cc0-6b47-40a2-9059-07df3e5c1da2
# ╠═6d88a013-dd93-4209-99af-e64c83a4c83a
# ╠═d3f302d4-6569-4835-8457-3ee20c5d3d7c
# ╠═ae2907c2-806a-4940-8f58-72c44316c1db
# ╠═9e1b13bc-2d08-40fc-831e-64385f0e1775
# ╠═cfa29cf0-7962-477e-a24b-0f579dd91acd
# ╠═a8781fa6-8059-4f07-bc5a-82fe3625e376
# ╠═8b971093-7bfb-45b8-99e7-d6d83e9eb927
# ╠═26b3d671-8171-43dc-9463-b1aac2eb63ab
# ╠═b09e2368-d666-40ba-ba60-53d07cdff0db
# ╠═09a8a6f6-20ba-4466-bb2f-46d1b98fb94d
# ╠═bbc43d93-740e-47d3-a25b-cb087fe4442c
# ╠═145c34ea-d02d-401e-85d6-8b3f03830170
# ╠═9551cc52-203c-4efb-b470-5936b2b1bda2
# ╠═d9a5f34f-e1a8-41c2-92d0-9df40346c0db
# ╠═2ab11992-2119-44d6-90f9-a097d6115917
# ╠═85ea43ef-391e-4974-9cb9-057823f2cde1
# ╠═62587e2f-e4f3-46ae-b194-a6d3fca31266
# ╠═62dc8977-4643-4316-a4fb-b3042acc0eb2
# ╠═f7349d08-ccb7-4791-86c9-664363b19966
# ╠═6c8a2a0c-78a0-4dcb-99e9-d423b10466e0
# ╠═61073048-6b9a-4669-837e-9b396f23c850
# ╠═e320ad53-42a2-4459-83cd-9ff968117bba
# ╠═0da36c94-d422-4584-bddc-5db1d5019ce0
# ╠═83a21a74-9b67-4e53-9be0-d0d87d5c36ff
# ╠═a0696c6c-40bd-4777-ac96-875a8a39632c
# ╠═0b8edfb8-2a71-499d-95eb-26e58162ac5c
# ╠═eefab908-0593-44f2-a63f-d0b48a930515
# ╠═0d458d93-8e0a-4a36-9d19-5712d6837b0c
