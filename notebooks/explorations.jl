### A Pluto.jl notebook ###
# v1.0.3

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
end

# ╔═╡ 29b912c5-050e-428c-b43b-f4d0cb864ee8
begin
	data_dir = normpath(@__DIR__, "..", "data")
	image_files = filter(f -> endswith(f, ".png"), readdir(data_dir))
	@bind selected_file Select(image_files)
end

# ╔═╡ 7150a515-01ea-4d3a-a14c-fbfc203a2796
img = load(joinpath(data_dir, selected_file))

# ╔═╡ 0395677b-0591-4355-839d-3e77073ada88
@bind threshold Slider(0.0:0.01:1.0, default=0.5, show_value=true)

# ╔═╡ 12a81b46-569b-4957-90c0-e7619529ac15
# ╠═╡ skip_as_script = true
#=╠═╡
threshold
  ╠═╡ =#

# ╔═╡ d5fea232-3c63-4f81-a62a-03f65aefadd3
@bind blur_σ Slider(0.5:0.5:10.0, default= 2.0, show_value=true)

# ╔═╡ b0d94450-5d36-45aa-ae3b-38dc98f5da3b
blur_σ

# ╔═╡ bf19544a-c1db-46de-82e9-828ed9e9c4e2
begin
	# Test functions defined in src/MachineVisionPipeline.jl
	blurred = imfilter(img, Kernel.gaussian(blur_σ))
	mask = Gray.(blurred) .> threshold # produces boolean array : must be converted into a RGB or Gray representation

	# display image
	mosaicview(img, blurred, RGB.(mask); nrow=1)
end

# ╔═╡ Cell order:
# ╠═5dafaf16-ce2a-11f1-9500-1729d89884be
# ╠═29b912c5-050e-428c-b43b-f4d0cb864ee8
# ╠═7150a515-01ea-4d3a-a14c-fbfc203a2796
# ╠═12a81b46-569b-4957-90c0-e7619529ac15
# ╠═0395677b-0591-4355-839d-3e77073ada88
# ╠═b0d94450-5d36-45aa-ae3b-38dc98f5da3b
# ╠═d5fea232-3c63-4f81-a62a-03f65aefadd3
# ╠═bf19544a-c1db-46de-82e9-828ed9e9c4e2
