module MachineVisionPipeline

using Images, FileIO, Statistics

include("io.jl")
include("filtering.jl")

export load_images, median_filter, trimmed_filter

end # module
