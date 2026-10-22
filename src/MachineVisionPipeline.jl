module MachineVisionPipeline

using Images, FileIO

const EXTS = (".png", ".jpg", ".jpeg")

"Returns sorted image filenames in `data` and Dict of name => grayscale Float 64 matrix"
function load_images(folder::AbstractString)
    isdir(folder) || error("Not a folder: $folder")
    files = sort(filter(f -> lowercase(splitext(f)[2]) in EXTS, readdir(folder)))
    imgs = Dict{String, Matrix{Float64}}()

    for f in files
        img = load(joinpath(folder, f))
        imgs[f] = Float64.(Gray.(img))
    end

    return files, imgs
end

function main(args)
    length(args) == 1 || error("usage: julia MachineVisionPipeline.jl data")
    files, imgs = load_images(args[1])
    for f in files
        println(rpad(f, 25), " size=", size(imgs[f]), " min/max=", extrema(imgs[f]))
    end
end

end # module

# Run main only when executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    MachineVisionPipeline.main(ARGS)
end