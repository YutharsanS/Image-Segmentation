using Images, FileIO

const EXTS = (".png", ".jpg", ".jpeg")

"Return the sorted image filenames in `folder` and a Dict of name => grayscale Float64 matrix."
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
