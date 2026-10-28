using Test, MachineVisionPipeline
using Images, Statistics

@testset "filtering" begin
    t = fill(0.2, 5, 5); t[3, 3] = 1.0
    @test median_filter(t, 3)[3, 3] == 0.2
    @test trimmed_filter(t, 3, 0)[3, 3] ≈ (8 * 0.2 + 1.0) / 9

    img = rand(40, 50)
    @test median_filter(img, 5) ≈ mapwindow(median, img, (5, 5))

    @test_throws ArgumentError median_filter(img, 4)
end
