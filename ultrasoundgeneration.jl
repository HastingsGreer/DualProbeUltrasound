using PyCall
itk = pyimport("itk")
np = pyimport("numpy")


using StaticArrays
using CuArrays, CUDAnative
using CuArrays, CuTextures
using NRRD


using LinearAlgebra

Vec3 = SVector{3,Float32}
Mat3 = SMatrix{3, 3, Float32, 9}

struct TwoSensorProbe
    side_offset::Float64
    angle::Float64
end

struct SliceParams #measured from center
    origin::Array{Float64,1}
    direction::Array{Float64, 2}
    width::Float64
    height::Float64
    px::Int64
end

struct ItkImage
    texture::Any
    direction::Mat3
    spacing::Vec3
    ItkImage(pyobject) = new(
        CuTexture(CuTextureArray(Float32.(np.array(pyobject)))), 
        Mat3(itk.array_from_matrix(pyobject.GetDirection())),
        Vec3(np.array(pyobject.GetSpacing()))
    )
end

function get_origin_direction(probe::TwoSensorProbe, origin, direction, idx)
    @assert(idx == 1 || idx == -1)
    @. origin = origin + idx * probe.side_offset * direction[:, 3]
    angle = probe.angle * idx
    direction = direction * [1 0 0; 0 cos(angle) sin(angle); 0 -sin(angle) cos(angle)]
    return origin, direction
end

function random_small_rotation(factor)
    R, _ = qr(randn((3, 3)))
    lambda, J = eigen(R)
    return real(J * Diagonal(lambda .^ (1 / factor)) * inv(J))
end

function generate_sample(jimage, annotation)
    initial_position_randomness_scale = 15 #mm
    movement_scale = 10 #mm
    slice_idx = rand(50:length(annotation[1]) - 50) #vals["slice_idx"]

    direction = random_small_rotation(15) * [0 0 1; -1 0 0; 0 -1 0]

    #direction = vals["direction"]

    # depends on annotation having 1 mm spacing along spine
    origin = (
        [annotation[2][slice_idx], -annotation[1][slice_idx], -slice_idx] 
        .+ initial_position_randomness_scale .* randn(Float64, 3) 
    )

    #debug
    #origin = vals["origin"]

    probe = TwoSensorProbe(20, -np.pi / 4)

    a, b = slice_multiprobe(jimage, probe, origin, direction)

    movement_relative_to_image_1 = randn(3) .* movement_scale
    rotation_relative_to_image_1 = random_small_rotation(25)

    direction_2 = direction * rotation_relative_to_image_1
    origin_2 = origin .+ direction * movement_relative_to_image_1

    c, d = slice_multiprobe(jimage, probe, origin_2, direction_2)

    return Dict(["data"=> [a, b, c, d], "classes"=> [movement_relative_to_image_1, rotation_relative_to_image_1]])
end


function warp(dst, texture, origin::Vec3, direction::Mat3, image_direction::Mat3, image_spacing::Vec3)
    i::Int32 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j::Int32 = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    #u = (Float32(i) - 1f0) / (Float32(size(dst, 1)) - 1f0)
    #v = (Float32(j) - 1f0) / (Float32(size(dst, 2)) - 1f0)
    #x = u  #+ 0.02f0 * CUDAnative.sin(30v)
    #y = v #+ 0.03f0 * CUDAnative.sin(20u)
    
    pixel_position = origin .+ direction[:, 1] .* Float32(j) 
    pixel_position = pixel_position .+ direction[:, 2] .* Float32(i)
    
    pixel_position = pixel_position ./ image_spacing
    direction_flat = (image_direction[1, 1], image_direction[2, 2], image_direction[3, 3])
    pixel_position = pixel_position ./ direction_flat
    
    
    @inbounds dst[i,j] = texture(pixel_position[3] / 223, pixel_position[2] / 512, pixel_position[1] / 512)
    return nothing
end

function spine_slice(image, params)
    output_direction = params.direction * Diagonal([params.height / params.px, params.width / params.px, 0])
    
    output_origin = params.origin .+ params.direction * [-params.height / 2; -params.width / 2; 0]
    
    outimg_d = CuArray{Float32}(undef, 128, 128)
    @cuda threads = (128, 1) blocks = (1, 128) warp(
        outimg_d, image.texture, Vec3(output_origin), Mat3(output_direction), jimage.direction, jimage.spacing
    )
    return Array(outimg_d)
end

function slice_multiprobe(jimage, probe, origin, direction)
    res = []
    params_array = []
    for idx = [-1, 1]
        slice_origin, slice_direction = get_origin_direction(probe, origin, direction, idx)
        #println(slice_origin)
        #print(slice_direction)
        params = SliceParams(slice_origin, slice_direction, 100, 100, 128)
        x = spine_slice(jimage, params)
        push!(res, x)
        push!(params_array, params)
    end
    return res
end