using PyCall
itk = pyimport("itk")
np = pyimport("numpy")


using StaticArrays
using CuArrays, CUDAnative
using CuArrays, CuTextures
using NRRD
using Rotations


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
    array::Any
    direction::Mat3
    spacing::Vec3
    largestPossibleRegion::Vec3
    ItkImage(pyobject) = new(
        np.array(pyobject), 
        Mat3(itk.array_from_matrix(pyobject.GetDirection())),
        Vec3(np.array(pyobject.GetSpacing())),
        Vec3(np.array(pyobject.GetLargestPossibleRegion().GetSize()))
    )
end

struct GPUItkImage
    texture::Any
    direction::Mat3
    spacing::Vec3
    largestPossibleRegion::Vec3
    GPUItkImage(object) = new(
        CuTexture(CuTextureArray(Float32.(object.array))), 
        object.direction,
        object.spacing,
        object.largestPossibleRegion
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
    #R, _ = qr(randn((3, 3)))
    #lambda, J = eigen(R)
    #return real(J * Diagonal(lambda .^ (1 / factor)) * inv(J))
    return RotXYZ(pi / factor .* randn(3)...)
end

function generate_sample(jimage::ItkImage, annotation)
    gimage= GPUItkImage(jimage)
    
    res = generate_sample(gimage, annotation)
    
    CuTextures.unsafe_free!(gimage.texture)
    return res
end

function generate_sample(jimage::GPUItkImage, annotation)
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


function warp(dst, texture, origin::Vec3, direction::Mat3, image_direction::Mat3, image_spacing::Vec3, size::Vec3)
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
    
    
    @inbounds dst[i,j] = texture(pixel_position[3] / size[3], pixel_position[2] / size[2], pixel_position[1] / size[1])
    return nothing
end

function spine_slice(image, params)
    output_direction = params.direction * Diagonal([params.height / params.px, params.width / params.px, 0])
    
    output_origin = params.origin .+ params.direction * [-params.height / 2; -params.width / 2; 0]
    
    outimg_d = CuArray{Float32}(undef, 128, 128)
    @cuda threads = (128, 1) blocks = (1, 128) warp(
        outimg_d, image.texture, Vec3(output_origin), Mat3(output_direction), image.direction, image.spacing,
        image.largestPossibleRegion
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

function generate_data(jimages, annotations)
    res = []#::Array{Dict{String,Array{T,1} where T}, 1} = []
    for(jimage, annotation) = zip(jimages, annotations)
        gimage = GPUItkImage(jimage)
        for(i) = 1:67
            push!(res, generate_sample(gimage, annotation))
            #push!(res, ultrasoundgeneration.generate_sample(
            #        jimage, annotation))
        end
        CuTextures.unsafe_free!(gimage.texture)
    end
    print("i")
    
    data = [] #::Array{Array{Float32, 3}, 1} = []
    classes = []
    for elem = res
        data_entry = cat(
            [reshape(x, Val(3)) for x in elem["data"]]...;
            dims=3
        ).* 1.0f0
        data_entry .+= 1000
        data_entry ./= 2000
        
        push!(data, np.array(data_entry))
        
        c = elem["classes"] 
        class_entry = vcat(c[1] ./ 4, 4 .* [c[2].theta1, c[2].theta2, c[2].theta3])
        #class_entry[[4, 8, 12]] .-= 40
        push!(classes, class_entry)
    end
    return data, classes
end

function longcat(data)
    data2 = zeros(Float32, (length(data), size(data[1])...))
    for i = 1:length(data)
        data2[i, :, :, :] = data[i]
    end
    return data2
end
