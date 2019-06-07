
## Convolutional Neural Network Module
module convolutional_neural_network

include("./useful_functions.jl")
using .useful_functions
using Distributed

export NeuralNet, ConvLayer, PoolLayer, FCLayer, forward_propagation, backward_propagation, rand_normal,
    max_index, relu, drelu, leaky_relu, dleaky_relu, sigmoid, dsigmoid, tanh, dtanh, softmax, dsoftmax,
    cross_entropy, dcross_entropy, dmse, mae, dmae, pad

# Structure of Kernel
mutable struct Filter
    window::Array{Int}
    stride::Int
    channels::Int
    padding::Int
    func::Function
    __ws__::Union{Array{Float64}, Nothing}
end

# Constructor for Filter
Filter(window::Array{Int}, stride::Int, channels::Int, padding::Int, func::Function) =
    Filter(window, stride, channels, padding, func, channels == 0 ? nothing : rand(window..., channels))

# Supertype of all types of Layers of the Neural Network
abstract type Layer end

# Structure of Convolutional Layer
mutable struct ConvLayer <: Layer
    filters::Array{Filter}
    activ_func::Function
    dactiv_func::Function
    __bs__::Array{Float64}
end

# Constructors for Convolutional Layer
ConvLayer(window::Array{Int}, stride::Int, padding::Int, filters::Int, activ_func::Function, dactiv_func::Function) =
    ConvLayer([Filter(window, stride, 0, padding, sum) for i = 1:filters], activ_func, dactiv_func, fill(0.5, filters))

ConvLayer(window::Array{Int}, stride::Int, channels::Int, padding::Int, filters::Int, activ_func::Function, dactiv_func::Function) =
        ConvLayer([Filter(window, stride, channels, padding, sum) for i = 1:filters], activ_func, dactiv_func, fill(0.5, filters))

ConvLayer(window::Array{Int}, stride::Int, padding::Int, filters::Int) =
    ConvLayer([Filter(window, stride, 0, padding, sum) for i = 1:filters], relu, drelu, fill(0.5, filters))

ConvLayer(window::Array{Int}, stride::Int, channels::Int, padding::Int, filters::Int) =
    ConvLayer([Filter(window, stride, channels, padding, sum) for i = 1:filters], relu, drelu, fill(0.5, filters))

ConvLayer(window::Array{Int}, stride::Int, filters::Int) = ConvLayer(window, stride, 0, filters)

# Structure of Pooling Layer
mutable struct PoolLayer <: Layer
    filter::Filter
    index_cache::Union{Array{Bool}, Nothing}
end

# Constructor for Pooling Layer
PoolLayer(window::Array{Int}, stride::Int, channels::Int, func::Function) =
    PoolLayer(Filter(window, stride, channels, 0, func, ones(Float64, window..., channels)), nothing)
PoolLayer(window::Array{Int}, stride::Int, func::Function) =
    PoolLayer(Filter(window, stride, 0, 0, func), nothing)
PoolLayer(window::Array{Int}, stride::Int) = PoolLayer(Filter(window, stride, 0, 0, maximum), nothing)
PoolLayer(window::Array{Int}, stride::Int, channels::Int) =
    PoolLayer(Filter(window, stride, channels, 0, maximum, ones(Float64, window..., channels)), nothing)

# Structure of Fully Connected Layer
mutable struct FCLayer <: Layer
    neurons::Int
    activ_func::Function
    dactiv_func::Function
    __ws__::Union{Array{Float64}, Nothing}
    __bs__::Array{Float64}
end

# Constructors for Fully Connected Layer
FCLayer(neurons::Int, activ_func::Function, dactiv_func::Function) =
    FCLayer(neurons, activ_func, dactiv_func, nothing, fill(0.5, neurons, 1))

FCLayer(neurons::Int) = FCLayer(neurons, relu, drelu)

# Structure of Neural Network
mutable struct NeuralNet
    input_dims::Array{Int}
    layers::Array{Layer}
    loss_func::Function
    dloss_func::Function
end

# Constructor for Neural Network
function NeuralNet(input_dims::Array{Int}, layers::Layer...; loss_func::Function=cross_entropy, dloss_func::Function=dcross_entropy)
    in_dims = copy(input_dims)
    layers = [layers...]
    println(input_dims)
    for i = 1:length(layers)
        if typeof(layers[i]) == FCLayer
            print([layers[i].neurons, prod(input_dims)], ' ')
            layers[i] = FCLayer(layers[i].neurons, layers[i].activ_func,
                layers[i].dactiv_func, rand_normal(layers[i].neurons, prod(input_dims), stddev=0.01),
                layers[i].__bs__)
            input_dims = [layers[i].neurons, 1]
        elseif typeof(layers[i]) == ConvLayer
            if length(input_dims) < 3
                push!(input_dims, 1)
            end
            w, s, p = layers[i].filters[1].window, layers[i].filters[1].stride, layers[i].filters[1].padding
            layers[i] = ConvLayer(w, s, input_dims[end], p,
                length(layers[i].__bs__), layers[i].activ_func, layers[i].dactiv_func)
            input_dims[1] = div(input_dims[1] + 2 * p - w[1], s) + 1
            input_dims[2] = div(input_dims[2] + 2 * p - w[2], s) + 1
            input_dims[3] = length(layers[i].__bs__)
        elseif typeof(layers[i]) == PoolLayer
            if length(input_dims) < 3
                push!(input_dims, 1)
            end
            w, s = layers[i].filter.window, layers[i].filter.stride
            layers[i] = PoolLayer(w, s, input_dims[end])
            input_dims[1] = div(input_dims[1] - w[1], s) + 1
            input_dims[2] = div(input_dims[2] - w[2], s) + 1
        end
        println(input_dims)
    end
    return NeuralNet(in_dims, layers, loss_func, dloss_func)
end

function pad(arr, padding)
    if padding == 0
        return arr
    end
    dims = length(size(arr))
    if dims == 3
        parr = zeros(Float64, size(arr)[1] + 2*padding, size(arr)[2] + 2*padding, size(arr)[3])
        for i = 1:size(arr)[3]
            parr[(1 + padding):(size(parr)[1] - padding), (1 + padding):(size(parr)[2] - padding), i] = arr[:, :, i]
        end
        return parr
    elseif dims == 2
        parr = zeros(Float64, size(arr)[1] + 2*padding, size(arr)[2] + 2*padding)
        #println(size(parr), ' ', size(arr))
        parr[(1 + padding):(size(parr)[1] - padding), (1 + padding):(size(parr)[2] - padding)] = arr
        return parr
    end
    return arr
end

function unpad(arr, padding)
    arr[(1 + padding):(size(arr)[1] - padding), (1 + padding):(size(arr)[2] - padding), :]
end

# Filter for Convolutional Layer
function conv_filter(f::Filter, arr::Array{Float64})
    arr = pad(arr, f.padding)
    result = zeros(Float64, div(size(arr)[1] - f.window[1], f.stride) + 1,
        div(size(arr)[2] - f.window[2], f.stride) + 1)
    i1, j1 = 1, 1
    for i = 1:size(result)[1], j = 1:size(result)[2]
        i1, j1 = 1 + (i - 1) * f.stride, 1 + (j - 1) * f.stride
        result[i, j] = f.func(arr[i1:(i1 + f.window[1] - 1), j1:(j1 + f.window[2] - 1), :] .* f.__ws__)
    end
    result
end

# Filter for Pooling Layer
function pool_filter(f::Filter, arr::Array{Float64})
    result = zeros(Float64, div(size(arr)[1] + 2 * f.padding - f.window[1], f.stride) + 1,
        div(size(arr)[2] + 2 * f.padding - f.window[2], f.stride) + 1, f.channels)
    i1, j1 = 1, 1
    index = zeros(Bool, size(arr))
    for i = 1:size(result)[1], j = 1:size(result)[2], k = 1:f.channels
        i1, j1 = 1 + (i - 1) * f.stride, 1 + (j - 1) * f.stride
        i2, j2 = i1 + f.window[1] - 1, j1 + f.window[2] - 1
        result[i, j, k] = f.func(arr[i1:i2, j1:j2, k])
        index[i1:i2, j1:j2, k] = arr[i1:i2, j1:j2, k] .== result[i, j, k]
    end
    index, result
end

# Propagation through Convolutional Layer
propagate(conv::ConvLayer, x::Array{Float64}) =
let z = cat(pmap((f, b) -> conv_filter(f, x) .+ b, conv.filters, conv.__bs__)..., dims=3)
    z, conv.activ_func(z)
end

# Propagation through Pooling Layer
function propagate(pool::PoolLayer, x::Array{Float64})
    index, result = pool_filter(pool.filter, x)
    pool.index_cache = index
    result, result
end

# Propagation through Fully Connected Layer
propagate(fc::FCLayer, x::Array{Float64}) =
let z = fc.__ws__ * x[1:end] + fc.__bs__
    z, fc.activ_func(z)
end

# Forward Propagation
function forward_propagation(net, x)
    for layer in net.layers
        _, x = propagate(layer, x)
    end
    x
end

# Backward Propagation through Convolutional Layer
function back_propagate(conv::ConvLayer, z, a, a_prev, del, lr)
    #println(size.([z, a, a_prev, del]))
    del = reshape(del, size(a))
    dz = conv.dactiv_func(z, a) .* del
    del = pad(zeros(size(a_prev)), conv.filters[1].padding)
    for (i, f) in Iterators.enumerate(conv.filters)
        dw = zeros(size(f.__ws__))
        #println(size.([dw, dz, a_prev]))
        for j = 1:f.channels
            dw[:, :, j] = 
            conv_filter(Filter([size(dz[:, :, i])...], f.stride, f.channels, f.padding, f.func, dz[:, :, i]),
            a_prev[:, :, j])
        end
        #println(size.([dw, a[:, :, 1], dz[:, :, i], a_prev]))
        db = sum(dz[:, :, i]) / prod(size(dz[:, :, i]))
        for r = 1:size(dz)[1], c = 1:size(dz)[2]
            del[r:(r + f.window[1] - 1), c:(c + f.window[2] - 1), :] +=  dz[r, c, i] .* f.__ws__
        end
        f.__ws__ -= lr*dw
        conv.__bs__[i] -= lr*db
    end
    unpad(del, conv.filters[1].padding)
end

# Backward Propagation through Pooling Layer
back_propagate(pool::PoolLayer, z, a, a_prev, del, lr) = begin
    del = reshape(del, size(z))
    #println(size(del), ' ',cat(pool.filter.window, [1], dims=1), ' ', size(pool.index_cache))
    repeat(del, inner=cat(pool.filter.window, [1], dims=1)) .* pool.index_cache
end

# Backward Propagation through Fully Connected Layer
function back_propagate(fc::FCLayer, z, a, a_prev, del, lr)
    dz = fc.activ_func == softmax ? dsoftmax(z, a) * del : fc.dactiv_func(z, a) .* del
    dw = dz * transpose(a_prev[1:end])
    db = dz
    del = transpose(fc.__ws__) * dz
    fc.__ws__ -= lr*dw
    fc.__bs__ -= lr*db
    del
end

# Backward Propagation
function backward_propagation(net, x, y; lr=0.01)
    zs, as = Array{Float64}[], Array{Float64}[]
    push!(as, x)
    #println(size(x))
    for layer in net.layers
        z, a = propagate(layer, as[end])
        push!(zs, z)
        push!(as, a)
        #println(size(a))
    end
    loss = net.loss_func(as[end], y)
    #println(as[end])
    dloss = net.dloss_func(as[end], y)
    for layer in Iterators.reverse(net.layers)
        dloss = back_propagate(layer, pop!(zs), pop!(as), as[end], dloss, lr)
    end
    loss
end

end