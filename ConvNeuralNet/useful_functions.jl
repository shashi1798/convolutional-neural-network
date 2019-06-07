module useful_functions

export rand_normal, max_index, relu, drelu, leaky_relu, dleaky_relu, sigmoid, dsigmoid, tanh, dtanh,
    softmax, dsoftmax, cross_entropy, dcross_entropy, mse, dmse, mae, dmae

# Normal distribution with 0 mean and given Standard deviation
function rand_normal(dims::Int...; stddev::Float64 = 0.01)
    if stddev <= 0.0
       error("Standard deviation must be positive")
    end
    u1 = rand(dims...)
    u2 = rand(dims...)
    r = @. sqrt(-2.0 * log(u1))
    theta = 2.0 * pi * u2
    return @. stddev * (r * sin(theta))
end

# Returns the index and value of the maximum value
function max_index(arr)
    max_val = 0
    index = 0
    for i = 1:length(arr)
        if arr[i] > max_val
            max_val = arr[i]
            index = i
        end
    end
    return index, max_val
end

## ACTIVATION FUNCTIONS

# Relu activation Function
relu(z) = @. max(0, z)

# Derivative of Relu Function
drelu(z, a) = [x >= 0 ? 1 : 0 for x in z]

# Leaky Relu activation Function
leaky_relu(z) = @. max(0.01 * z, z)

# Derivative of Leaky Relu Function
dleaky_relu(z, a) = [x >= 0 ? 1 : 0.01 for x in z]

# Sigmoid activation Function
sigmoid(z) = @. 1 / (1 + exp(-z))

# Derivative of Sigmoid Function
dsigmoid(z, a) = @. a * (1 - a)

# Derivative of Tanh Function
dtanh(z, a) = @. 1 - a * a

# Softmax Function
softmax(z) =
let exps = @. exp(z)
    exps / sum(exps)
end

# Derivative of Softmax Function
function dsoftmax(z, a)
 dim = size(a)[1]
    perr = zeros(dim, dim)
    for i = 1:dim
     for j = 1:dim
         if i == j
             perr[i, j] = a[i] * (1 - a[j])
         else
             perr[i, j] = -a[i] * a[j]
         end
     end
    end
    return perr
end

## LOSS FUNCTIONS

# Cross Entropy Function
cross_entropy(pred, y) = -sum(@. y * log([p > -Inf ? p : 1 for p in pred]))

# Derivative of Cross Entropy Function
dcross_entropy(pred, y) = @. -y / pred

# Mean Square Error
mse(pred, y) = sum((pred - y).^2) / length(pred)

# Derivative of Mean Square Error
dmse(pred, y) = 2*(pred - y) / length(pred)

# Mean Absolute Error
mae(pred, y) = sum(abs.(pred - y)) / length(pred)

# Derivative of Mean Absolute Error
dmae(pred, y) = [pi > yi ? 1 : -1 for (pi, yi) in zip(pred, y)]

end