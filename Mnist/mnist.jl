include("../ConvNeuralNet/convolutional_neural_network.jl")

using Random
using .convolutional_neural_network

# Read images from file
function read_image(path)
    println("Reading Images...")
    bytes = open(path) do file
        read(file)
    end
    number_of_images = bytes[5] * 256^3 + bytes[6] * 256^2 + bytes[7] * 256 + bytes[8]
    rows = Int(bytes[12])
    cols = Int(bytes[16])
    println("Number of images = $number_of_images, rows = $rows, cols = $cols, Image size = $rows x $cols")
	images = zeros(number_of_images, rows, cols)
	i = 1
	for i = 1:number_of_images
		for r = 1:rows
			for c = 1:cols
				images[i, r, c] = bytes[28(28*i + r) + c - 796] / 255.0
			end
		end
    end
    println("Done.")
    return images
end

# Read labels from file
function read_label(path)
    bytes = open(path) do file
        read(file)
    end
    number_of_labels = bytes[5] * 256^3 + bytes[6] * 256^2 + bytes[7] * 256 + bytes[8]
    println("Number of labels = $number_of_labels")
    labels = zeros(number_of_labels, 10)
    i = 1
    for byte in bytes[9:end]
        labels[i, byte + 1] = 1.0
        i += 1
    end
    println("Done.")
    return labels
end

# Read all the image and label files
function read_data()
    x_train = read_image("train-images.idx3-ubyte")
    y_train = read_label("train-labels.idx1-ubyte")
    x_test = read_image("t10k-images.idx3-ubyte")
    y_test = read_label("t10k-labels.idx1-ubyte")
    return [x_train, y_train, x_test, y_test]
end

function train(net, x_train, y_train, lr)
    println("\n\nLearning Rate = $lr\n")
    for i = 1:60000
        x, y = x_train[i, :, :], y_train[i, :]
        loss = backward_propagation(net, x, y, lr=lr)
        if floor(rand() * 3000) == 5.0
            println("Loss: ", loss)
        end
    end
end

function test(net, x_test, y_test)
    correct = 0
    n = 10000
    for i = 1:n
        x, y = x_test[i, :, :], y_test[i, :]
        pred = forward_propagation(net, x)
        correct += max_index(pred)[1] == max_index(y)[1]
    end
    println((correct * 100) / n)
end

function main()
    Random.seed!(0)
    x_train, y_train, x_test, y_test = read_data()
    net = NeuralNet([28, 28, 1],
        FCLayer(180),
        FCLayer(10, softmax, dsoftmax)
    )
    @time train(net, x_train, y_train, 0.0096)
    @time test(net, x_test, y_test)
end

main()