package nn
 
import (
	"encoding/gob"
	"math/rand"
	"time"
)

// Network represents the neural network with one hidden layer
type Network struct {
	InputSize   int
	HiddenSize  int
	OutputSize  int
	LearningRate float64

	// Weights and biases
	W1 [][]float64 // Input to Hidden
	B1 []float64
	W2 [][]float64 // Hidden to Output
	B2 []float64
}

// CreateNetwork initializes a new neural network with given sizes and learning rate
func CreateNetwork(inputSize, hiddenSize, outputSize int, learningRate float64) *Network {
	rand.Seed(time.Now().UnixNano())

	// Initialize weights with small random values
	w1 := make([][]float64, inputSize)
	for i := 0; i < inputSize; i++ {
		w1[i] = make([]float64, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			w1[i][j] = rand.Float64()*0.1 - 0.05 // [-0.05, 0.05]
		}
	}

	b1 := make([]float64, hiddenSize) // Initialized to 0

	w2 := make([][]float64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		w2[i] = make([]float64, outputSize)
		for j := 0; j < outputSize; j++ {
			w2[i][j] = rand.Float64()*0.1 - 0.05 // [-0.05, 0.05]
		}
	}

	b2 := make([]float64, outputSize) // Initialized to 0

	return &Network{
		InputSize:    inputSize,
		HiddenSize:   hiddenSize,
		OutputSize:   outputSize,
		LearningRate: learningRate,
		W1:           w1,
		B1:           b1,
		W2:           w2,
		B2:           b2,
	}
}

// Train performs a single training step with given input and target
func (net *Network) Train(input, target []float64) {
	// Forward pass
	hiddenInputs := make([]float64, net.HiddenSize)
	for i := 0; i < net.HiddenSize; i++ {
		sum := net.B1[i]
		for j := 0; j < net.InputSize; j++ {
			sum += input[j] * net.W1[j][i]
		}
		hiddenInputs[i] = Sigmoid(sum)
	}

	outputs := make([]float64, net.OutputSize)
	for i := 0; i < net.OutputSize; i++ {
		sum := net.B2[i]
		for j := 0; j < net.HiddenSize; j++ {
			sum += hiddenInputs[j] * net.W2[j][i]
		}
		outputs[i] = Sigmoid(sum)
	}

	// Calculate output errors (target - output)
	outputErrors := make([]float64, net.OutputSize)
	for i := 0; i < net.OutputSize; i++ {
		outputErrors[i] = target[i] - outputs[i]
	}

	// Calculate gradients for W2 and B2
	dW2 := make([][]float64, net.HiddenSize)
	dB2 := make([]float64, net.OutputSize)
	for i := 0; i < net.OutputSize; i++ {
		dB2[i] = outputErrors[i] * SigmoidDerivative(outputs[i])
		for j := 0; j < net.HiddenSize; j++ {
			if dW2[j] == nil {
				dW2[j] = make([]float64, net.OutputSize)
			}
			dW2[j][i] = hiddenInputs[j] * dB2[i]
		}
	}

	// Calculate hidden layer errors
	hiddenErrors := make([]float64, net.HiddenSize)
	for i := 0; i < net.HiddenSize; i++ {
		errorSum := 0.0
		for j := 0; j < net.OutputSize; j++ {
			errorSum += net.W2[i][j] * outputErrors[j]
		}
		hiddenErrors[i] = errorSum * SigmoidDerivative(hiddenInputs[i])
	}

	// Calculate gradients for W1 and B1
	dW1 := make([][]float64, net.InputSize)
	dB1 := make([]float64, net.HiddenSize)
	for i := 0; i < net.HiddenSize; i++ {
		dB1[i] = hiddenErrors[i]
		for j := 0; j < net.InputSize; j++ {
			if dW1[j] == nil {
				dW1[j] = make([]float64, net.HiddenSize)
			}
			dW1[j][i] = input[j] * dB1[i]
		}
	}

	// Update weights and biases
	for i := 0; i < net.OutputSize; i++ {
		net.B2[i] += net.LearningRate * dB2[i]
		for j := 0; j < net.HiddenSize; j++ {
			net.W2[j][i] += net.LearningRate * dW2[j][i]
		}
	}

	for i := 0; i < net.HiddenSize; i++ {
		net.B1[i] += net.LearningRate * dB1[i]
		for j := 0; j < net.InputSize; j++ {
			net.W1[j][i] += net.LearningRate * dW1[j][i]
		}
	}
}

// Predict returns the network's output for a given input
func (net *Network) Predict(input []float64) []float64 {
	// Forward pass
	hiddenInputs := make([]float64, net.HiddenSize)
	for i := 0; i < net.HiddenSize; i++ {
		sum := net.B1[i]
		for j := 0; j < net.InputSize; j++ {
			sum += input[j] * net.W1[j][i]
		}
		hiddenInputs[i] = Sigmoid(sum)
	}

	outputs := make([]float64, net.OutputSize)
	for i := 0; i < net.OutputSize; i++ {
		sum := net.B2[i]
		for j := 0; j < net.HiddenSize; j++ {
			sum += hiddenInputs[j] * net.W2[j][i]
		}
		outputs[i] = Sigmoid(sum)
	}

	return outputs
}

// CalculateLoss computes the Mean Squared Error over the training data
func (net *Network) CalculateLoss(trainingData [][][]float64) float64 {
	totalLoss := 0.0
	for _, sample := range trainingData {
		prediction := net.Predict(sample[0])
		for i := 0; i < net.OutputSize; i++ {
			error := sample[1][i] - prediction[i]
			totalLoss += error * error
		}
	}
	return totalLoss / float64(len(trainingData))
}

func init() {
	gob.Register(&Network{})
}
