package nn

import (
	"math"
)

// Sigmoid activation function
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidDerivative computes the derivative of the sigmoid function
func SigmoidDerivative(sigmoidOutput float64) float64 {
	return sigmoidOutput * (1.0 - sigmoidOutput)
}
