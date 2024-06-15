// special thanks to Chang Sau Sheong for the informative arctile
// on building neural networks

package main

import (
	"github.com/OSokunbi/neural-net/nn"
)

func main() {
	// Greater number of hidden nodes will give you a more accurate model
	net := nn.CreateNetwork(2, 5, 1, 0.5)
	trainingData := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}
	for i := 0; i < 10000; i++ {
		for _, data := range trainingData {
			net.Train(data[0], data[1])
		}
	}
	nn.MatrixPrint(net.Predict([]float64{0, 0}))
	nn.MatrixPrint(net.Predict([]float64{0, 1}))
	nn.MatrixPrint(net.Predict([]float64{1, 0}))
	nn.MatrixPrint(net.Predict([]float64{1, 1}))
	// save the model so you dont have to retrain it every time
	nn.Save(net)	
}

// loading the model would look like this {
// func main() {
//  net := CreateNetwork(2, 5, 1, 0.5)
//  nn.Load(&net)
//  nn.MatrixPrint(net.Predict([]float64{0, 0}))
//  nn.MatrixPrint(net.Predict([]float64{0, 1}))
//  nn.MatrixPrint(net.Predict([]float64{1, 0}))
//  nn.MatrixPrint(net.Predict([]float64{1, 1}))
// }
