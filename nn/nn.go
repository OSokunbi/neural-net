package nn

import (
  "gonum.org/v1/gonum/mat"
  "gonum.org/v1/gonum/stat/distuv"
  "os"
  "math"
  "fmt"
)

// neural net
type Network struct {
  inputs        int
  hiddens        int
  outputs       int
  hiddenWeights *mat.Dense
  outputWeights *mat.Dense
  learningRate  float64
}

func CreateNetwork(input, hidden, output int, rate float64) Network{
  net := Network {
    inputs:       input,
    hiddens:      hidden,
    outputs:      output,
    learningRate: rate,
  }
  net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
  net.outputWeights = mat.NewDense(net.outputs, net.hiddens, randomArray(net.hiddens*net.outputs, float64(net.hiddens)))
  return net
}

func randomArray(size int, v float64) []float64 {
  dist := distuv.Uniform{
    Min: -1 / math.Sqrt(v),
    Max: 1 / math.Sqrt(v),
  }

  data := make([]float64, size)
  for i := 0; i < size; i++{
    data[i] = dist.Rand()
  }
  return data
}

func(net Network) Predict(inputData []float64) mat.Matrix {
  // forward propgation
  inputs := mat.NewDense(len(inputData), 1, inputData)
  hiddenInputs := dot(net.hiddenWeights, inputs)
  hiddenOutputs := apply(sigmoid, hiddenInputs)
  finalInputs := dot(net.outputWeights, hiddenOutputs)
  finalOutputs := apply(sigmoid, finalInputs)
  return finalOutputs
}

func (net *Network) Train(inputData []float64, targetData []float64) {
  // forward propgation
  inputs := mat.NewDense(len(inputData), 1, inputData)
  hiddenInputs := dot(net.hiddenWeights, inputs)
  hiddenOutputs := apply(sigmoid, hiddenInputs)
  finalInputs := dot(net.outputWeights, hiddenOutputs)
  finalOutputs := apply(sigmoid, finalInputs)

  // find errors
  targets := mat.NewDense(len(targetData), 1, targetData)
  outputErrors := subtract(targets, finalOutputs)
  hiddenErrors := dot(net.outputWeights.T(), outputErrors)

  // backpropgation
  net.outputWeights = add(net.outputWeights,
                          scale(net.learningRate,
                                 dot(multiply(outputErrors, sigmoidPrime(finalOutputs)),
                                      hiddenOutputs.T()))).(*mat.Dense)

  net.hiddenWeights = add(net.hiddenWeights,
                          scale(net.learningRate,
                                dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
                                    inputs.T()))).(*mat.Dense)
}

// A set of helper functions
// dot product operation on two matrices
func dot(m, n mat.Matrix) mat.Matrix {
  r, _ := m.Dims()
  _, c := n.Dims()
  o := mat.NewDense(r, c, nil)
  o.Product(m, n)
  return o
}

// apply a function to a matrix
func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
  r, c := m.Dims()
  o := mat.NewDense(r, c, nil)
  o.Apply(fn, m)
  return o
}

// scale a matrix 
func scale(s float64, m mat.Matrix) mat.Matrix {
  r, c := m.Dims()
  o := mat.NewDense(r, c, nil)
  o.Scale(s, m)
  return o
}

// multiple two matrices
func multiply(m, n mat.Matrix) mat.Matrix{
  r, c := m.Dims()
  o := mat.NewDense(r, c, nil)
  o.MulElem(m, n)
  return o
}

// add function to another
func add(m, n mat.Matrix) mat.Matrix {
  r, c := m.Dims()
  o := mat.NewDense(r, c, nil)
  o.Add(m, n)
  return o
}

// subtract function from another
func subtract(m, n mat.Matrix) mat.Matrix {
  r, c := m.Dims()
  o := mat.NewDense(r, c, nil)
  o.Sub(m, n)
  return o
}

// add a scalar value to each element in the matrix
func addScalar(x float64, m mat.Matrix) mat.Matrix{
  r, c := m.Dims()
  a := make([]float64, r*c)
  for i := 0; i < r*c; i++ {
    a[i] = x
  }

  n := mat.NewDense(r, c, a)
  return add(m, n)
}

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
  r, _ := m.Dims()
  o := make([]float64, r)
  for i := range o {
    o[i] = 1.0
  }

  ones := mat.NewDense(r, 1, o)
  return multiply(m, subtract(ones, m))
}

func Save(net Network) {
  h, err := os.Create("data/hweights.model")
  defer h.Close()
  if err == nil {
    net.hiddenWeights.MarshalBinaryTo(h)
  } else {
    fmt.Println(err)
  }

  o, err := os.Create("data/oweights.model")
  defer o.Close()
  if err == nil {
    net.outputWeights.MarshalBinaryTo(o)
  } else {
    fmt.Println(err)
  }
  return
}

  func Load(net *Network) {
    h, err := os.Open("data/hweights.model")
    defer h.Close()
    if err == nil {
      net.hiddenWeights.Reset()
      net.hiddenWeights.UnmarshalBinaryFrom(h)
    }
    o, err := os.Open("data/oweights.model")
    defer o.Close()
    if err == nil {
      net.outputWeights.Reset()
      net.outputWeights.UnmarshalBinaryFrom(o)
    }
    return
}
func MatrixPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}