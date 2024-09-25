package nn

import (
	"encoding/gob"
	"fmt"
	"os"
)

// Save saves the network's weights and biases to a file
func Save(net *Network, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("error creating file for saving: %v", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(net)
	if err != nil {
		return fmt.Errorf("error encoding network: %v", err)
	}

	return nil
}

// Load loads the network's weights and biases from a file
func Load(filePath string) (*Network, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening file for loading: %v", err)
	}
	defer file.Close()

	var net Network
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&net)
	if err != nil {
		return nil, fmt.Errorf("error decoding network: %v", err)
	}

	return &net, nil
}

// MatrixPrint prints the output in a readable format
func MatrixPrint(output []float64) {
	for _, val := range output {
		fmt.Printf("%.4f ", val)
	}
	fmt.Println()
}
