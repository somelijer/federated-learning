package randomgenerator

import (
	"math/rand"
	"time"

	"main.go/model"
)

func RandomWeights() model.Weights {
	rand.Seed(time.Now().UnixNano())

	// Generisanje slučajnih težina za Conv1Weight
	conv1Weight := make([][][][]float64, 1)
	for i := range conv1Weight {
		conv1Weight[i] = make([][][]float64, 1)
		for j := range conv1Weight[i] {
			conv1Weight[i][j] = make([][]float64, 1)
			for k := range conv1Weight[i][j] {
				conv1Weight[i][j][k] = make([]float64, 2)
				for l := range conv1Weight[i][j][k] {
					conv1Weight[i][j][k][l] = rand.Float64()
				}
			}
		}
	}

	// Generisanje slučajnih težina za Conv1Bias
	conv1Bias := make([]float64, 2)
	for i := range conv1Bias {
		conv1Bias[i] = rand.Float64()
	}

	// Generisanje slučajnih težina za Conv2Bias
	conv2Bias := make([]float64, 2)
	for i := range conv2Bias {
		conv2Bias[i] = rand.Float64()
	}

	// Generisanje slučajnih težina za Fc1Weight
	fc1Weight := make([][]float64, 1)
	for i := range fc1Weight {
		fc1Weight[i] = make([]float64, 2)
		for j := range fc1Weight[i] {
			fc1Weight[i][j] = rand.Float64()
		}
	}

	// Generisanje slučajnih težina za Fc1Bias
	fc1Bias := make([]float64, 2)
	for i := range fc1Bias {
		fc1Bias[i] = rand.Float64()
	}

	// Generisanje slučajnih težina za Fc2Weight
	fc2Weight := make([][]float64, 1)
	for i := range fc2Weight {
		fc2Weight[i] = make([]float64, 2)
		for j := range fc2Weight[i] {
			fc2Weight[i][j] = rand.Float64()
		}
	}

	// Generisanje slučajnih težina za Fc2Bias
	fc2Bias := make([]float64, 2)
	for i := range fc2Bias {
		fc2Bias[i] = rand.Float64()
	}

	// Generisanje slučajnih težina za Fc3Weight
	fc3Weight := make([][]float64, 1)
	for i := range fc3Weight {
		fc3Weight[i] = make([]float64, 2)
		for j := range fc3Weight[i] {
			fc3Weight[i][j] = rand.Float64()
		}
	}

	// Generisanje slučajnih težina za Fc3Bias
	fc3Bias := make([]float64, 2)
	for i := range fc3Bias {
		fc3Bias[i] = rand.Float64()
	}

	return model.Weights{
		Conv1Weight: conv1Weight,
		Conv1Bias:   conv1Bias,
		Conv2Bias:   conv2Bias,
		Fc1Weight:   fc1Weight,
		Fc1Bias:     fc1Bias,
		Fc2Weight:   fc2Weight,
		Fc2Bias:     fc2Bias,
		Fc3Weight:   fc3Weight,
		Fc3Bias:     fc3Bias,
	}
}
