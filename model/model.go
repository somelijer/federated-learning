package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// neuralNet contains all of the information
// that defines a trained neural network.
type neuralNet struct {
	config     neuralNetConfig
	wHidden1   *mat.Dense
	bHidden1   *mat.Dense
	wHidden2   *mat.Dense
	bHidden2   *mat.Dense
	wOut       *mat.Dense
	bOut       *mat.Dense
}

// neuralNetConfig defines our neural network
// architecture and learning parameters.
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons1 int
	hiddenNeurons2 int
	numEpochs     int
	learningRate  float64
}

func main() {

	// Form the training matrices.
	inputs, labels := makeInputsAndLabels("data/train.csv")

	// Define our network architecture and learning parameters.
	config := neuralNetConfig{
		inputNeurons:  28 * 28,
		outputNeurons: 10,
		hiddenNeurons1: 64,
		hiddenNeurons2: 64,
		numEpochs:     1000,
		learningRate:  0.01,
	}

	// Train the neural network.
	network := newNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Form the testing matrices.
	testInputs, testLabels := makeInputsAndLabels("data/test.csv")

	// Make the predictions using the trained model.
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate the accuracy of our model.
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Get the label.
		labelRow := mat.Row(nil, i, testLabels)
		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Accumulate the true positive/negative count.
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Calculate the accuracy (subset accuracy).
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Output the Accuracy value to standard out.
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}

// NewNetwork initializes a new neural network.
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// train trains a neural network using backpropagation.
func (nn *neuralNet) train(x, y *mat.Dense) error {

	// Initialize biases/weights.
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden1 := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons1, nil)
	bHidden1 := mat.NewDense(1, nn.config.hiddenNeurons1, nil)
	wHidden2 := mat.NewDense(nn.config.hiddenNeurons1, nn.config.hiddenNeurons2, nil)
	bHidden2 := mat.NewDense(1, nn.config.hiddenNeurons2, nil)
	wOut := mat.NewDense(nn.config.hiddenNeurons2, nn.config.outputNeurons, nil)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	wHidden1Raw := wHidden1.RawMatrix().Data
	bHidden1Raw := bHidden1.RawMatrix().Data
	wHidden2Raw := wHidden2.RawMatrix().Data
	bHidden2Raw := bHidden2.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHidden1Raw,
		bHidden1Raw,
		wHidden2Raw,
		bHidden2Raw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Use backpropagation to adjust the weights and biases.
	if err := nn.backpropagate(x, y, wHidden1, bHidden1, wHidden2, bHidden2, wOut, bOut, output); err != nil {
		return err
	}

	// Define our trained neural network.
	nn.wHidden1 = wHidden1
	nn.bHidden1 = bHidden1
	nn.wHidden2 = wHidden2
	nn.bHidden2 = bHidden2
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

// backpropagate completes the backpropagation method.
func (nn *neuralNet) backpropagate(x, y, wHidden1, bHidden1, wHidden2, bHidden2, wOut, bOut, output *mat.Dense) error {

	// Loop over the number of epochs utilizing
	// backpropagation to train our model.
	for i := 0; i < nn.config.numEpochs; i++ {

		// Complete the feed forward process.
		hiddenLayerInput1 := new(mat.Dense)
		hiddenLayerInput1.Mul(x, wHidden1)
		addBHidden1 := func(_, col int, v float64) float64 { return v + bHidden1.At(0, col) }
		hiddenLayerInput1.Apply(addBHidden1, hiddenLayerInput1)

		hiddenLayerActivations1 := new(mat.Dense)
		applyReLU := func(_, _ int, v float64) float64 { return relu(v) }
		hiddenLayerActivations1.Apply(applyReLU, hiddenLayerInput1)

		hiddenLayerInput2 := new(mat.Dense)
		hiddenLayerInput2.Mul(hiddenLayerActivations1, wHidden2)
		addBHidden2 := func(_, col int, v float64) float64 { return v + bHidden2.At(0, col) }
		hiddenLayerInput2.Apply(addBHidden2, hiddenLayerInput2)

		hiddenLayerActivations2 := new(mat.Dense)
		hiddenLayerActivations2.Apply(applyReLU, hiddenLayerInput2)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations2, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySoftmax, outputLayerInput)

		// Complete the backpropagation.
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySoftmaxPrime := func(_, _ int, v float64) float64 { return softmaxPrime(v) }
		slopeOutputLayer.Apply(applySoftmaxPrime, output)
		slopeHiddenLayer2 := new(mat.Dense)
		slopeHiddenLayer2.Apply(reluPrime, hiddenLayerActivations2)
		slopeHiddenLayer1 := new(mat.Dense)
		slopeHiddenLayer1.Apply(reluPrime, hiddenLayerActivations1)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer2 := new(mat.Dense)
		errorAtHiddenLayer2.Mul(dOutput, wOut.T())

		dHiddenLayer2 := new(mat.Dense)
		dHiddenLayer2.MulElem(errorAtHiddenLayer2, slopeHiddenLayer2)
		errorAtHiddenLayer1 := new(mat.Dense)
		errorAtHiddenLayer1.Mul(dHiddenLayer2, wHidden2.T())

		dHiddenLayer1 := new(mat.Dense)
		dHiddenLayer1.MulElem(errorAtHiddenLayer1, slopeHiddenLayer1)

		// Adjust the parameters.
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations2.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHidden2Adj := new(mat.Dense)
		wHidden2Adj.Mul(hiddenLayerActivations1.T(), dHiddenLayer2)
		wHidden2Adj.Scale(nn.config.learningRate, wHidden2Adj)
		wHidden2.Add(wHidden2, wHidden2Adj)

		bHidden2Adj, err := sumAlongAxis(0, dHiddenLayer2)
		if err != nil {
			return err
		}
		bHidden2Adj.Scale(nn.config.learningRate, bHidden2Adj)
		bHidden2.Add(bHidden2, bHidden2Adj)

		wHidden1Adj := new(mat.Dense)
		wHidden1Adj.Mul(x.T(), dHiddenLayer1)
		wHidden1Adj.Scale(nn.config.learningRate, wHidden1Adj)
		wHidden1.Add(wHidden1, wHidden1Adj)

		bHidden1Adj, err := sumAlongAxis(0, dHiddenLayer1)
		if err != nil {
			return err
		}
		bHidden1Adj.Scale(nn.config.learningRate, bHidden1Adj)
		bHidden1.Add(bHidden1, bHidden1Adj)
	}

	return nil
}

// predict makes a prediction based on a trained
// neural network.
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

	// Check to make sure that our neuralNet value
	// represents a trained model.
	if nn.wHidden1 == nil || nn.wHidden2 == nil || nn.wOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.bHidden1 == nil || nn.bHidden2 == nil || nn.bOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Complete the feed forward process.
	hiddenLayerInput1 := new(mat.Dense)
	hiddenLayerInput1.Mul(x, nn.wHidden1)
	addBHidden1 := func(_, col int, v float64) float64 { return v + nn.bHidden1.At(0, col) }
	hiddenLayerInput1.Apply(addBHidden1, hiddenLayerInput1)

	hiddenLayerActivations1 := new(mat.Dense)
	applyReLU := func(_, _ int, v float64) float64 { return relu(v) }
	hiddenLayerActivations1.Apply(applyReLU, hiddenLayerInput1)

	hiddenLayerInput2 := new(mat.Dense)
	hiddenLayerInput2.Mul(hiddenLayerActivations1, nn.wHidden2)
	addBHidden2 := func(_, col int, v float64) float64 { return v + nn.bHidden2.At(0, col) }
	hiddenLayerInput2.Apply(addBHidden2, hiddenLayerInput2)

	hiddenLayerActivations2 := new(mat.Dense)
	hiddenLayerActivations2.Apply(applyReLU, hiddenLayerInput2)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations2, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySoftmax, outputLayerInput)

	return output, nil
}

// relu implements the ReLU function
// for use in activation functions.
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// reluPrime implements the derivative
// of the ReLU function for backpropagation.
func reluPrime(_, _ int, x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// softmax implements the softmax function
// for use in activation functions.
func softmax(_, col int, v float64, output *mat.Dense) float64 {
	cols := output.RawMatrix().Cols
	max := output.At(0, col)
	for i := 1; i < cols; i++ {
		if output.At(0, i) > max {
			max = output.At(0, i)
		}
	}
	sum := 0.0
	for i := 0; i < cols; i++ {
		sum += math.Exp(output.At(0, i) - max)
	}
	return math.Exp(v-max) / sum
}

// softmaxPrime implements the derivative
// of the softmax function for backpropagation.
func softmaxPrime(x float64) float64 {
	return x * (1 - x)
}

// sumAlongAxis sums a matrix along a
// particular dimension, preserving the
// other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Open the dataset file.
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 785 // 784 inputs + 1 label

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold all the
	// float values that will eventually be
	// used to form matrices.
	inputsData := make([]float64, 28*28*len(rawCSVData))
	labelsData := make([]float64, 10*len(rawCSVData))

	// Will track the current index of matrix values.
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats.
	for idx, record := range rawCSVData {

		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Loop over the float columns.
		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 0 {
				label := int(parsedVal)
				labelsData[labelsIndex+label] = 1.0
				labelsIndex += 10
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal / 255.0 // Normalize input
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData)-1, 28*28, inputsData)
	labels := mat.NewDense(len(rawCSVData)-1, 10, labelsData)
	return inputs, labels
}
