package main

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"encoding/binary"
	"os"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)



//++++++++++++++++++++++++++++++++++++++
func readIDXFile(filename string) []byte {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Read magic number
	var magicNumber int32
	if err := binary.Read(file, binary.BigEndian, &magicNumber); err != nil {
		log.Fatal("could not read magic number:", err)
	}
	if magicNumber != 2049 && magicNumber != 2051 {
		log.Fatalf("unexpected magic number %d", magicNumber)
	}

	// Read number of items
	var numItems int32
	if err := binary.Read(file, binary.BigEndian, &numItems); err != nil {
		log.Fatal("could not read number of items:", err)
	}

	// Read label or image data
	switch magicNumber {
	case 2049: // Labels file
		labels := make([]byte, numItems)
		if _, err := file.Read(labels); err != nil {
			log.Fatal("could not read labels data:", err)
		}
		return labels
	case 2051: // Images file
		var numRows, numCols int32
		if err := binary.Read(file, binary.BigEndian, &numRows); err != nil {
			log.Fatal("could not read number of rows:", err)
		}
		if err := binary.Read(file, binary.BigEndian, &numCols); err != nil {
			log.Fatal("could not read number of columns:", err)
		}

		imageSize := numRows * numCols
		images := make([]byte, numItems*imageSize)
		if _, err := file.Read(images); err != nil {
			log.Fatal("could not read images data:", err)
		}
		return images
	default:
		log.Fatalf("unknown magic number %d", magicNumber)
	}

	return nil
}

// loadMNISTDataset loads the MNIST dataset from IDX files and returns images and labels.
func loadMNISTDataset(imagesFile, labelsFile string) ([][]float64, []int) {
	// Read images and labels from IDX files
	imagesData := readIDXFile(imagesFile)
	labelsData := readIDXFile(labelsFile)

	// Prepare the dataset
	numItems := len(labelsData)
	numRows := 28
	numCols := 28
	imageSize := numRows * numCols

	images := make([][]float64, numItems)
	labels := make([]int, numItems)

	// Process images and labels
	for i := 0; i < numItems; i++ {
		images[i] = make([]float64, imageSize)
		labels[i] = int(labelsData[i])

		for j := 0; j < imageSize; j++ {
			images[i][j] = float64(imagesData[i*imageSize+j]) / 255.0 // Normalize pixel values
		}
	}

	return images, labels
}



//++++++++++++++++++++++++++++++++++++++


	type neuralNet struct {
		config  neuralNetConfig
		wHidden *mat.Dense
		bHidden *mat.Dense
		wOut    *mat.Dense
		bOut    *mat.Dense
	}
	
	// neuralNetConfig defines our neural network
	// architecture and learning parameters.
	type neuralNetConfig struct {
		inputNeurons  int
		outputNeurons int
		hiddenNeurons int
		numEpochs     int
		learningRate  float64
	}
	
func main() {
	
	// Form the training matrices.
	trainImagesFile := "data/train-images-idx3-ubyte"
	trainLabelsFile := "data/train-labels-idx1-ubyte"
	testImagesFile := "data/t10k-images-idx3-ubyte"
	testLabelsFile := "data/t10k-labels-idx1-ubyte"

	// Load training and test datasets
	trainImages, trainLabels := loadMNISTDataset(trainImagesFile, trainLabelsFile)
	testImages, testLabels := loadMNISTDataset(testImagesFile, testLabelsFile)

	// Example usage
	fmt.Println("Training images count:", len(trainImages))
	fmt.Println("Training labels count:", len(trainLabels))
	fmt.Println("Test images count:", len(testImages))
	fmt.Println("Test labels count:", len(testLabels))


	
		// Define our network architecture and learning parameters.
 		config := neuralNetConfig{
			inputNeurons:  784,
			outputNeurons: 10,
			hiddenNeurons: 625,
			numEpochs:     3,
			learningRate:  0.02,
		} 

		//trainInputs := mat.NewDense(len(trainImages), 28*28, nil)
		//trainLabelsMat := mat.NewDense(len(trainLabels), 10, nil)
	
		// Train the neural network.
		network := newNetwork(config)
		if err := network.train(trainImages, trainLabels); err != nil {
			log.Fatal(err)
		}
	

//----------------------------------------
		fmt.Println("Lenght of data: %d",len(testLabels))

		numItems := len(testLabels)
		oneHotEncoded := mat.NewDense(numItems, 10, nil)
	
		// Iterate over each element of y to create the one-hot encoded vector
		for i, label := range testLabels {
			// Set the appropriate element in the current row to 1
			oneHotEncoded.Set(i, label, 1.0)
		}
		

		cols := len(testImages[0])
		input := mat.NewDense(numItems, cols, nil)

		fmt.Println("Lenght of input: %d,%d",len(testLabels),cols)

		// Populate the dense matrix with data from x
		for i := 0; i < numItems; i++ {
			for j := 0; j < cols; j++ {
				input.Set(i, j, testImages[i][j])
			}
		}

//----------------------------------------

		//Make the predictions using the trained model.
		predictions, err := network.predict(input)
		if err != nil {
			log.Fatal(err)
		}
	
		// Calculate the accuracy of our model.
		var truePosNeg int
		numPreds, _ := predictions.Dims()
		for i := 0; i < numPreds; i++ {

			// Get the label.
			labelRow := mat.Row(nil, i, oneHotEncoded)
			var prediction int
			for idx, label := range labelRow {
				if label == 1.0 {
					prediction = idx
					break
				}
			}

			// Accumulate the true positive/negative count.
			if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
				fmt.Println(mat.Row(nil, i, predictions))
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

	func matToSlice(m *mat.Dense) []int {
		_, cols := m.Dims()
		result := make([]int, cols)
	
		for i := 0; i < cols; i++ {
			result[i] = int(m.At(0, i))
		}
	
		return result
	}
	
	// train trains a neural network using backpropagation.
	func (nn *neuralNet) train(x [][]float64, y []int) error {

		fmt.Println("Lenght of data: %d",len(y))

		numItems := len(y)
		oneHotEncoded := mat.NewDense(numItems, 10, nil)
	
		// Iterate over each element of y to create the one-hot encoded vector
		for i, label := range y {
			// Set the appropriate element in the current row to 1
			oneHotEncoded.Set(i, label, 1.0)
		}
		

		cols := len(x[0])
		input := mat.NewDense(numItems, cols, nil)

		fmt.Println("Lenght of input: %d,%d",len(y),cols)

		// Populate the dense matrix with data from x
		for i := 0; i < numItems; i++ {
			for j := 0; j < cols; j++ {
				input.Set(i, j, x[i][j])
			}
		}
	
		// Initialize biases/weights.
		randSource := rand.NewSource(time.Now().UnixNano())
		randGen := rand.New(randSource)
	
		wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
		bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
		wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
		bOut := mat.NewDense(1, nn.config.outputNeurons, nil)
	
		wHiddenRaw := wHidden.RawMatrix().Data
		bHiddenRaw := bHidden.RawMatrix().Data
		wOutRaw := wOut.RawMatrix().Data
		bOutRaw := bOut.RawMatrix().Data
	
		for _, param := range [][]float64{
			wHiddenRaw,
			bHiddenRaw,
			wOutRaw,
			bOutRaw,
		} {
			for i := range param {
				param[i] = randGen.Float64()
			}
		}
	
		// Define the output of the neural network.
		output := new(mat.Dense)

		fmt.Print(output)
	
		// Use backpropagation to adjust the weights and biases.
		if err := nn.backpropagate(input, oneHotEncoded, wHidden, bHidden, wOut, bOut, output); err != nil {
			return err
		}
	
		// Define our trained neural network.
		nn.wHidden = wHidden
		nn.bHidden = bHidden
		nn.wOut = wOut
		nn.bOut = bOut
	
		return nil
	}
	
	// backpropagate completes the backpropagation method.
	func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {
	
		// Loop over the number of epochs utilizing
		// backpropagation to train our model.
		for i := 0; i < nn.config.numEpochs; i++ {

			fmt.Println("Epoch number: ",i)
			// Complete the feed forward process.
			hiddenLayerInput := new(mat.Dense)
			hiddenLayerInput.Mul(x, wHidden)
			addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
			hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)
	
			hiddenLayerActivations := new(mat.Dense)
			applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
			hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)
	
			outputLayerInput := new(mat.Dense)
			outputLayerInput.Mul(hiddenLayerActivations, wOut)
			addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
			outputLayerInput.Apply(addBOut, outputLayerInput)
			output.Apply(applySigmoid, outputLayerInput)
	
			// Complete the backpropagation.
			networkError := new(mat.Dense)
			networkError.Sub(y, output)
	
			slopeOutputLayer := new(mat.Dense)
			applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
			slopeOutputLayer.Apply(applySigmoidPrime, output)
			slopeHiddenLayer := new(mat.Dense)
			slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)
	
			dOutput := new(mat.Dense)
			dOutput.MulElem(networkError, slopeOutputLayer)
			errorAtHiddenLayer := new(mat.Dense)
			errorAtHiddenLayer.Mul(dOutput, wOut.T())
	
			dHiddenLayer := new(mat.Dense)
			dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)
	
			// Adjust the parameters.
			wOutAdj := new(mat.Dense)
			wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
			wOutAdj.Scale(nn.config.learningRate, wOutAdj)
			wOut.Add(wOut, wOutAdj)
	
			bOutAdj, err := sumAlongAxis(0, dOutput)
			if err != nil {
				return err
			}
			bOutAdj.Scale(nn.config.learningRate, bOutAdj)
			bOut.Add(bOut, bOutAdj)
	
			wHiddenAdj := new(mat.Dense)
			wHiddenAdj.Mul(x.T(), dHiddenLayer)
			wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
			wHidden.Add(wHidden, wHiddenAdj)
	
			bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
			if err != nil {
				return err
			}
			bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
			bHidden.Add(bHidden, bHiddenAdj)
		}
	
		return nil
	}
	
	// predict makes a prediction based on a trained
	// neural network.
	func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
	
		// Check to make sure that our neuralNet value
		// represents a trained model.
		if nn.wHidden == nil || nn.wOut == nil {
			return nil, errors.New("the supplied weights are empty")
		}
		if nn.bHidden == nil || nn.bOut == nil {
			return nil, errors.New("the supplied biases are empty")
		}
	
		// Define the output of the neural network.
		output := new(mat.Dense)
	
		// Complete the feed forward process.
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, nn.wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)
	
		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)
	
		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
		addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)
	
		return output, nil
	}
	
	// sigmoid implements the sigmoid function
	// for use in activation functions.
	func sigmoid(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	
	// sigmoidPrime implements the derivative
	// of the sigmoid function for backpropagation.
	func sigmoidPrime(x float64) float64 {
		return sigmoid(x) * (1.0 - sigmoid(x))
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