package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
)

var (
	imagesFilePath = "data/train-images-idx3-ubyte"
	labelsFilePath = "data/train-labels-idx1-ubyte"
)

// MNISTResponse represents the structure of the JSON response
type MNISTResponse struct {
	Images [][]float32 `json:"images"`
	Labels []int       `json:"labels"`
}

// readIDXFile reads and returns the content of an IDX file.
func readIDXFile(filename string) ([]byte, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magicNumber int32
	if err := binary.Read(file, binary.BigEndian, &magicNumber); err != nil {
		return nil, fmt.Errorf("could not read magic number: %v", err)
	}

	var numItems int32
	if err := binary.Read(file, binary.BigEndian, &numItems); err != nil {
		return nil, fmt.Errorf("could not read number of items: %v", err)
	}

	switch magicNumber {
	case 2049:
		labels := make([]byte, numItems)
		if _, err := file.Read(labels); err != nil {
			return nil, fmt.Errorf("could not read labels data: %v", err)
		}
		return labels, nil
	case 2051:
		var numRows, numCols int32
		if err := binary.Read(file, binary.BigEndian, &numRows); err != nil {
			return nil, fmt.Errorf("could not read number of rows: %v", err)
		}
		if err := binary.Read(file, binary.BigEndian, &numCols); err != nil {
			return nil, fmt.Errorf("could not read number of columns: %v", err)
		}
		imageSize := numRows * numCols
		images := make([]byte, numItems*imageSize)
		if _, err := file.Read(images); err != nil {
			return nil, fmt.Errorf("could not read images data: %v", err)
		}
		return images, nil
	default:
		return nil, fmt.Errorf("unknown magic number %d", magicNumber)
	}
}

func handleMNISTDataRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	count := r.URL.Query().Get("count")
	if count == "" {
		count = "20000"
	}
	var numItems int
	if _, err := fmt.Sscanf(count, "%d", &numItems); err != nil || numItems <= 0 || numItems > 60000 {
		http.Error(w, "Invalid 'count' query parameter", http.StatusBadRequest)
		return
	}

	images, err := readIDXFile(imagesFilePath)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error reading images file: %v", err), http.StatusInternalServerError)
		return
	}

	labels, err := readIDXFile(labelsFilePath)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error reading labels file: %v", err), http.StatusInternalServerError)
		return
	}

	numRows, numCols := 28, 28
	imageSize := numRows * numCols

	mnistResponse := MNISTResponse{
		Images: make([][]float32, numItems),
		Labels: make([]int, numItems),
	}

	for i := 0; i < numItems; i++ {
		mnistResponse.Images[i] = make([]float32, imageSize)
		for j := 0; j < imageSize; j++ {
			mnistResponse.Images[i][j] = float32(images[i*imageSize+j]) / 255.0
		}
		mnistResponse.Labels[i] = int(labels[i])
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(mnistResponse); err != nil {
		http.Error(w, fmt.Sprintf("Error encoding response: %v", err), http.StatusInternalServerError)
		return
	}
}

func main() {
	http.HandleFunc("/mnist_data", handleMNISTDataRequest)

	port := ":8080"
	fmt.Printf("Server listening on port %s...\n", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatal(err)
	}
}
