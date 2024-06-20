package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"sync"
	"time"

	"github.com/asynkron/protoactor-go/actor"
)

type MNISTData struct {
    Images [][]float32
    Labels []int
}

// MNISTResponse represents the structure of the JSON response
type MNISTResponse struct {
	Images [][]float32 `json:"images"`
	Labels []int       `json:"labels"`
}

type TrainingActor struct {
    commActorPID *actor.PID
}

var mnistData *MNISTData = nil
var weights *Weights = nil



func (state *TrainingActor) Receive(ctx actor.Context) {
	
    switch msg := ctx.Message().(type) {
    case MNISTData:
        fmt.Println("Training Actor received MNIST data")
		mnistData = &msg
		fmt.Println("Training Actor starting with random weights")
		var wg sync.WaitGroup

		wg.Add(2)
	
		go startServer(&wg)
		go runPythonScript(&wg)
	
		wg.Wait()
		fmt.Println("Both server and Python script have finished executing.")
        //ctx.Send(state.commActorPID, msg)
    case LocalWeights:
        fmt.Println("Training Actor received weights")
        // Process received weights
        time.Sleep(1 * time.Second) // Simulate processing time
        ctx.Send(state.commActorPID, msg)
	case NoWeightsStart:
		fmt.Println("Training Actor starting with random weights")
		var wg sync.WaitGroup

		wg.Add(2)
	
		go startServer(&wg)
		go runPythonScript(&wg)
	
		wg.Wait()
		fmt.Println("Both server and Python script have finished executing.")
    }
}

func runPythonScript(wg *sync.WaitGroup) {
	defer wg.Done()
	cmd := exec.Command("python", "train.py")

	// Combine stdout and stderr
	cmd.Stdout = log.Writer()
	cmd.Stderr = log.Writer()

	// Run the command
	err := cmd.Run()
	if err != nil {
		log.Fatalf("cmd.Run() failed with %s\n", err)
	}
	fmt.Printf("Python script finished executing")
}

func handleMNISTDataRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if mnistData == nil {
		http.Error(w, "No MNIST data available", http.StatusNotFound)
		return
	}

	mnistResponse := MNISTResponse{
		Images: mnistData.Images,
		Labels: mnistData.Labels,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(mnistResponse); err != nil {
		http.Error(w, fmt.Sprintf("Error encoding response: %v", err), http.StatusInternalServerError)
		return
	}

	fmt.Printf("Sending MNIST data for training")
}

func handleInitialWeightsRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if weights == nil {
		http.Error(w, "No weights available", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(weights); err != nil {
		http.Error(w, fmt.Sprintf("Error encoding weights: %v", err), http.StatusInternalServerError)
		return
	}
}

func handleWeightsRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var tempa Weights
	if err := json.NewDecoder(r.Body).Decode(&tempa); err != nil {
		http.Error(w, fmt.Sprintf("Error decoding weights: %v", err), http.StatusBadRequest)
		return
	}

	weights = &tempa

	//fmt.Println("conv1 bias: ",tempa.Conv1Bias)

	fmt.Println("Weights received successfully")
	w.WriteHeader(http.StatusOK)
}

func startServer(wg *sync.WaitGroup) {
	defer wg.Done()
	http.HandleFunc("/mnist_data", handleMNISTDataRequest)
	http.HandleFunc("/weights", handleWeightsRequest)
	http.HandleFunc("/initial_weights", handleInitialWeightsRequest)

	port := ":8080"
	fmt.Printf("Server listening on port %s...\n", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatal(err)
	}
}


