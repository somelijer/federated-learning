package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"time"

	"github.com/asynkron/protoactor-go/actor"
)

// type Weights struct {
// 	WeightsArray []float64
// }

type Weights struct {
	Conv1Weight [][][][]float64 `json:"conv1_weight"`
	Conv1Bias   []float64       `json:"conv1_bias"`
	Conv2Weight [][][][]float64 `json:"conv2_weight"`
	Conv2Bias   []float64       `json:"conv2_bias"`
	Fc1Weight   [][]float64     `json:"fc1_weight"`
	Fc1Bias     []float64       `json:"fc1_bias"`
	Fc2Weight   [][]float64     `json:"fc2_weight"`
	Fc2Bias     []float64       `json:"fc2_bias"`
	Fc3Weight   [][]float64     `json:"fc3_weight"`
	Fc3Bias     []float64       `json:"fc3_bias"`
}
type LocalWeights struct {
	weights Weights
}

type RemoteWeights struct {
	weights Weights
}

type NoWeightsStart struct {
	rand int
}


type AggregatorActor struct {
	localWeights LocalWeights
	remoteWeights RemoteWeights
	unprocessedRemoteWeights bool 
	trainingPID  *actor.PID
}

type CommunicationActor struct {
	connections   map[string]net.Conn
	aggregatorPID *actor.PID
}

type TrainingPIDMsg struct {
	TrainingPID *actor.PID
}



func (state *AggregatorActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case LocalWeights:
		fmt.Println("Aggregator received local weights")
		if(state.unprocessedRemoteWeights){
			fmt.Println("Agregator calculating avarage weights")
			state.localWeights.weights = averageWeights(state.localWeights.weights, msg.weights,1)
			state.unprocessedRemoteWeights = false
		}else{
			fmt.Println("No new remote weights")
		}
		var localWeights LocalWeights
		localWeights.weights = msg.weights
		ctx.Send(state.trainingPID, localWeights)
	case RemoteWeights:
		fmt.Println("Aggregator received remote weights")
		state.unprocessedRemoteWeights = true
		state.remoteWeights.weights = msg.weights
	case TrainingPIDMsg:
		state.trainingPID = msg.TrainingPID
		fmt.Println("Aggregator received trainingActorPID")

	}
}

func (state *CommunicationActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case LocalWeights:
		fmt.Println("Communication actor received weights from training actor")
		//TODO
		//Ovde implementirati logiku za slanje tezina ostalim klasterima
		fmt.Println("Send weights to other systems")

		//Ovaj deo za sada simulira da smo dobili poruku tipa RemoteWeights dok ne implementiramo logiku za dobavljanje tezina iz ostalih klastera

		var localWeights LocalWeights
		localWeights.weights = msg.weights
		ctx.Send(state.aggregatorPID, localWeights)

	case RemoteWeights:
		var localWeights LocalWeights
		localWeights.weights = msg.weights
		ctx.Send(state.aggregatorPID, localWeights)
	}
}

func (state *CommunicationActor) broadcastWeights(weights Weights) {
	for _, conn := range state.connections {
		data, err := json.Marshal(weights)
		if err != nil {
			fmt.Println("Error marshalling weights:", err)
			continue
		}
		_, err = conn.Write(data)
		if err != nil {
			fmt.Println("Error broadcasting weights:", err)
		}
	}
}

func (state *CommunicationActor) receiveWeights(ctx actor.Context) {
	buffer := make([]byte, 4096)
	for _, conn := range state.connections {
		n, err := conn.Read(buffer)
		if err != nil {
			continue
		}
		var weights Weights
		err = json.Unmarshal(buffer[:n], &weights)
		if err != nil {
			fmt.Println("Error unmarshalling weights:", err)
			continue
		}
		fmt.Println("Weights from nodes", err)
		ctx.Send(ctx.Self(), weights)
	}
}

// func averageWeights(w1, w2 Weights, count int) Weights {
// 	for i := range w1.WeightsArray {
// 		w1.WeightsArray[i] = (w1.WeightsArray[i]*float64(count) + w2.WeightsArray[i]) / float64(count+1)
// 	}
// 	return w1
// }

func averageWeights(w1, w2 Weights, count int) Weights {
	// Kopiramo w1 u novi objekat
	avgWeights := w1

	// Prosecanje Conv1Weight
	for i := range w1.Conv1Weight {
		for j := range w1.Conv1Weight[i] {
			for k := range w1.Conv1Weight[i][j] {
				for l := range w1.Conv1Weight[i][j][k] {
					avgWeights.Conv1Weight[i][j][k][l] = (w1.Conv1Weight[i][j][k][l]*float64(count) + w2.Conv1Weight[i][j][k][l]) / float64(count+1)
				}
			}
		}
	}

	// Prosecanje Conv1Bias
	for i := range w1.Conv1Bias {
		avgWeights.Conv1Bias[i] = (w1.Conv1Bias[i]*float64(count) + w2.Conv1Bias[i]) / float64(count+1)
	}

	// Prosecanje Conv2Bias
	for i := range w1.Conv2Bias {
		avgWeights.Conv2Bias[i] = (w1.Conv2Bias[i]*float64(count) + w2.Conv2Bias[i]) / float64(count+1)
	}

	// Prosecanje Fc1Weight
	for i := range w1.Fc1Weight {
		for j := range w1.Fc1Weight[i] {
			avgWeights.Fc1Weight[i][j] = (w1.Fc1Weight[i][j]*float64(count) + w2.Fc1Weight[i][j]) / float64(count+1)
		}
	}

	// Prosecanje Fc1Bias
	for i := range w1.Fc1Bias {
		avgWeights.Fc1Bias[i] = (w1.Fc1Bias[i]*float64(count) + w2.Fc1Bias[i]) / float64(count+1)
	}

	// Prosecanje Fc2Weight
	for i := range w1.Fc2Weight {
		for j := range w1.Fc2Weight[i] {
			avgWeights.Fc2Weight[i][j] = (w1.Fc2Weight[i][j]*float64(count) + w2.Fc2Weight[i][j]) / float64(count+1)
		}
	}

	// Prosecanje Fc2Bias
	for i := range w1.Fc2Bias {
		avgWeights.Fc2Bias[i] = (w1.Fc2Bias[i]*float64(count) + w2.Fc2Bias[i]) / float64(count+1)
	}

	// Prosecanje Fc3Weight
	for i := range w1.Fc3Weight {
		for j := range w1.Fc3Weight[i] {
			avgWeights.Fc3Weight[i][j] = (w1.Fc3Weight[i][j]*float64(count) + w2.Fc3Weight[i][j]) / float64(count+1)
		}
	}

	// Prosecanje Fc3Bias
	for i := range w1.Fc3Bias {
		avgWeights.Fc3Bias[i] = (w1.Fc3Bias[i]*float64(count) + w2.Fc3Bias[i]) / float64(count+1)
	}

	return avgWeights
}

// func generateRandomWeights() Weights {
// 	rand.Seed(time.Now().UnixNano())
// 	return Weights{
// 		WeightsArray: generateRandomVector(10),
// 	}
// }

func randomWeights() Weights {
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

	return Weights{
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

func generateRandomVector(size int) []float64 {
	vector := make([]float64, size)
	for i := range vector {
		vector[i] = rand.Float64()
	}
	return vector
}

func main() {
	//peers := []string{"localhost:8001", "localhost:8002"} // primer adresa

	system := actor.NewActorSystem()
	rootContext := system.Root

	aggregatorProps := actor.PropsFromProducer(func() actor.Actor {
		return &AggregatorActor{unprocessedRemoteWeights: false}
	})
	aggregatorPID := rootContext.Spawn(aggregatorProps)

	commProps := actor.PropsFromProducer(func() actor.Actor {
		return &CommunicationActor{connections: make(map[string]net.Conn), aggregatorPID: aggregatorPID}
	})
	commPID := rootContext.Spawn(commProps)

	trainingProps := actor.PropsFromProducer(func() actor.Actor {
		return &TrainingActor{commActorPID: commPID}
	})
	trainingPID := rootContext.Spawn(trainingProps)

	dataloaderProps := actor.PropsFromProducer(func() actor.Actor {
		return &DataloaderActor{commActorPID: commPID,trainingPID: trainingPID}
	})
	dataloaderPropsPID := rootContext.Spawn(dataloaderProps)
	rootContext.Send(dataloaderPropsPID, LoadDataMsg{})

	rootContext.Send(aggregatorPID, TrainingPIDMsg{TrainingPID: trainingPID})

	// Generisanje i slanje nasumičnih težina za testiranje
	/*for i := 0; i < 5; i++ {
		weights := randomWeights()
		var localWeights LocalWeights
		localWeights.weights = weights
		//fmt.Println("Main send weights: ", localWeights.weights)
		rootContext.Send(trainingPID, localWeights)
		time.Sleep(1 * time.Second)
	}*/

	// Dajemo malo vremena za obradu
	select {}
}
