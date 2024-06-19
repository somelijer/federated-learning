package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"time"

	"github.com/asynkron/protoactor-go/actor"
)

type Weights struct {
	WeightsArray []float64
}

type TrainingActor struct {
	aggregatorPID *actor.PID
}

type AggregatorActor struct {
	weights      Weights
	count        int
	commActorPID *actor.PID
}

type CommunicationActor struct {
	connections   map[string]net.Conn
	aggregatorPID *actor.PID
}

type CommPIDMsg struct {
	CommPID *actor.PID
}

func (state *TrainingActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case Weights:
		fmt.Println("Training Actor received weights")
		time.Sleep(1 * time.Second) //Sluzi da simulira obradu, da ne bi imali rafalni ispis u konzoli
		ctx.Send(state.aggregatorPID, msg)
	}
}

func (state *AggregatorActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case Weights:
		fmt.Println("Aggregator received weights")
		time.Sleep(1 * time.Second) //Sluzi da simulira obradu, da ne bi imali rafalni ispis u konzoli
		if state.count == 0 {
			state.weights = msg
		} else {
			state.weights = averageWeights(state.weights, msg, state.count)
		}
		state.count++
		fmt.Println("Received weights: ", msg)
		fmt.Println("Average weights: ", state.weights)
		ctx.Send(state.commActorPID, state.weights)
	case CommPIDMsg:
		state.commActorPID = msg.CommPID
		fmt.Println("Aggregator received commActorPID")

	}
}

func (state *CommunicationActor) Receive(ctx actor.Context) {
	switch /*msg :=*/ ctx.Message().(type) {
	case Weights:
		fmt.Println("Communication actor received weights from aggregator")
		time.Sleep(1 * time.Second)              //Sluzi da simulira obradu, da ne bi imali rafalni ispis u konzoli
		randomWeights := generateRandomWeights() //Ovo za sada simulira tezine dobavljene iz ostalih Aktorskih sistema
		ctx.Send(state.aggregatorPID, randomWeights)

		// 	state.broadcastWeights(msg)
		// default:
		// 	state.receiveWeights(ctx)
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

func averageWeights(w1, w2 Weights, count int) Weights {
	for i := range w1.WeightsArray {
		w1.WeightsArray[i] = (w1.WeightsArray[i]*float64(count) + w2.WeightsArray[i]) / float64(count+1)
	}
	return w1
}

func generateRandomWeights() Weights {
	rand.Seed(time.Now().UnixNano())
	return Weights{
		WeightsArray: generateRandomVector(10),
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
		return &AggregatorActor{}
	})
	aggregatorPID := rootContext.Spawn(aggregatorProps)

	commProps := actor.PropsFromProducer(func() actor.Actor {
		return &CommunicationActor{connections: make(map[string]net.Conn), aggregatorPID: aggregatorPID}
	})
	commPID := rootContext.Spawn(commProps)

	rootContext.Send(aggregatorPID, CommPIDMsg{CommPID: commPID})

	trainingProps := actor.PropsFromProducer(func() actor.Actor {
		return &TrainingActor{aggregatorPID: aggregatorPID}
	})
	trainingPID := rootContext.Spawn(trainingProps)

	// Generisanje i slanje nasumičnih težina za testiranje
	for i := 0; i < 5; i++ {
		weights := generateRandomWeights()
		rootContext.Send(trainingPID, weights)
		time.Sleep(1 * time.Second)
	}

	// Dajemo malo vremena za obradu
	select {}
}
