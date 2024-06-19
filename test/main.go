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

type LocalWeights struct {
	weights Weights
}

type RemoteWeights struct {
	weights Weights
}

type TrainingActor struct {
	commActorPID *actor.PID
}

type AggregatorActor struct {
	localWeights LocalWeights
	count        int
	trainingPID  *actor.PID
}

type CommunicationActor struct {
	connections   map[string]net.Conn
	aggregatorPID *actor.PID
}

type TrainingPIDMsg struct {
	TrainingPID *actor.PID
}

func (state *TrainingActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case LocalWeights:
		fmt.Println("Training Actor received weights")
		fmt.Println("Training Actor weights: ", msg.weights.WeightsArray)
		time.Sleep(1 * time.Second) //Sluzi da simulira obradu, da ne bi imali rafalni ispis u konzoli
		ctx.Send(state.commActorPID, msg)
	}
}

func (state *AggregatorActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case RemoteWeights:
		fmt.Println("Aggregator received weights")

		if state.count == 0 {
			state.localWeights.weights = msg.weights
		} else {
			state.localWeights.weights = averageWeights(state.localWeights.weights, msg.weights, state.count)
		}
		state.count++
		fmt.Println("Received weights: ", msg.weights.WeightsArray)
		fmt.Println("Average weights: ", state.localWeights.weights.WeightsArray)
		time.Sleep(1 * time.Second) //Sluzi da simulira obradu, da ne bi imali rafalni ispis u konzoli
		ctx.Send(state.trainingPID, state.localWeights)
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
		randomWeights := generateRandomWeights()
		var remoteWeights RemoteWeights
		remoteWeights.weights = randomWeights
		fmt.Println("Comm send weights: ", remoteWeights.weights)
		time.Sleep(1 * time.Second) //Sluzi da simulira obradu, da ne bi imali rafalni ispis u konzoli
		ctx.Send(state.aggregatorPID, remoteWeights)

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

	trainingProps := actor.PropsFromProducer(func() actor.Actor {
		return &TrainingActor{commActorPID: commPID}
	})
	trainingPID := rootContext.Spawn(trainingProps)

	rootContext.Send(aggregatorPID, TrainingPIDMsg{TrainingPID: trainingPID})

	// Generisanje i slanje nasumičnih težina za testiranje
	for i := 0; i < 5; i++ {
		weights := generateRandomWeights()
		var localWeights LocalWeights
		localWeights.weights = weights
		fmt.Println("Main send weights: ", localWeights.weights)
		rootContext.Send(trainingPID, localWeights)
		time.Sleep(1 * time.Second)
	}

	// Dajemo malo vremena za obradu
	select {}
}
