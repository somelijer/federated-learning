package main

import (
	"fmt"
	"time"

	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/cluster"
	"github.com/asynkron/protoactor-go/cluster/clusterproviders/automanaged"
	"github.com/asynkron/protoactor-go/cluster/identitylookup/disthash"
	"github.com/asynkron/protoactor-go/remote"
	"main.go/converter"
	"main.go/messages"
	"main.go/model"
	randomgenerator "main.go/random-generator"
)

type TrainingActor struct {
	commActorPID *actor.PID
}

type AggregatorActor struct {
	localWeights model.LocalWeights
	count        int
	trainingPID  *actor.PID
}

type CommunicationActor struct {
	aggregatorPID         *actor.PID
	otherCommunicationPID *actor.PID
}

type DataloaderActor struct {
	commActorPID *actor.PID
	TrainingPID  *actor.PID
}

type TrainingPIDMsg struct {
	TrainingPID *actor.PID
}

func (state *TrainingActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case model.LocalWeights:
		fmt.Println("Training Actor received weights")
		time.Sleep(2 * time.Second) //Sluzi da simulira obradu, da ne bi imali rafalni ispis u konzoli
		localWeightsMessage := &messages.LocalWeights{
			Weights: converter.ToProtoWeights(msg.Weights),
		}
		ctx.Send(state.commActorPID, localWeightsMessage)
	}
}

func (state *AggregatorActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case *messages.RemoteWeights:
		fmt.Println("Aggregator received weights")

		if state.count == 0 {
			state.localWeights.Weights = converter.FromProtoWeights(msg.Weights)
		} else {
			state.localWeights.Weights = averageWeights(state.localWeights.Weights, converter.FromProtoWeights(msg.Weights), state.count)
		}
		state.count++
		fmt.Println("Received weights: ", converter.FromProtoWeights(msg.Weights))
		fmt.Println("Average weights: ", state.localWeights.Weights)
		time.Sleep(1 * time.Second) //Sluzi da simulira obradu, da ne bi imali rafalni ispis u konzoli
		ctx.Send(state.trainingPID, state.localWeights)
	case model.TrainingPIDMsg:
		state.trainingPID = msg.TrainingPID
		fmt.Println("Aggregator received trainingActorPID")

	}
}

func (state *CommunicationActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case *messages.LocalWeights:
		fmt.Println("Communication actor received weights from training actor")

		remoteWeights := &messages.RemoteWeights{
			Weights: msg.Weights,
		}

		time.Sleep(1 * time.Second)
		//Slanje tezina ostalim clanovima klastera
		ctx.Send(state.otherCommunicationPID, remoteWeights)

		fmt.Println("Send weights to other systems")

	case *messages.RemoteWeights:
		fmt.Println("RECIEVED WEIGHTS FRON ANOTHER NODE")
		remoteWeights := &messages.RemoteWeights{
			Weights: msg.Weights,
		}
		time.Sleep(1 * time.Second)
		ctx.Send(state.aggregatorPID, remoteWeights)

	case *messages.AggregatorPIDMsg:
		state.aggregatorPID = converter.ProtoToActorPID(msg.AggregatorPID)
	case *messages.OtherCommunicationPIDMsg:
		state.otherCommunicationPID = converter.ProtoToActorPID(msg.OtherCommPID)
	}
}

func averageWeights(w1, w2 model.Weights, count int) model.Weights {
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

func main() {

	system := actor.NewActorSystem()
	rootContext := system.Root

	config := remote.Configure("127.0.0.1", 8081)

	provider := automanaged.NewWithConfig(1*time.Second, 6331, "localhost:6331")
	lookup := disthash.New()

	aggregatorProps := actor.PropsFromProducer(func() actor.Actor {
		return &AggregatorActor{}
	})

	aggregatorPID := rootContext.Spawn(aggregatorProps)
	// aggregatorPID, err := rootContext.SpawnNamed(aggregatorProps, "first-agregator")
	// if err != nil {
	// 	fmt.Printf("Failed to spawn actor: %v", err)
	// 	return
	// }

	aggregatorPID.Address = "127.0.0.1:8081"

	clusterKind := cluster.NewKind(
		"Ponger",
		actor.PropsFromProducer(func() actor.Actor {
			return &CommunicationActor{}
		}))

	clusterConfig := cluster.Configure("cluster-example", provider, lookup, config, cluster.WithKinds(clusterKind))
	c := cluster.New(system, clusterConfig)

	c.StartMember()
	defer c.Shutdown(false)

	commPID := cluster.GetCluster(system).Get("ponger-1", "Ponger")

	otherCommPID := cluster.GetCluster(system).Get("ponger-2", "Ponger")

	aggregatorPIDMsg := &messages.AggregatorPIDMsg{
		AggregatorPID: converter.ActorToProtoPID(aggregatorPID),
	}

	otherCommunicationPIDMsg := &messages.OtherCommunicationPIDMsg{
		OtherCommPID: converter.ActorToProtoPID(otherCommPID),
	}

	rootContext.Send(commPID, aggregatorPIDMsg)
	rootContext.Send(commPID, otherCommunicationPIDMsg)

	trainingProps := actor.PropsFromProducer(func() actor.Actor {
		return &TrainingActor{commActorPID: commPID}
	})
	trainingPID := rootContext.Spawn(trainingProps)

	rootContext.Send(aggregatorPID, model.TrainingPIDMsg{TrainingPID: trainingPID})

	// Generisanje i slanje nasumičnih težina za testiranje
	for i := 0; i < 20; i++ {
		weights := randomgenerator.RandomWeights()
		var localWeights model.LocalWeights
		localWeights.Weights = weights
		//fmt.Println("Main send weights: ", localWeights.weights)
		rootContext.Send(trainingPID, localWeights)
		time.Sleep(1 * time.Second)
	}

	// Dajemo malo vremena za obradu
	select {}
}

// func getPort() int {
// 	if len(os.Args) > 1 {
// 		port, err := strconv.Atoi(os.Args[1])
// 		if err != nil {
// 			log.Fatalf("Invalid port number: %s", os.Args[1])
// 		}
// 		return port
// 	}
// 	return 8081 // Default port for node instances
// }

// func getClusterPort() int {
// 	if len(os.Args) > 2 {
// 		port, err := strconv.Atoi(os.Args[2])
// 		if err == nil {
// 			return port
// 		}
// 	}
// 	return 6331 // Default cluster port for node instances
// }
