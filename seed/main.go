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
)

type AggregatorActor struct {
	localWeights             model.LocalWeights
	remoteWeights            model.RemoteWeights
	unprocessedRemoteWeights bool
	trainingPID              *actor.PID
}

type CommunicationActor struct {
	aggregatorPID         *actor.PID
	otherCommunicationPID *actor.PID
}

func (state *AggregatorActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case *messages.LocalWeights:
		fmt.Println("Aggregator received local weights")
		if state.unprocessedRemoteWeights {
			fmt.Println("Agregator calculating avarage weights")
			state.localWeights.Weights = averageWeights(state.remoteWeights.Weights, converter.FromProtoWeights(msg.Weights), 1)
			state.unprocessedRemoteWeights = false
		} else {
			var localWeights model.LocalWeights
			localWeights.Weights = converter.FromProtoWeights(msg.Weights)
			state.localWeights = localWeights
			fmt.Println("No new remote weights")
		}
		ctx.Send(state.trainingPID, state.localWeights)
	case *messages.RemoteWeights:
		fmt.Println("Aggregator received remote weights")
		state.unprocessedRemoteWeights = true
		state.remoteWeights.Weights = converter.FromProtoWeights(msg.Weights)
	case model.TrainingPIDMsg:
		state.trainingPID = msg.TrainingPID
		fmt.Println("Aggregator received trainingActorPID")

	}
}

func (state *CommunicationActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case *messages.LocalWeights:
		if state.aggregatorPID.Address == "127.0.0.1:8081" {
			fmt.Println("seed: Communication actor received weights from training actor")
		}
		if state.aggregatorPID.Address == "127.0.0.1:8082" {
			fmt.Println("node: Communication actor received weights from training actor")
		}

		remoteWeights := &messages.RemoteWeights{
			Weights: msg.Weights,
			Id:      "nzm lol",
		}

		if state.aggregatorPID.Address == "127.0.0.1:8081" {
			fmt.Println("seed: Send weights to other systems")
		}
		if state.aggregatorPID.Address == "127.0.0.1:8082" {
			fmt.Println("node: Send weights to other systems")
		}

		//Slanje tezina ostalim clanovima klastera
		ctx.Send(state.otherCommunicationPID, remoteWeights)
		localWeights := &messages.LocalWeights{
			Weights: msg.Weights,
		}
		ctx.Send(state.aggregatorPID, localWeights)

	case *messages.RemoteWeights:
		if state.aggregatorPID.Address == "127.0.0.1:8081" {
			fmt.Println("seed: RECIEVED WEIGHTS FROM ANOTHER NODE")
		}
		if state.aggregatorPID.Address == "127.0.0.1:8082" {
			fmt.Println("node: RECIEVED WEIGHTS FROM ANOTHER NODE")
		}

		remoteWeights := &messages.RemoteWeights{
			Weights: msg.Weights,
			Id:      "nzm lol",
		}
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

	// Prosecanje Conv2Weight
	for i := range w1.Conv2Weight {
		for j := range w1.Conv2Weight[i] {
			for k := range w1.Conv2Weight[i][j] {
				for l := range w1.Conv2Weight[i][j][k] {
					avgWeights.Conv2Weight[i][j][k][l] = (w1.Conv2Weight[i][j][k][l]*float64(count) + w2.Conv2Weight[i][j][k][l]) / float64(count+1)
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

	return avgWeights
}

func main() {

	system := actor.NewActorSystem()
	rootContext := system.Root

	//config := remote.Configure("127.0.0.1", 8081)

	// Configure remote with custom gRPC options
	config := remote.Configure("127.0.0.1", 8081, remote.WithEndpointManagerBatchSize(9000000), remote.WithEndpointWriterBatchSize(9000000))

	provider := automanaged.NewWithConfig(1*time.Second, 6331, "localhost:6331")
	lookup := disthash.New()

	aggregatorProps := actor.PropsFromProducer(func() actor.Actor {
		return &AggregatorActor{unprocessedRemoteWeights: false}
	})

	aggregatorPID := rootContext.Spawn(aggregatorProps)

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

	dataloaderProps := actor.PropsFromProducer(func() actor.Actor {
		return &DataloaderActor{commActorPID: commPID, trainingPID: trainingPID}
	})
	dataloaderPropsPID := rootContext.Spawn(dataloaderProps)
	rootContext.Send(dataloaderPropsPID, LoadDataMsg{})

	rootContext.Send(aggregatorPID, model.TrainingPIDMsg{TrainingPID: trainingPID})

	select {}
}
