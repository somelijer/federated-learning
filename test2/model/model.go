package model

import (
	"github.com/asynkron/protoactor-go/actor"
)

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
	Weights Weights
}

type RemoteWeights struct {
	Weights Weights
}

type TrainingPIDMsg struct {
	TrainingPID *actor.PID
}

type AggregatorPIDMsg struct {
	AggregatorPID *actor.PID
}

type OtherCommunicationPIDMsg struct {
	OtherCommPID *actor.PID
}
