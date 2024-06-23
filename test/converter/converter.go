package converter

import (
	"github.com/asynkron/protoactor-go/actor"
	"main.go/messages"
	"main.go/model"
)

func ToProtoWeights(w model.Weights) *messages.Weights {

	// Convert Conv1Weight
	var protoConv1Weight []*messages.Conv1Weight
	for _, w1 := range w.Conv1Weight {
		var w1List []*messages.Conv1Weight2
		for _, w2 := range w1 {
			var w2List []*messages.Conv1Weight3
			for _, w3 := range w2 {
				for _, w4 := range w3 {
					w2List = append(w2List, &messages.Conv1Weight3{Conv1Weight: w4})
				}
			}
			w1List = append(w1List, &messages.Conv1Weight2{Conv1Weight: w2List})
		}
		protoConv1Weight = append(protoConv1Weight, &messages.Conv1Weight{Conv1Weight: w1List})
	}

	var protoConv2Weight []*messages.Conv1Weight
	for _, w1 := range w.Conv2Weight {
		var w1List []*messages.Conv1Weight2
		for _, w2 := range w1 {
			var w2List []*messages.Conv1Weight3
			for _, w3 := range w2 {
				for _, w4 := range w3 {
					w2List = append(w2List, &messages.Conv1Weight3{Conv1Weight: w4})
				}
			}
			w1List = append(w1List, &messages.Conv1Weight2{Conv1Weight: w2List})
		}
		protoConv2Weight = append(protoConv2Weight, &messages.Conv1Weight{Conv1Weight: w1List})
	}

	// Convert Fc1Weight, Fc2Weight, Fc3Weight
	convertFcWeight := func(fcWeight [][]float64) []*messages.FcWeight {
		var protoFcWeight []*messages.FcWeight
		for _, fc1 := range fcWeight {
			var fc1List []*messages.FcWeight2
			for _, fc2 := range fc1 {
				fc1List = append(fc1List, &messages.FcWeight2{Fc1Weight: fc2})
			}
			protoFcWeight = append(protoFcWeight, &messages.FcWeight{Fc1Weight: fc1List})
		}
		return protoFcWeight
	}

	return &messages.Weights{
		Conv1Weight: protoConv1Weight,
		Conv1Bias:   w.Conv1Bias,
		Conv2Weight: protoConv2Weight,
		Conv2Bias:   w.Conv2Bias,
		Fc1Weight:   convertFcWeight(w.Fc1Weight),
		Fc1Bias:     w.Fc1Bias,
		Fc2Weight:   convertFcWeight(w.Fc2Weight),
		Fc2Bias:     w.Fc2Bias,
	}
}

func FromProtoWeights(pw *messages.Weights) model.Weights {
	// Convert Conv1Weight
	var conv1Weight [][][][]float64
	for _, pw1 := range pw.Conv1Weight {
		var w1 [][][]float64
		for _, pw2 := range pw1.Conv1Weight {
			var w2 [][]float64
			for _, pw3 := range pw2.Conv1Weight {
				w2 = append(w2, []float64{pw3.Conv1Weight})
			}
			w1 = append(w1, w2)
		}
		conv1Weight = append(conv1Weight, w1)
	}


	var conv2Weight [][][][]float64
	for _, pw1 := range pw.Conv2Weight {
		var w1 [][][]float64
		for _, pw2 := range pw1.Conv1Weight {
			var w2 [][]float64
			for _, pw3 := range pw2.Conv1Weight {
				w2 = append(w2, []float64{pw3.Conv1Weight})
			}
			w1 = append(w1, w2)
		}
		conv2Weight = append(conv2Weight, w1)
	}

	// Convert Fc1Weight, Fc2Weight, Fc3Weight
	convertProtoFcWeight := func(protoFcWeight []*messages.FcWeight) [][]float64 {
		var fcWeight [][]float64
		for _, pw1 := range protoFcWeight {
			var w1 []float64
			for _, pw2 := range pw1.Fc1Weight {
				w1 = append(w1, pw2.Fc1Weight)
			}
			fcWeight = append(fcWeight, w1)
		}
		return fcWeight
	}

	return model.Weights{
		Conv1Weight: conv1Weight,
		Conv1Bias:   pw.Conv1Bias,
		Conv2Weight: conv2Weight,
		Conv2Bias:   pw.Conv2Bias,
		Fc1Weight:   convertProtoFcWeight(pw.Fc1Weight),
		Fc1Bias:     pw.Fc1Bias,
		Fc2Weight:   convertProtoFcWeight(pw.Fc2Weight),
		Fc2Bias:     pw.Fc2Bias,
	}
}

func ProtoToActorPID(protoPID *messages.PID) *actor.PID {
	return &actor.PID{
		Address:   protoPID.Address,
		Id:        protoPID.Id,
		RequestId: protoPID.RequestId,
	}
}

func ActorToProtoPID(actorPID *actor.PID) *messages.PID {
	return &messages.PID{
		Address:   actorPID.Address,
		Id:        actorPID.Id,
		RequestId: actorPID.RequestId,
	}
}
