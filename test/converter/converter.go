package converter

import (
	"github.com/asynkron/protoactor-go/actor"
	"main.go/messages"
	"main.go/model"
)

func ToProtoWeights(w model.Weights) *messages.Weights {
	// Convert Conv1Weight
	var protoConv1Weight []*messages.ConvWeight
	for _, w1 := range w.Conv1Weight {
		var kernelList []*messages.Kernel
		for _, w2 := range w1 {
			var rowList []*messages.Row
			for _, w3 := range w2 {
				var valueList []float64
				for _, w4 := range w3 {
					valueList = append(valueList, float64(w4))
				}
				rowList = append(rowList, &messages.Row{Value: valueList})
			}
			kernelList = append(kernelList, &messages.Kernel{Row: rowList})
		}
		protoConv1Weight = append(protoConv1Weight, &messages.ConvWeight{Kernel: kernelList})
	}

	// Convert Conv2Weight
	var protoConv2Weight []*messages.ConvWeight
	for _, w1 := range w.Conv2Weight {
		var kernelList []*messages.Kernel
		for _, w2 := range w1 {
			var rowList []*messages.Row
			for _, w3 := range w2 {
				var valueList []float64
				for _, w4 := range w3 {
					valueList = append(valueList, float64(w4))
				}
				rowList = append(rowList, &messages.Row{Value: valueList})
			}
			kernelList = append(kernelList, &messages.Kernel{Row: rowList})
		}
		protoConv2Weight = append(protoConv2Weight, &messages.ConvWeight{Kernel: kernelList})
	}

	// Convert Fc1Weight, Fc2Weight
	convertFcWeight := func(fcWeight [][]float64) []*messages.FcWeight {
		var protoFcWeight []*messages.FcWeight
		for _, fc1 := range fcWeight {
			var fc1List []*messages.FcWeight2
			for _, fc2 := range fc1 {
				fc1List = append(fc1List, &messages.FcWeight2{Fc1Weight: float64(fc2)})
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
		var kernelList [][][]float64
		for _, pw2 := range pw1.Kernel {
			var rowList [][]float64
			for _, pw3 := range pw2.Row {
				rowList = append(rowList, pw3.Value)
			}
			kernelList = append(kernelList, rowList)
		}
		conv1Weight = append(conv1Weight, kernelList)
	}

	// Convert Conv2Weight
	var conv2Weight [][][][]float64
	for _, pw1 := range pw.Conv2Weight {
		var kernelList [][][]float64
		for _, pw2 := range pw1.Kernel {
			var rowList [][]float64
			for _, pw3 := range pw2.Row {
				rowList = append(rowList, pw3.Value)
			}
			kernelList = append(kernelList, rowList)
		}
		conv2Weight = append(conv2Weight, kernelList)
	}

	// Convert Fc1Weight, Fc2Weight
	convertProtoFcWeight := func(protoFcWeight []*messages.FcWeight) [][]float64 {
		var fcWeight [][]float64
		for _, pw1 := range protoFcWeight {
			var fc1List []float64
			for _, pw2 := range pw1.Fc1Weight {
				fc1List = append(fc1List, pw2.Fc1Weight)
			}
			fcWeight = append(fcWeight, fc1List)
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
