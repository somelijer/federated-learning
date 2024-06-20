package main

import (
    "encoding/binary"
    "fmt"
    "os"

    "github.com/asynkron/protoactor-go/actor"
)

var (
    imagesFilePath = "data/train-images-idx3-ubyte"
    labelsFilePath = "data/train-labels-idx1-ubyte"
)

type LoadDataMsg struct{}

type DataloaderActor struct {
    commActorPID *actor.PID
    trainingPID  *actor.PID
}

func (state *DataloaderActor) Receive(ctx actor.Context) {
    switch ctx.Message().(type) {
    case LoadDataMsg:
        images, err := readIDXFile(imagesFilePath)
        if err != nil {
            fmt.Printf("Error reading images file: %v\n", err)
            return
        }

        labels, err := readIDXFile(labelsFilePath)
        if err != nil {
            fmt.Printf("Error reading labels file: %v\n", err)
            return
        }

        numRows, numCols := 28, 28
        imageSize := numRows * numCols

        mnistData := MNISTData{
            Images: make([][]float32, 20000),
            Labels: make([]int, 20000),
        }

        for i := 0; i < 20000; i++ {
            mnistData.Images[i] = make([]float32, imageSize)
            for j := 0; j < imageSize; j++ {
                mnistData.Images[i][j] = float32(images[i*imageSize+j]) / 255.0
            }
            mnistData.Labels[i] = int(labels[i])
        }
		fmt.Printf("Loaded and sending mnist data to training actor pid: "+state.trainingPID.Id)
        ctx.Send(state.trainingPID, mnistData)
    case TrainingPIDMsg:
        state.trainingPID = ctx.Message().(TrainingPIDMsg).TrainingPID
    }
}

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
