package main

import (

    "compress/gzip"
    "encoding/binary"
    "fmt"
    "io"
    "log"
    "os"

    "gorgonia.org/tensor"
)

func readImages(filename string) (tensor.Tensor, error) {
    f, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer f.Close()

    gz, err := gzip.NewReader(f)
    if err != nil {
        return nil, err
    }
    defer gz.Close()

    var magic, numImages, rows, cols int32
    if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
        return nil, err
    }
    if magic != 2051 {
        return nil, fmt.Errorf("invalid magic number %d", magic)
    }
    if err := binary.Read(gz, binary.BigEndian, &numImages); err != nil {
        return nil, err
    }
    if err := binary.Read(gz, binary.BigEndian, &rows); err != nil {
        return nil, err
    }
    if err := binary.Read(gz, binary.BigEndian, &cols); err != nil {
        return nil, err
    }

    data := make([]byte, numImages*rows*cols)
    if _, err := io.ReadFull(gz, data); err != nil {
        return nil, err
    }

    imgTensor := tensor.New(tensor.WithShape(int(numImages), int(rows), int(cols)), tensor.Of(tensor.Uint8), tensor.WithBacking(data))
    return imgTensor, nil
}

func readLabels(filename string) (tensor.Tensor, error) {
    f, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer f.Close()

    gz, err := gzip.NewReader(f)
    if err != nil {
        return nil, err
    }
    defer gz.Close()

    var magic, numLabels int32
    if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
        return nil, err
    }
    if magic != 2049 {
        return nil, fmt.Errorf("invalid magic number %d", magic)
    }
    if err := binary.Read(gz, binary.BigEndian, &numLabels); err != nil {
        return nil, err
    }

    labels := make([]byte, numLabels)
    if _, err := io.ReadFull(gz, labels); err != nil {
        return nil, err
    }

    labelTensor := tensor.New(tensor.WithShape(int(numLabels)), tensor.Of(tensor.Uint8), tensor.WithBacking(labels))
    return labelTensor, nil
}

func main() {
    trainImages, err := readImages("dataset/train-images-idx3-ubyte.gz")
    if err != nil {
        log.Fatal(err)
    }
    trainLabels, err := readLabels("dataset/train-labels-idx1-ubyte.gz")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(trainImages.Shape())
    fmt.Println(trainLabels.Shape())
}
