package main

import (
	"fashion_mnist_nn/mnist"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"runtime/pprof"
	"syscall"

	_ "net/http/pprof"

	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"time"

	"gopkg.in/cheggaaa/pb.v1"
)

var (
	epochs     = flag.Int("epochs", 100, "Number of epochs to train for")
	dataset    = flag.String("dataset", "train", "Which dataset to train on? Valid options are \"train\" or \"test\"")
	dtype      = flag.String("dtype", "float64", "Which dtype to use")
	batchsize  = flag.Int("batchsize", 100, "Batch size")
	cpuprofile = flag.String("cpuprofile", "", "CPU profiling")
)

const loc = "mnist/"

var dt tensor.Dtype

func parseDtype() {
	switch *dtype {
	case "float64":
		dt = tensor.Float64
	case "float32":
		dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

type feedforwardNet struct {
	g      *G.ExprGraph
	w0, w1, w2, w3 *G.Node // weights
	out    *G.Node
}

func newFeedForwardNet(g *G.ExprGraph) *feedforwardNet {
	w0 := G.NewMatrix(g, dt, G.WithShape(28*28, 512), G.WithName("w0"), G.WithInit(G.GlorotN(1.0)))
	w1 := G.NewMatrix(g, dt, G.WithShape(512, 256), G.WithName("w1"), G.WithInit(G.GlorotN(1.0)))
	w2 := G.NewMatrix(g, dt, G.WithShape(256, 128), G.WithName("w2"), G.WithInit(G.GlorotN(1.0)))
	w3 := G.NewMatrix(g, dt, G.WithShape(128, 10), G.WithName("w3"), G.WithInit(G.GlorotN(1.0)))

	return &feedforwardNet{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
		w3: w3,
	}
}

func (m *feedforwardNet) learnables() G.Nodes {
	return G.Nodes{m.w0, m.w1, m.w2, m.w3}
}

func calculateAccuracy(g *G.ExprGraph, inputs, targets tensor.Tensor, m *feedforwardNet, bs int) (float64, error) {
	numExamples := inputs.Shape()[0]
	correct := 0
	total := 0

	x := G.NewMatrix(g, dt, G.WithShape(bs, 28*28), G.WithName("x"))
	y := G.NewMatrix(g, dt, G.WithShape(bs, 10), G.WithName("y"))

	if err := m.fwd(x); err != nil {
		return 0, err
	}

	vm := G.NewTapeMachine(g)
	defer vm.Close()

	for i := 0; i < numExamples; i += bs {
		end := i + bs
		if end > numExamples {
			end = numExamples
		}

		xVal, err := inputs.Slice(G.S(i, end))
		if err != nil {
			return 0, err
		}

		yVal, err := targets.Slice(G.S(i, end))
		if err != nil {
			return 0, err
		}

		if err = xVal.(*tensor.Dense).Reshape(end-i, 28*28); err != nil {
			return 0, err
		}

		G.Let(x, xVal)
		G.Let(y, yVal)
		if err = vm.RunAll(); err != nil {
			return 0, err
		}

		predictions := m.out.Value().Data().([]float64)
		actuals := yVal.Data().([]float64)

		for j := 0; j < end-i; j++ {
			predIdx := argmax(predictions[j*10 : (j+1)*10])
			actualIdx := argmax(actuals[j*10 : (j+1)*10])
			if predIdx == actualIdx {
				correct++
			}
			total++
		}

		vm.Reset()
	}

	return float64(correct) / float64(total), nil
}

func argmax(slice []float64) int {
	maxIdx := 0
	maxVal := slice[0]
	for i, val := range slice {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}

func (m *feedforwardNet) fwd(x *G.Node) (err error) {
	var l0, a0, l1, a1, l2, a2, l3 *G.Node

	bs := x.Shape()[0]
	x = G.Must(G.Reshape(x, tensor.Shape{bs, 28*28}))

	if l0, err = G.Mul(x, m.w0); err != nil {
		return errors.Wrap(err, "Unable to multiply x and w0")
	}
	if a0, err = G.Rectify(l0); err != nil {
		return errors.Wrap(err, "Unable to activate l0")
	}

	if l1, err = G.Mul(a0, m.w1); err != nil {
		return errors.Wrap(err, "Unable to multiply a0 and w1")
	}
	if a1, err = G.Rectify(l1); err != nil {
		return errors.Wrap(err, "Unable to activate l1")
	}

	if l2, err = G.Mul(a1, m.w2); err != nil {
		return errors.Wrap(err, "Unable to multiply a1 and w2")
	}
	if a2, err = G.Rectify(l2); err != nil {
		return errors.Wrap(err, "Unable to activate l2")
	}

	if l3, err = G.Mul(a2, m.w3); err != nil {
		return errors.Wrap(err, "Unable to multiply a2 and w3")
	}

	m.out, err = G.SoftMax(l3)
	return err
}

func main() {
	flag.Parse()
	parseDtype()
	rand.Seed(1337)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	doneChan := make(chan bool, 1)

	var inputs, targets tensor.Tensor
	var err error

	if inputs, targets, err = mnist.Load("test", loc, dt); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Train inputs:", inputs.Shape())
	fmt.Println("Train data:", targets.Shape())

	numExamples := inputs.Shape()[0]
	bs := *batchsize
	if err := inputs.Reshape(numExamples, 28*28); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Train inputs reshaped:", inputs.Shape())

	g := G.NewGraph()
	x := G.NewMatrix(g, dt, G.WithShape(bs, 28*28), G.WithName("x"))
	y := G.NewMatrix(g, dt, G.WithShape(bs, 10), G.WithName("y"))
	m := newFeedForwardNet(g)
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	losses := G.Must(G.Log(G.Must(G.HadamardProd(m.out, y))))
	cost := G.Must(G.Mean(losses))
	cost = G.Must(G.Neg(cost))

	var costVal G.Value
	G.Read(cost, &costVal)

	if _, err = G.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	prog, locMap, _ := G.Compile(g)
	vm := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap), G.BindDualValues(m.learnables()...))
	solver := G.NewAdamSolver(G.WithBatchSize(float64(bs)), G.WithLearnRate(0.0001))
	defer vm.Close()

	var profiling bool
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		profiling = true
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	go cleanup(sigChan, doneChan, profiling)

	batches := numExamples / bs
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(80)

	for i := 0; i < *epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(G.S(start, end)); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = targets.Slice(G.S(start, end)); err != nil {
				log.Fatal("Unable to slice y")
			}
			if err = xVal.(*tensor.Dense).Reshape(bs, 28*28); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			G.Let(x, xVal)
			G.Let(y, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d, batch %d. Error: %v", i, b, err)
			}
			if err = solver.Step(G.NodesToValueGrads(m.learnables())); err != nil {
				log.Fatalf("Failed to update nodes with gradients at epoch %d, batch %d. Error %v", i, b, err)
			}
			vm.Reset()
			bar.Increment()
		}
		log.Printf("Epoch %d | cost %v", i, costVal)

		if i%5 == 0 {
			accuracy, err := calculateAccuracy(g, inputs, targets, m, bs)
			if err != nil {
				log.Fatalf("Failed to calculate accuracy: %v", err)
			}
			log.Printf("Epoch %d | Training Accuracy: %.2f%%", i, accuracy*100)
		}
	}
}

func cleanup(sigChan chan os.Signal, doneChan chan bool, profiling bool) {
	select {
	case <-sigChan:
		log.Println("EMERGENCY EXIT!")
		if profiling {
			log.Println("Stop profiling")
			pprof.StopCPUProfile()
		}
		os.Exit(1)

	case <-doneChan:
		return
	}
}

func handlePprof(sigChan chan os.Signal, doneChan chan bool) {
	var profiling bool
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		profiling = true
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	go cleanup(sigChan, doneChan, profiling)
}
