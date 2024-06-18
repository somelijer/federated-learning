package main

import (
	"fmt"
	"math/rand"
	"strconv"
	"time"
)

type ActorA struct {
	inbox chan string
	state int
}

type ActorB struct {
	inbox chan string
	state int
}

func NewActorA() *ActorA {
	return &ActorA{
		inbox: make(chan string),
		state: 0,
	}
}

func NewActorB() *ActorB {
	return &ActorB{
		inbox: make(chan string),
		state: 0,
	}
}

func (a *ActorA) loop(b *ActorB) {
	for {
		select {
		case msg := <-a.inbox:
			fmt.Println("ActorA primio:", msg)
			rand.Seed(time.Now().UnixNano())
			rNumber := rand.Intn(11)
			if rNumber > 5 {
				a.state += 1
			} else {
				a.state -= 1
			}

			response := "Pozdrav od AktoraA, moje trenutno stanje je " + strconv.Itoa(a.state)
			time.Sleep(1 * time.Second) // Simulacija obrade
			b.inbox <- response
		}
	}
}

func (b *ActorB) loop(a *ActorA) {
	for {
		select {
		case msg := <-b.inbox:
			fmt.Println("ActorB primio:", msg)
			rand.Seed(time.Now().UnixNano())
			rNumber := rand.Intn(11)
			if rNumber > 5 {
				b.state += 1
			} else {
				b.state -= 1
			}
			response := "Pozdrav od AktoraB, moje trenutno stanje je " + strconv.Itoa(b.state)
			time.Sleep(1 * time.Second) // Simulacija obrade
			a.inbox <- response
		}
	}
}

func (a *ActorA) Send(msg string) {
	a.inbox <- msg
}

func (b *ActorB) Send(msg string) {
	b.inbox <- msg
}

func main() {
	actorA := NewActorA()
	actorB := NewActorB()

	go actorA.loop(actorB)
	go actorB.loop(actorA)

	//Pokretanje komunikacije
	actorA.Send("Zdravo, pokreni razmenu poruka!")

	//Ovo omogucava programu da radi u beskonacnost
	select {}
}
