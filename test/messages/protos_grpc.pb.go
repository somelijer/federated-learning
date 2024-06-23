// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.3.0
// - protoc             v3.20.3
// source: protos.proto

package messages

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

const (
	Ponger_Ping_FullMethodName = "/messages.Ponger/Ping"
)

// PongerClient is the client API for Ponger service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type PongerClient interface {
	Ping(ctx context.Context, in *PingMessage, opts ...grpc.CallOption) (*PongMessage, error)
}

type pongerClient struct {
	cc grpc.ClientConnInterface
}

func NewPongerClient(cc grpc.ClientConnInterface) PongerClient {
	return &pongerClient{cc}
}

func (c *pongerClient) Ping(ctx context.Context, in *PingMessage, opts ...grpc.CallOption) (*PongMessage, error) {
	out := new(PongMessage)
	err := c.cc.Invoke(ctx, Ponger_Ping_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// PongerServer is the server API for Ponger service.
// All implementations must embed UnimplementedPongerServer
// for forward compatibility
type PongerServer interface {
	Ping(context.Context, *PingMessage) (*PongMessage, error)
	mustEmbedUnimplementedPongerServer()
}

// UnimplementedPongerServer must be embedded to have forward compatible implementations.
type UnimplementedPongerServer struct {
}

func (UnimplementedPongerServer) Ping(context.Context, *PingMessage) (*PongMessage, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Ping not implemented")
}
func (UnimplementedPongerServer) mustEmbedUnimplementedPongerServer() {}

// UnsafePongerServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to PongerServer will
// result in compilation errors.
type UnsafePongerServer interface {
	mustEmbedUnimplementedPongerServer()
}

func RegisterPongerServer(s grpc.ServiceRegistrar, srv PongerServer) {
	s.RegisterService(&Ponger_ServiceDesc, srv)
}

func _Ponger_Ping_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PingMessage)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PongerServer).Ping(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Ponger_Ping_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PongerServer).Ping(ctx, req.(*PingMessage))
	}
	return interceptor(ctx, in, info, handler)
}

// Ponger_ServiceDesc is the grpc.ServiceDesc for Ponger service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Ponger_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "messages.Ponger",
	HandlerType: (*PongerServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Ping",
			Handler:    _Ponger_Ping_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "protos.proto",
}
