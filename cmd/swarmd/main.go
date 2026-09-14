package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	neturl "net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/pranavthakur/layers/native/internal/proc"
	"github.com/pranavthakur/layers/native/internal/proto"
	"github.com/pranavthakur/layers/native/internal/room"
	"github.com/pranavthakur/layers/native/internal/ui"
)

const (
	appName        = "swarmd"
	controlPort    = 7830
	defaultBackend = "unknown"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	switch os.Args[1] {
	case "host":
		if err := hostCommand(os.Args[2:]); err != nil {
			log.Fatal(err)
		}
	case "join":
		if err := joinCommand(os.Args[2:]); err != nil {
			log.Fatal(err)
		}
	case "version":
		fmt.Printf("%s protocol=%d\n", appName, proto.Version)
	default:
		usage()
		os.Exit(2)
	}
}

func hostCommand(args []string) error {
	fs := flag.NewFlagSet("host", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	model := fs.String("model", "", "path to the GGUF model")
	llamaModel := fs.String("llama-model", "", "actual GGUF path for automatic llama-server startup")
	llamaHF := fs.String("llama-hf", "", "Hugging Face model repo[:quant] for automatic startup")
	llamaBinary := fs.String("llama-server", "build/llama/bin/llama-server", "llama-server executable")
	startLlama := fs.Bool("start-llama", true, "start llama-server after a worker joins")
	port := fs.Int("port", controlPort, "control-plane TCP port")
	uiPort := fs.Int("ui-port", 7840, "local chat UI port")
	llamaURL := fs.String("llama-url", "http://127.0.0.1:7841", "llama-server base URL")
	name := fs.String("name", hostname(), "host node name")
	backend := fs.String("backend", defaultBackend, "GPU backend label")
	vram := fs.Int64("vram-mb", 0, "available VRAM in megabytes")
	once := fs.Bool("once", false, "accept one peer and exit")
	if err := fs.Parse(args); err != nil {
		return err
	}

	code, err := room.NewCode()
	if err != nil {
		return err
	}

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		return fmt.Errorf("listen on control port %d: %w", *port, err)
	}
	defer listener.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	modelPath := *llamaModel
	if modelPath == "" && isFile(*model) {
		modelPath = *model
	}
	if modelPath != "" && *llamaHF != "" {
		return errors.New("use only one of --llama-model and --llama-hf")
	}
	modelLabel := *model
	if modelLabel == "" {
		modelLabel = firstNonEmpty(modelPath, *llamaHF)
	}
	fmt.Printf("room=%s\ncontrol=%s\nmodel=%s\n", code, listener.Addr(), valueOrDash(modelLabel))
	fmt.Println("waiting for workers...")

	state := newHostState()
	var launchMu sync.Mutex
	launchLlama := func() {
		if !*startLlama || (modelPath == "" && *llamaHF == "") {
			return
		}
		launchMu.Lock()
		defer launchMu.Unlock()
		if state.hasLlama() {
			return
		}
		nodes := state.nodes(proto.NodeInfo{
			Name:    *name,
			Backend: *backend,
			VRAMMB:  *vram,
		})
		process, err := startLlamaServer(*llamaBinary, modelPath, *llamaHF, *llamaURL, nodes)
		if err != nil {
			log.Printf("llama-server was not started: %v", err)
			return
		}
		state.setLlama(process)
		log.Printf("llama-server started with %d worker(s)", len(nodes)-1)
		go func() {
			if err := <-process.Done(); err != nil {
				log.Printf("llama-server exited: %v", err)
			}
		}()
	}
	defer func() {
		if process := state.llamaProcess(); process != nil {
			_ = process.Stop()
		}
	}()

	uiServer := &http.Server{
		Addr: fmt.Sprintf("127.0.0.1:%d", *uiPort),
		Handler: ui.NewHandler(ui.Config{
			LlamaURL: *llamaURL,
			Status: func() ui.Status {
				state.mu.RLock()
				defer state.mu.RUnlock()
				return ui.Status{
					Room:        code,
					Model:       modelLabel,
					Control:     listener.Addr().String(),
					LlamaURL:    *llamaURL,
					WorkerCount: state.workerCount,
				}
			},
		}),
	}
	go func() {
		if err := uiServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("UI unavailable on %s: %v", uiServer.Addr, err)
		}
	}()
	defer func() {
		_ = uiServer.Shutdown(context.Background())
	}()
	fmt.Printf("ui=http://%s\nllama=%s\n", uiServer.Addr, *llamaURL)

	accepted := make(chan struct{}, 1)
	go func() {
		<-ctx.Done()
		_ = listener.Close()
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return fmt.Errorf("accept worker: %w", err)
		}
		go handleWorker(ctx, conn, code, proto.NodeInfo{
			Name:    *name,
			Backend: *backend,
			VRAMMB:  *vram,
		}, modelLabel, accepted, state, launchLlama)

		if *once {
			<-accepted
			return nil
		}
	}
}

type hostState struct {
	mu          sync.RWMutex
	workerCount int
	workers     map[string]proto.NodeInfo
	llama       *proc.Process
}

func newHostState() *hostState {
	return &hostState{workers: make(map[string]proto.NodeInfo)}
}

func (s *hostState) addWorker(node proto.NodeInfo) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.workers[node.Name] = node
	s.workerCount = len(s.workers)
}

func (s *hostState) removeWorker(node proto.NodeInfo) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.workers, node.Name)
	s.workerCount = len(s.workers)
}

func (s *hostState) nodes(host proto.NodeInfo) []proto.NodeInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()
	nodes := []proto.NodeInfo{host}
	for _, worker := range s.workers {
		nodes = append(nodes, worker)
	}
	return nodes
}

func (s *hostState) hasLlama() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.llama != nil
}

func (s *hostState) setLlama(process *proc.Process) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.llama = process
}

func (s *hostState) llamaProcess() *proc.Process {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.llama
}

func handleWorker(
	ctx context.Context,
	conn net.Conn,
	code string,
	host proto.NodeInfo,
	model string,
	accepted chan<- struct{},
	state *hostState,
	onJoin func(),
) {
	defer conn.Close()
	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 1024), 64*1024)
	encoder := newEncoder(conn)

	if !scanner.Scan() {
		return
	}
	msg, err := proto.Decode(scanner.Bytes())
	if err != nil {
		send(encoder, proto.Message{Type: "reject", Version: proto.Version, Message: "invalid hello"})
		return
	}
	if msg.Type != "hello" {
		send(encoder, proto.Message{Type: "reject", Version: proto.Version, Message: "first message must be hello"})
		return
	}
	if msg.Version != proto.Version {
		send(encoder, proto.Message{Type: "reject", Version: proto.Version, Message: "protocol version mismatch"})
		return
	}
	if msg.Room != code {
		send(encoder, proto.Message{Type: "reject", Version: proto.Version, Message: "room code mismatch"})
		return
	}
	state.addWorker(msg.Node)
	defer func() {
		state.removeWorker(msg.Node)
	}()

	fmt.Printf("worker joined name=%s backend=%s vram_mb=%d rpc=%s\n",
		msg.Node.Name, msg.Node.Backend, msg.Node.VRAMMB, valueOrDash(msg.Node.RPCAddr))
	send(encoder, proto.Message{
		Type:    "welcome",
		Version: proto.Version,
		Room:    code,
		Node:    host,
		Model:   proto.ModelInfo{Path: model},
	})
	select {
	case accepted <- struct{}{}:
	default:
	}
	onJoin()

	for scanner.Scan() {
		incoming, err := proto.Decode(scanner.Bytes())
		if err != nil {
			continue
		}
		switch incoming.Type {
		case "heartbeat":
			send(encoder, proto.Message{Type: "heartbeat", Version: proto.Version})
		case "bye":
			return
		}
	}
	if err := scanner.Err(); err != nil && !errors.Is(err, net.ErrClosed) && ctx.Err() == nil {
		log.Printf("worker %s disconnected: %v", msg.Node.Name, err)
	}
}

func startLlamaServer(binary, model, hfRepo, baseURL string, nodes []proto.NodeInfo) (*proc.Process, error) {
	parsed, err := neturl.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("parse llama URL: %w", err)
	}
	if parsed.Port() == "" {
		parsed.Host = net.JoinHostPort(parsed.Hostname(), "7841")
	}
	rpcAddrs := make([]string, 0, len(nodes)-1)
	for _, node := range nodes[1:] {
		if node.RPCAddr == "" {
			return nil, fmt.Errorf("worker %q did not advertise an RPC address", node.Name)
		}
		rpcAddrs = append(rpcAddrs, node.RPCAddr)
	}
	if err := waitForRPCServers(rpcAddrs, 15*time.Second); err != nil {
		return nil, err
	}

	args := []string{"--host", "127.0.0.1",
		"--port", parsed.Port(),
		"--rpc", strings.Join(rpcAddrs, ","),
		"--split-mode", "layer",
		"-ngl", "99",
	}
	if hfRepo != "" {
		args = append([]string{"-hf", hfRepo}, args...)
	} else {
		args = append([]string{"-m", model}, args...)
	}
	if split, ok := tensorSplit(nodes); ok {
		args = append(args, "--tensor-split", split)
	}
	return proc.Start(binary, args...)
}

func waitForRPCServers(addresses []string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		allReady := true
		var lastErr error
		for _, address := range addresses {
			connection, err := net.DialTimeout("tcp", address, 500*time.Millisecond)
			if err != nil {
				allReady = false
				lastErr = fmt.Errorf("%s: %w", address, err)
				break
			}
			_ = connection.Close()
		}
		if allReady {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("worker RPC did not become ready: %w", lastErr)
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func tensorSplit(nodes []proto.NodeInfo) (string, bool) {
	var total int64
	for _, node := range nodes {
		if node.VRAMMB <= 0 {
			return "", false
		}
		total += node.VRAMMB
	}
	if total == 0 {
		return "", false
	}

	parts := make([]string, len(nodes))
	for i, node := range nodes {
		parts[i] = strconv.FormatFloat(float64(node.VRAMMB)/float64(total), 'f', 4, 64)
	}
	return strings.Join(parts, ","), true
}

func joinCommand(args []string) error {
	fs := flag.NewFlagSet("join", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	code := fs.String("code", "", "room code")
	host := fs.String("host", "", "host address or IP")
	port := fs.Int("port", controlPort, "host control-plane TCP port")
	name := fs.String("name", hostname(), "worker node name")
	backend := fs.String("backend", defaultBackend, "GPU backend label")
	vram := fs.Int64("vram-mb", 0, "available VRAM in megabytes")
	rpcServer := fs.String("rpc-server", "build/llama/bin/ggml-rpc-server", "ggml-rpc-server executable")
	rpcHost := fs.String("rpc-host", "0.0.0.0", "RPC bind host")
	rpcPort := fs.Int("rpc-port", 50052, "RPC port")
	rpcAddr := fs.String("rpc-addr", "", "address where this worker will expose llama RPC")
	noRPC := fs.Bool("no-rpc", false, "do not start ggml-rpc-server")
	once := fs.Bool("once", false, "complete the handshake and exit")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*code) == "" || strings.TrimSpace(*host) == "" {
		return errors.New("join requires --code and --host")
	}

	var (
		rpcProcess *proc.Process
		err        error
	)
	if !*noRPC {
		rpcProcess, err = proc.Start(*rpcServer,
			"--host", *rpcHost,
			"--port", strconv.Itoa(*rpcPort),
		)
		if err != nil {
			return err
		}
		defer func() {
			_ = rpcProcess.Stop()
		}()
	}
	if *rpcAddr == "" && !*noRPC {
		*rpcAddr = advertisedAddr(*host, *rpcPort)
	}

	address := net.JoinHostPort(*host, fmt.Sprintf("%d", *port))
	conn, err := net.DialTimeout("tcp", address, 5*time.Second)
	if err != nil {
		return fmt.Errorf("connect to host %s: %w", address, err)
	}
	defer conn.Close()

	encoder := newEncoder(conn)
	send(encoder, proto.Message{
		Type:    "hello",
		Version: proto.Version,
		Room:    strings.ToUpper(strings.TrimSpace(*code)),
		Node: proto.NodeInfo{
			Name:    *name,
			Backend: *backend,
			VRAMMB:  *vram,
			RPCAddr: *rpcAddr,
		},
	})

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 1024), 64*1024)
	if !scanner.Scan() {
		return errors.New("host closed connection before welcome")
	}
	reply, err := proto.Decode(scanner.Bytes())
	if err != nil {
		return fmt.Errorf("decode host response: %w", err)
	}
	if reply.Type != "welcome" {
		return fmt.Errorf("host rejected join: %s", valueOrDash(reply.Message))
	}

	fmt.Printf("joined room=%s host=%s model=%s\n",
		reply.Room, reply.Node.Name, valueOrDash(reply.Model.Path))
	if *once {
		return nil
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	fmt.Println("connected; press Ctrl-C to leave")
	for {
		select {
		case <-ctx.Done():
			send(encoder, proto.Message{Type: "bye", Version: proto.Version})
			return nil
		case <-ticker.C:
			send(encoder, proto.Message{Type: "heartbeat", Version: proto.Version})
		}
	}
}

func advertisedAddr(remoteHost string, port int) string {
	connection, err := net.DialTimeout("udp", net.JoinHostPort(remoteHost, "9"), 500*time.Millisecond)
	if err != nil {
		return net.JoinHostPort("127.0.0.1", strconv.Itoa(port))
	}
	defer connection.Close()
	local := connection.LocalAddr().(*net.UDPAddr)
	return net.JoinHostPort(local.IP.String(), strconv.Itoa(port))
}

func isFile(path string) bool {
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular()
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

type messageEncoder interface {
	Encode(any) error
}

type encoder struct {
	writer *bufio.Writer
}

func newEncoder(conn net.Conn) *encoder {
	return &encoder{writer: bufio.NewWriter(conn)}
}

func (e *encoder) Encode(value any) error {
	data, err := proto.Encode(value.(proto.Message))
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if _, err := e.writer.Write(data); err != nil {
		return err
	}
	return e.writer.Flush()
}

func send(encoder messageEncoder, msg proto.Message) {
	if err := encoder.Encode(msg); err != nil {
		log.Printf("send %s: %v", msg.Type, err)
	}
}

func hostname() string {
	name, err := os.Hostname()
	if err != nil || name == "" {
		return "unknown"
	}
	return name
}

func valueOrDash(value string) string {
	if value == "" {
		return "-"
	}
	return value
}

func usage() {
	fmt.Fprintf(os.Stderr, `layers swarmd

Usage:
  swarmd host [flags]
  swarmd join [flags]
  swarmd version

Run "swarmd host -h" or "swarmd join -h" for command options.
`)
}
