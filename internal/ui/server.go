package ui

import (
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

//go:embed index.html
var assets embed.FS

type Status struct {
	Room        string `json:"room"`
	Model       string `json:"model"`
	Control     string `json:"control"`
	LlamaURL    string `json:"llama_url"`
	WorkerCount int    `json:"worker_count"`
	LLMReady    bool   `json:"llm_ready"`
}

type Config struct {
	Status   func() Status
	LlamaURL string
}

func NewHandler(config Config) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", serveIndex)
	mux.HandleFunc("/api/status", func(w http.ResponseWriter, _ *http.Request) {
		status := config.Status()
		status.LLMReady = llamaReady(config.LlamaURL)
		writeJSON(w, http.StatusOK, status)
	})
	mux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		proxyChat(w, r, config.LlamaURL)
	})
	return mux
}

func serveIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	data, err := assets.ReadFile("index.html")
	if err != nil {
		http.Error(w, "UI asset unavailable", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(data)
}

func proxyChat(w http.ResponseWriter, r *http.Request, llamaURL string) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if strings.TrimSpace(llamaURL) == "" {
		http.Error(w, "llama-server URL is not configured", http.StatusServiceUnavailable)
		return
	}

	body := http.MaxBytesReader(w, r.Body, 2<<20)
	request, err := http.NewRequestWithContext(
		r.Context(),
		http.MethodPost,
		strings.TrimRight(llamaURL, "/")+"/v1/chat/completions",
		body,
	)
	if err != nil {
		http.Error(w, "create LLM request: "+err.Error(), http.StatusBadGateway)
		return
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")

	response, err := (&http.Client{Timeout: 0}).Do(request)
	if err != nil {
		http.Error(w, "llama-server unavailable: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer response.Body.Close()

	for key, values := range response.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	w.WriteHeader(response.StatusCode)
	_, _ = io.Copy(w, response.Body)
}

func llamaReady(llamaURL string) bool {
	if strings.TrimSpace(llamaURL) == "" {
		return false
	}
	client := http.Client{Timeout: 750 * time.Millisecond}
	response, err := client.Get(strings.TrimRight(llamaURL, "/") + "/health")
	if err != nil {
		return false
	}
	defer response.Body.Close()
	return response.StatusCode >= http.StatusOK && response.StatusCode < http.StatusMultipleChoices
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		fmt.Fprintln(w, `{"error":"encode response"}`)
	}
}
