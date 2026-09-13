package proto

import "encoding/json"

const Version = 1

type Message struct {
	Type    string    `json:"type"`
	Version int       `json:"version"`
	Room    string    `json:"room,omitempty"`
	Node    NodeInfo  `json:"node,omitempty"`
	Model   ModelInfo `json:"model,omitempty"`
	Message string    `json:"message,omitempty"`
}

type NodeInfo struct {
	Name     string `json:"name"`
	Backend  string `json:"backend"`
	VRAMMB   int64  `json:"vram_mb"`
	RPCAddr  string `json:"rpc_addr,omitempty"`
	LlamaSHA string `json:"llama_sha,omitempty"`
}

type ModelInfo struct {
	Path   string `json:"path,omitempty"`
	SHA256 string `json:"sha256,omitempty"`
}

func Encode(msg Message) ([]byte, error) {
	return json.Marshal(msg)
}

func Decode(data []byte) (Message, error) {
	var msg Message
	err := json.Unmarshal(data, &msg)
	return msg, err
}
