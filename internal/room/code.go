package room

import (
	"crypto/rand"
	"fmt"
)

const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

func NewCode() (string, error) {
	const length = 6
	raw := make([]byte, length)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("generate room code: %w", err)
	}

	code := make([]byte, length)
	for i, value := range raw {
		code[i] = alphabet[int(value)%len(alphabet)]
	}
	return string(code), nil
}
