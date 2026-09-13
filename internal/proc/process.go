package proc

import (
	"fmt"
	"os"
	"os/exec"
	"sync"
)

type Process struct {
	cmd  *exec.Cmd
	done chan error
	once sync.Once
}

func Start(binary string, args ...string) (*Process, error) {
	cmd := exec.Command(binary, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start %s: %w", binary, err)
	}

	process := &Process{cmd: cmd, done: make(chan error, 1)}
	go func() {
		process.done <- cmd.Wait()
	}()
	return process, nil
}

func (p *Process) Done() <-chan error {
	return p.done
}

func (p *Process) Stop() error {
	var stopErr error
	p.once.Do(func() {
		if p.cmd.Process == nil {
			return
		}
		stopErr = p.cmd.Process.Kill()
		<-p.done
	})
	return stopErr
}
