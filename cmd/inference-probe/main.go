// inference-probe synthesizes phoneme requests and records inputs and raw audio.
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"time"

	kokoro "github.com/androiddrew/go-kokoro"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() (err error) {
	var config kokoro.Config
	var input, output string
	flag.StringVar(&config.ORTLibrary, "ort", "", "ONNX Runtime library path or OS-loader name")
	flag.StringVar(&config.ModelPath, "model", "", "Kokoro v1.0 model")
	flag.StringVar(&config.VoicesPath, "voices", "", "voices-v1.0.bin NPZ archive")
	flag.StringVar(&config.VocabPath, "vocab", "", "vocabulary JSON (default: embedded Kokoro v1.0)")
	flag.StringVar((*string)(&config.Provider), "provider", "cpu", "cpu or cuda")
	flag.BoolVar(&config.Verbose, "verbose", false, "ORT node-placement diagnostics")
	flag.StringVar(&input, "input", "", "JSON request array with unique ids")
	flag.StringVar(&output, "output", "", "new output directory (parent must exist)")
	flag.Parse()
	if input == "" || output == "" || flag.NArg() != 0 {
		return errors.New("provide -input and -output, plus model/voices/ort paths")
	}
	data, err := os.ReadFile(input)
	if err != nil {
		return err
	}
	var requests []struct {
		ID string `json:"id"`
		kokoro.Request
	}
	if err = json.Unmarshal(data, &requests); err != nil {
		return err
	}
	seen := make(map[string]bool)
	for _, r := range requests {
		if !regexp.MustCompile(`^[a-zA-Z0-9_-]+$`).MatchString(r.ID) || seen[r.ID] {
			return fmt.Errorf("invalid/duplicate case ID %q", r.ID)
		}
		seen[r.ID] = true
	}
	if len(requests) == 0 {
		return errors.New("empty request list")
	}
	if err = os.Mkdir(output, 0755); err != nil {
		return err
	}
	start := time.Now()
	engine, err := kokoro.New(config)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, engine.Close()) }()
	initSeconds := time.Since(start).Seconds()
	type caseResult struct {
		ID string `json:"id"`
		kokoro.Result
		Seconds float64 `json:"seconds"`
		Samples int     `json:"samples"`
	}
	results := []caseResult{}
	for _, request := range requests {
		start = time.Now()
		result, err := engine.Synthesize(context.Background(), request.Request)
		if err != nil {
			return fmt.Errorf("case %s: %w", request.ID, err)
		}
		seconds := time.Since(start).Seconds()
		if err := writeFile(filepath.Join(output, request.ID+".wav"), func(f *os.File) error { return kokoro.WriteWAV(f, result.Samples) }); err != nil {
			return err
		}
		for i, chunk := range result.Chunks {
			if err := writeFile(filepath.Join(output, fmt.Sprintf("%s-%d.f32", request.ID, i)), func(f *os.File) error { return binary.Write(f, binary.LittleEndian, chunk.Raw) }); err != nil {
				return err
			}
		}
		results = append(results, caseResult{request.ID, result, seconds, len(result.Samples)})
	}
	report := struct {
		Provider    kokoro.Provider  `json:"provider"`
		Model       kokoro.ModelInfo `json:"model"`
		Voices      []string         `json:"voices"`
		InitSeconds float64          `json:"initialization_seconds"`
		Cases       []caseResult     `json:"cases"`
	}{config.Provider, engine.ModelInfo(), engine.Voices(), initSeconds, results}
	return writeFile(filepath.Join(output, "report.json"), func(f *os.File) error { enc := json.NewEncoder(f); enc.SetIndent("", "  "); return enc.Encode(report) })
}

func writeFile(path string, write func(*os.File) error) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	return errors.Join(write(f), f.Close())
}
