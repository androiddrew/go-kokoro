package kokoro

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func testConfig(t *testing.T) Config {
	t.Helper()
	library, dir := os.Getenv("KOKORO_TEST_ORT"), os.Getenv("KOKORO_TEST_ASSETS")
	if library == "" || dir == "" {
		t.Skip("set KOKORO_TEST_ORT and KOKORO_TEST_ASSETS for real-model checks")
	}
	return Config{ORTLibrary: library, ModelPath: filepath.Join(dir, "kokoro-v1.0.onnx"), VoicesPath: filepath.Join(dir, "voices-v1.0.bin")}
}

func TestRealModel(t *testing.T) {
	c := testConfig(t)
	bad := c
	bad.ModelPath = filepath.Join(t.TempDir(), "broken.onnx")
	if err := os.WriteFile(bad.ModelPath, []byte("not ONNX"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(bad); err == nil {
		t.Fatal("accepted broken model")
	}
	if ort.IsInitialized() {
		t.Fatal("failed init leaked runtime")
	}
	bad = c
	bad.ORTLibrary = filepath.Join(t.TempDir(), "missing-onnxruntime.so")
	if _, err := New(bad); err == nil {
		t.Fatal("accepted missing runtime")
	}
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := e.Close(); err != nil {
			t.Error(err)
		}
	})
	if len(e.Voices()) == 0 {
		t.Fatal("no voices")
	}
	bad = c
	bad.ORTLibrary += ".other"
	if _, err := New(bad); err == nil {
		t.Fatal("accepted conflicting runtime")
	}
	for _, request := range []Request{{"hello🐈", "af_heart", 1, true}, {"həlˈO", "missing", 1, true}, {"həlˈO", "af_heart", 0, true}, {"həlˈO", "af_heart", float32(math.NaN()), true}} {
		if _, err := e.Synthesize(context.Background(), request); err == nil {
			t.Fatal("accepted invalid request")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	request := Request{"həlˈO wˈɜɹld!", "af_heart", 1, true}
	prepared, err := e.Prepare(request.Phonemes, request.Voice, request.Speed)
	if err != nil {
		t.Fatal(err)
	}
	normal, err := e.Synthesize(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	direct, err := e.SynthesizePrepared(context.Background(), prepared, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(normal.Samples, direct.Samples) {
		t.Fatal("prepared tensors changed CPU output")
	}
	if direct.Timings.InferenceSeconds <= 0 || direct.Timings.PostprocessingSeconds <= 0 || e.Initialization().ModelSeconds <= 0 {
		t.Fatal("missing timing boundaries")
	}
	for _, mutate := range []func(*Chunk){
		func(c *Chunk) { c.Tokens = nil },
		func(c *Chunk) { c.Tokens[0] = 1 },
		func(c *Chunk) { c.Tokens[1] = 100000 },
		func(c *Chunk) { c.Style = nil },
		func(c *Chunk) { c.Style[0] = float32(math.NaN()) },
		func(c *Chunk) { c.Speed = 0 },
	} {
		bad := prepared[0]
		bad.Tokens = append([]int64(nil), bad.Tokens...)
		bad.Style = append([]float32(nil), bad.Style...)
		mutate(&bad)
		if _, err := e.SynthesizePrepared(context.Background(), []Chunk{bad}, true); err == nil {
			t.Fatal("accepted malformed prepared input")
		}
	}
	if _, err := e.SynthesizePrepared(ctx, prepared, true); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := e.Synthesize(ctx, request); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	// Multiple leases: closing one engine must not invalidate the other.
	other, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Prepare(request.Phonemes, request.Voice, 1); err == nil {
		t.Fatal("use after close")
	}
	if _, err := e.Synthesize(context.Background(), request); err == nil {
		t.Fatal("synthesis after close")
	}
	if _, err := e.SynthesizePrepared(context.Background(), prepared, true); err == nil {
		t.Fatal("prepared synthesis after close")
	}
	var wg sync.WaitGroup
	for range 2 {
		wg.Go(func() {
			result, err := other.Synthesize(context.Background(), request)
			if err != nil {
				t.Error(err)
				return
			}
			if len(result.Samples) < 2400 || result.SampleRate != 24000 {
				t.Error("invalid audio")
			}
		})
	}
	wg.Wait()
	if err := other.Close(); err != nil {
		t.Fatal(err)
	}
	if ort.IsInitialized() {
		t.Fatal("last Close leaked environment")
	}
	// Successful reinitialization after all sessions, options and tensors are gone.
	again, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := again.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestModelMetadata(t *testing.T) {
	inputs := []ort.InputOutputInfo{
		{Name: "tokens", OrtValueType: ort.ONNXTypeTensor, DataType: ort.TensorElementDataTypeInt64, Dimensions: ort.NewShape(1, -1)},
		{Name: "style", OrtValueType: ort.ONNXTypeTensor, DataType: ort.TensorElementDataTypeFloat, Dimensions: ort.NewShape(1, 256)},
		{Name: "speed", OrtValueType: ort.ONNXTypeTensor, DataType: ort.TensorElementDataTypeFloat, Dimensions: ort.NewShape(1)},
	}
	outputs := []ort.InputOutputInfo{{Name: "audio", OrtValueType: ort.ONNXTypeTensor, DataType: ort.TensorElementDataTypeFloat, Dimensions: ort.NewShape(-1)}}
	if _, err := validateModel(inputs, outputs); err != nil {
		t.Fatal(err)
	}
	inputs[2].DataType = ort.TensorElementDataTypeInt32
	if _, err := validateModel(inputs, outputs); err == nil {
		t.Fatal("accepted alternate speed dtype")
	}
	inputs[2].DataType = ort.TensorElementDataTypeFloat
	inputs[0].Dimensions[1] = 512
	if _, err := validateModel(inputs, outputs); err == nil {
		t.Fatal("accepted fixed context")
	}
}

func TestCallerOwnedEnvironment(t *testing.T) {
	c := testConfig(t)
	library := c.ORTLibrary
	c.ORTLibrary = ""
	if _, err := New(c); !errors.Is(err, ErrNotInitialized) {
		t.Fatalf("uninitialized caller-owned environment: %v", err)
	}
	ort.SetSharedLibraryPath(library)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			t.Error(err)
		}
	}()
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	result, err := e.Synthesize(context.Background(), Request{Phonemes: "həlˈO", Voice: "af_heart", Speed: 1, Trim: true})
	if err != nil || len(result.Samples) == 0 {
		t.Fatalf("caller-owned synthesis: %d samples, %v", len(result.Samples), err)
	}
	if err = e.Close(); err != nil {
		t.Fatal(err)
	}
	if !ort.IsInitialized() {
		t.Fatal("Close destroyed a caller-owned environment")
	}
}
