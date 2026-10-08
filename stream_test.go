package kokoro

import (
	"archive/zip"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/androiddrew/go-kokoro/internal/assets"
	"github.com/androiddrew/go-kokoro/internal/audio"
)

func TestStreamDeliveryOrderAndOwnership(t *testing.T) {
	chunks := []Chunk{{Phonemes: "first"}, {Phonemes: "second"}, {Phonemes: "third"}}
	var events []string
	var delivered []ChunkAudio
	run := func(chunk Chunk) ([]float32, error) {
		events = append(events, "run "+chunk.Phonemes)
		return []float32{float32(len(events)), .5}, nil
	}
	_, err := streamChunks(context.Background(), chunks, false, run, func(chunk ChunkAudio) error {
		events = append(events, "yield "+chunk.Phonemes)
		delivered = append(delivered, chunk)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"run first", "yield first", "run second", "yield second", "run third", "yield third"}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("delivery waited for later runs: %v", events)
	}
	for i, chunk := range delivered {
		if !reflect.DeepEqual(chunk.Samples(), []float32{float32(2*i + 1), .5}) {
			t.Fatalf("chunk %d changed after later runs: %v", i, chunk.Samples())
		}
	}
	delivered[0].Samples()[0] = -1
	if delivered[0].Raw[0] != -1 || delivered[1].Raw[0] != 3 {
		t.Fatal("Samples must alias its own Raw, not another chunk")
	}
}

func TestStreamTrimming(t *testing.T) {
	raw := make([]float32, 8192)
	for i := 3072; i < 4096; i++ {
		raw[i] = .5
	}
	start, end := audio.TrimBounds(raw)
	if start == 0 || end == len(raw) {
		t.Fatal("fixture must have trimmable leading and trailing silence")
	}
	for _, trim := range []bool{false, true} {
		_, err := streamChunks(context.Background(), []Chunk{{}}, trim, func(Chunk) ([]float32, error) {
			return append([]float32(nil), raw...), nil
		}, func(chunk ChunkAudio) error {
			wantStart, wantEnd := 0, len(raw)
			if trim {
				wantStart, wantEnd = start, end
			}
			if chunk.TrimStart != wantStart || chunk.TrimEnd != wantEnd || !reflect.DeepEqual(chunk.Raw, raw) || !reflect.DeepEqual(chunk.Samples(), raw[wantStart:wantEnd]) {
				t.Fatalf("trim=%v: incorrect audio or bounds", trim)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestStreamStopsOnErrorsAndCancellation(t *testing.T) {
	runErr, callbackErr := errors.New("native failure"), errors.New("consumer failure")
	for _, test := range []struct {
		name       string
		runs, sent int
		want       error
	}{
		{"cancel before run", 0, 0, context.Canceled},
		{"cancel during run", 1, 0, context.Canceled},
		{"cancel callback", 1, 1, context.Canceled},
		{"cancel final callback", 3, 3, context.Canceled},
		{"callback error", 1, 1, callbackErr},
		{"later run error", 2, 1, runErr},
		{"empty output", 2, 1, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if test.name == "cancel before run" {
				cancel()
			}
			runs, sent := 0, 0
			var retained []float32
			_, err := streamChunks(ctx, make([]Chunk, 3), true, func(Chunk) ([]float32, error) {
				runs++
				if test.name == "cancel during run" {
					cancel()
				}
				if runs == 2 {
					switch test.name {
					case "later run error":
						return nil, runErr
					case "empty output":
						return nil, nil
					}
				}
				return []float32{.5}, nil
			}, func(chunk ChunkAudio) error {
				sent++
				retained = chunk.Samples()
				switch test.name {
				case "cancel callback":
					cancel()
				case "cancel final callback":
					if sent == 3 {
						cancel()
					}
				case "callback error":
					return callbackErr
				}
				return nil
			})
			if err == nil || test.want != nil && !errors.Is(err, test.want) {
				t.Fatalf("error = %v, want %v", err, test.want)
			}
			if runs != test.runs || sent != test.sent {
				t.Fatalf("runs=%d, sent=%d; want %d, %d", runs, sent, test.runs, test.sent)
			}
			if sent > 0 && !reflect.DeepEqual(retained, []float32{.5}) {
				t.Fatal("error invalidated previously delivered audio")
			}
			if test.name == "callback error" && !strings.Contains(err.Error(), "chunk 0 callback") || test.name == "later run error" && !strings.Contains(err.Error(), "chunk 1") {
				t.Fatalf("missing error location: %v", err)
			}
		})
	}
}

// Synthetic voice rows encode their index, so chunk length/style selection can
// be checked without downloading assets or initializing ONNX Runtime.
func streamTestEngine(t *testing.T) *Engine {
	t.Helper()
	path := filepath.Join(t.TempDir(), "voices.bin")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	z := zip.NewWriter(f)
	w, err := z.Create("af_test.npy")
	if err != nil {
		t.Fatal(err)
	}
	header := "{'descr': '<f4', 'fortran_order': False, 'shape': (510, 1, 256), }\n"
	data := make([]byte, 10+len(header)+assets.VoiceRows*assets.StyleWidth*4)
	copy(data, "\x93NUMPY\x01\x00")
	binary.LittleEndian.PutUint16(data[8:], uint16(len(header)))
	copy(data[10:], header)
	for row := range assets.VoiceRows {
		for column := range assets.StyleWidth {
			offset := 10 + len(header) + (row*assets.StyleWidth+column)*4
			binary.LittleEndian.PutUint32(data[offset:], math.Float32bits(float32(row)))
		}
	}
	if _, err := w.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := errors.Join(z.Close(), f.Close()); err != nil {
		t.Fatal(err)
	}
	voices, err := assets.LoadVoices(path)
	if err != nil {
		t.Fatal(err)
	}
	vocab, err := assets.LoadVocab("")
	if err != nil {
		t.Fatal(err)
	}
	return &Engine{voices: voices, vocab: vocab}
}

func TestStreamPreparation(t *testing.T) {
	e := streamTestEngine(t)
	options := StreamOptions{FirstChunkPhonemes: 10, MaxChunkPhonemes: 20}
	chunks, err := prepareWithOptions("həlˈO, wˈɜɹld! həlˈO əɡˈɛn.", "af_test", 1.2, e.voices, e.vocab, options)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 || chunks[0].Phonemes != "həlˈO," {
		t.Fatalf("no shorter natural opening phrase: %v", chunks)
	}
	for _, chunk := range chunks {
		length := len([]rune(chunk.Phonemes))
		if chunk.StyleIndex != length || len(chunk.Style) != assets.StyleWidth || chunk.Style[0] != float32(length) || chunk.Speed != 1.2 {
			t.Fatalf("incorrect style or speed for %q", chunk.Phonemes)
		}
		want := []int64{0}
		for _, r := range chunk.Phonemes {
			want = append(want, e.vocab[r])
		}
		want = append(want, 0)
		if !reflect.DeepEqual(chunk.Tokens, want) {
			t.Fatalf("incorrect token boundaries for %q", chunk.Phonemes)
		}
	}
}

func TestStreamRejectsInvalidRequestsBeforeDelivery(t *testing.T) {
	e := streamTestEngine(t) // no session: any native run would fail this test
	request := Request{Phonemes: "həlˈO, wˈɜɹld!", Voice: "af_test", Speed: 1}
	yield := func(ChunkAudio) error {
		t.Fatal("delivered invalid request")
		return nil
	}
	for _, test := range []struct {
		name    string
		request Request
		options StreamOptions
	}{
		{"late unknown phoneme", Request{Phonemes: "həlˈO, wˈɜɹld🐈", Voice: "af_test", Speed: 1}, StreamOptions{FirstChunkPhonemes: 6}},
		{"voice", Request{Phonemes: "həlˈO", Voice: "missing", Speed: 1}, StreamOptions{}},
		{"speed", Request{Phonemes: "həlˈO", Voice: "af_test"}, StreamOptions{}},
		{"limits", request, StreamOptions{MaxChunkPhonemes: 10, FirstChunkPhonemes: 20}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := e.SynthesizeStream(context.Background(), test.request, test.options, yield); err == nil {
				t.Fatal("accepted invalid request")
			}
		})
	}
	if _, err := e.SynthesizeStream(context.Background(), request, StreamOptions{}, nil); err == nil {
		t.Fatal("accepted nil callback")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := e.SynthesizeStream(ctx, request, StreamOptions{}, yield); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	e.closed = true
	if _, err := e.SynthesizeStream(context.Background(), request, StreamOptions{}, yield); err == nil {
		t.Fatal("streamed after close")
	}
}

func TestRealModelStream(t *testing.T) {
	c := testConfig(t)
	if !nativeSubprocess(t) {
		return
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
	ctx := context.Background()
	request := Request{Phonemes: strings.Repeat("həlˈO wˈɜɹld! ", 40), Voice: "af_heart", Speed: 1, Trim: true}
	collected, err := e.Synthesize(ctx, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(collected.Chunks) < 2 {
		t.Fatal("fixture must exercise multiple native runs")
	}
	var chunks []ChunkAudio
	var samples []float32
	timings, err := e.SynthesizeStream(ctx, request, StreamOptions{}, func(chunk ChunkAudio) error {
		chunks = append(chunks, chunk)
		samples = append(samples, chunk.Samples()...)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(chunks, collected.Chunks) || !reflect.DeepEqual(samples, collected.Samples) {
		t.Fatal("zero-option stream changed prepared inputs, raw audio, trim bounds or concatenated CPU output")
	}
	if timings.PreparationSeconds <= 0 || timings.InferenceSeconds <= 0 || timings.PostprocessingSeconds <= 0 {
		t.Fatal("missing stream timings")
	}
	request.Phonemes = strings.Repeat("həlˈO wˈɜɹld! ", 8)
	var prepared []Chunk
	samples = nil
	_, err = e.SynthesizeStream(ctx, request, StreamOptions{FirstChunkPhonemes: 20, MaxChunkPhonemes: 60}, func(chunk ChunkAudio) error {
		limit := 60
		if len(prepared) == 0 {
			limit = 20
		}
		if len(chunk.Tokens)-2 > limit {
			t.Fatalf("chunk exceeds limit %d", limit)
		}
		prepared = append(prepared, chunk.Chunk)
		samples = append(samples, chunk.Samples()...)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	direct, err := e.SynthesizePrepared(ctx, prepared, true)
	if err != nil || !reflect.DeepEqual(samples, direct.Samples) {
		t.Fatalf("stream differs from exact prepared CPU input: %v", err)
	}
	stop := errors.New("consumer stopped")
	sent := 0
	_, err = e.SynthesizeStream(ctx, request, StreamOptions{FirstChunkPhonemes: 20}, func(ChunkAudio) error {
		sent++
		return stop
	})
	if !errors.Is(err, stop) || sent != 1 {
		t.Fatalf("consumer error did not stop streaming: %v, sent %d", err, sent)
	}
	if _, err := e.Prepare("həlˈO", request.Voice, 1); err != nil {
		t.Fatalf("engine unusable after callback error: %v", err)
	}
}
