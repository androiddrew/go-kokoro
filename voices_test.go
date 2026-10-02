package kokoro

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/androiddrew/go-kokoro/internal/assets"
	ort "github.com/yalue/onnxruntime_go"
)

func TestListVoicesWithoutRuntime(t *testing.T) {
	dir := os.Getenv("KOKORO_TEST_ASSETS")
	if dir == "" {
		t.Skip("set KOKORO_TEST_ASSETS for voice archive integration")
	}
	if ort.IsInitialized() {
		t.Fatal("test requires no live native environment")
	}
	names, err := ListVoices(filepath.Join(dir, "voices-v1.0.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.IsSorted(names) || !slices.Contains(names, "af_heart") || !slices.Contains(names, "bf_emma") {
		t.Fatalf("unexpected voice inventory: %v", names)
	}
	if ort.IsInitialized() {
		t.Fatal("voice listing initialized native runtime")
	}
	if _, err := ListVoices(filepath.Join(t.TempDir(), "missing.bin")); err == nil {
		t.Fatal("accepted missing voice archive")
	}
}

// TestPrepareFixtures checks the exact token, style and speed inputs Prepare
// builds for recorded phoneme strings (phonemized by go-g2p's neural and eSpeak
// fallbacks). It needs only the voice archive, not ONNX Runtime or the model.
func TestPrepareFixtures(t *testing.T) {
	dir := os.Getenv("KOKORO_TEST_ASSETS")
	if dir == "" {
		t.Skip("set KOKORO_TEST_ASSETS for voice archive integration")
	}
	voices, err := assets.LoadVoices(filepath.Join(dir, "voices-v1.0.bin"))
	if err != nil {
		t.Fatal(err)
	}
	vocab, err := assets.LoadVocab("")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"neural", "espeak"} {
		data, err := os.ReadFile("testdata/" + name + ".json")
		if err != nil {
			t.Fatal(err)
		}
		var fixtures []struct {
			Phonemes, Voice string
			Chunks          []struct {
				Phonemes   string
				Tokens     [][]int64
				StyleIndex int `json:"style_index"`
				Style      [][]float32
				Speed      []float32
			}
		}
		if err := json.Unmarshal(data, &fixtures); err != nil {
			t.Fatal(err)
		}
		for _, fixture := range fixtures {
			chunks, err := prepare(fixture.Phonemes, fixture.Voice, 1, voices, vocab)
			if err != nil {
				t.Fatal(err)
			}
			if len(chunks) != len(fixture.Chunks) {
				t.Fatalf("%s %q: %d chunks, want %d", name, fixture.Phonemes, len(chunks), len(fixture.Chunks))
			}
			for i, chunk := range chunks {
				want := fixture.Chunks[i]
				if chunk.Phonemes != want.Phonemes || !reflect.DeepEqual(chunk.Tokens, want.Tokens[0]) || !reflect.DeepEqual(chunk.Style, want.Style[0]) || chunk.StyleIndex != want.StyleIndex || chunk.Speed != want.Speed[0] {
					t.Fatalf("%s %q chunk %d: prepared input differs from fixture", name, fixture.Phonemes, i)
				}
			}
		}
	}
}
