package text_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	kokoro "github.com/androiddrew/go-kokoro"
	"github.com/androiddrew/go-kokoro/text"
)

var (
	shared     *text.Pipeline
	sharedErr  error
	sharedOnce sync.Once
)

// pipeline loads one pipeline for every test: the ONNX Runtime environment
// outlives it anyway, and Kokoro takes seconds to load.
func pipeline(t *testing.T) *text.Pipeline {
	t.Helper()
	library, dir := os.Getenv("KOKORO_TEST_ORT"), os.Getenv("KOKORO_TEST_ASSETS")
	if library == "" || dir == "" {
		t.Skip("set KOKORO_TEST_ORT and KOKORO_TEST_ASSETS for text-to-speech tests")
	}
	sharedOnce.Do(func() {
		shared, sharedErr = text.New(text.Config{
			ORTLibrary: library,
			Synthesis: kokoro.Config{
				ModelPath:  filepath.Join(dir, "kokoro-v1.0.onnx"),
				VoicesPath: filepath.Join(dir, "voices-v1.0.bin"),
			},
		})
	})
	if sharedErr != nil {
		t.Fatal(sharedErr)
	}
	return shared
}

func TestMain(m *testing.M) {
	code := m.Run()
	if shared != nil {
		if err := shared.Close(); err != nil {
			code = 1
		}
	}
	os.Exit(code)
}

func collect(t *testing.T, p *text.Pipeline, ctx context.Context, r text.Request) []text.Chunk {
	t.Helper()
	var chunks []text.Chunk
	for c, err := range p.Stream(ctx, r) {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, c)
	}
	return chunks
}

func TestStream(t *testing.T) {
	p := pipeline(t)
	ctx := context.Background()
	const gap = 100 * time.Millisecond
	req := text.Request{
		Text:  "**Hello** there. The meeting is at 3:05 p.m. on May 5, 2026! How are you today?",
		Voice: "af_heart", Speed: 1, Trim: true, Gap: gap,
		Prep: text.PrepOptions{Markdown: true, Normalize: true},
	}
	chunks := collect(t, p, ctx, req)
	var texts []string
	for _, c := range chunks {
		texts = append(texts, c.Text)
	}
	if len(chunks) != 3 {
		t.Fatalf("got %d chunks, want one per sentence: %q", len(chunks), texts)
	}
	if !strings.Contains(chunks[1].Text, "three oh five PM") || len(chunks[0].Diagnostics) != 0 {
		t.Errorf("sentence 2 not normalized: %q", chunks[1].Text)
	}
	gapSamples := int(gap.Seconds() * text.SampleRate)
	for i, c := range chunks {
		if len(c.PCM) <= gapSamples || c.Tokens == 0 {
			t.Fatalf("chunk %d: %d samples, %d tokens", i, len(c.PCM), c.Tokens)
		}
		leading := 0
		for _, s := range c.PCM {
			if s != 0 {
				break
			}
			leading++
		}
		if (i == 0 && leading >= gapSamples) || (i > 0 && leading < gapSamples) {
			t.Errorf("chunk %d starts with %d zero samples; the gap is %d after the first", i, leading, gapSamples)
		}
	}

	// Stopping early leaves the pipeline usable, including mid-sentence.
	for range p.Stream(ctx, req) {
		break
	}
	ctx2, cancel := context.WithCancel(ctx)
	cancel()
	for _, err := range p.Stream(ctx2, req) {
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled stream: %v", err)
		}
	}
	for _, err := range p.Stream(ctx, text.Request{Text: "...", Voice: "af_heart", Speed: 1}) {
		if err == nil {
			t.Fatal("punctuation alone produced audio")
		}
	}
	for _, err := range p.Stream(ctx, text.Request{Text: "Hello.", Voice: "jf_alpha", Speed: 1}) {
		if err == nil || !strings.Contains(err.Error(), "not a US or UK English voice") {
			t.Fatalf("Japanese voice: %v", err)
		}
	}
}

// TestStreamChunking splits long sentences into short model runs. Every run
// respects the limits, the first is the shortest, and each sentence's runs
// together speak the phonemes of its unchunked synthesis.
func TestStreamChunking(t *testing.T) {
	p := pipeline(t)
	ctx := context.Background()
	req := text.Request{
		Text:  "Once upon a time, in a small house at the edge of a quiet forest, there lived a little robot who loved nothing more than listening to the rain.",
		Voice: "bf_emma", Speed: 1, Trim: true,
	}
	strip := func(s string) string { return strings.Join(strings.Fields(s), "") }
	want := map[string]string{} // sentence -> its phonemes, unchunked
	var sentences []string
	for _, c := range collect(t, p, ctx, req) {
		if _, ok := want[c.Text]; !ok {
			sentences = append(sentences, c.Text)
		}
		want[c.Text] += strip(c.Phonemes)
	}
	req.Chunking = kokoro.StreamOptions{FirstChunkPhonemes: 20, MaxChunkPhonemes: 60}
	parts := collect(t, p, ctx, req)
	if len(parts) < len(sentences)+2 {
		t.Fatalf("chunked: %d chunks for %d sentences, want shorter runs", len(parts), len(sentences))
	}
	got := map[string]string{}
	for i, c := range parts {
		n := len([]rune(c.Phonemes))
		if i == 0 && n > 20 || n > 60 || len(c.PCM) == 0 {
			t.Errorf("chunk %d: %d phonemes, %d samples: %q", i, n, len(c.PCM), c.Phonemes)
		}
		if i > 0 && c.Text == parts[i-1].Text && c.Diagnostics != nil {
			t.Errorf("chunk %d repeats the sentence's diagnostics", i)
		}
		got[c.Text] += strip(c.Phonemes)
	}
	for _, sentence := range sentences {
		if got[sentence] != want[sentence] {
			t.Errorf("%q: chunks speak %q, unchunked %q", sentence, got[sentence], want[sentence])
		}
	}

	// Breaking out mid-sentence releases the engine.
	for range p.Stream(ctx, req) {
		break
	}
	if _, err := p.Synthesis.Prepare("həlˈO", "af_heart", 1); err != nil {
		t.Fatalf("engine unusable after breaking out of a sentence: %v", err)
	}
}

func TestStreamPronunciation(t *testing.T) {
	p := pipeline(t)
	// Past the neural fallback's 62-code-point limit, so its output is truncated.
	long := "Say " + strings.Repeat("zorblax", 10) + " now."
	req := text.Request{Text: long, Voice: "bf_emma", Speed: 1, Trim: true}
	var diagnostics int
	for c, err := range p.Stream(context.Background(), req) {
		if err != nil {
			t.Fatalf("lenient: %v", err)
		}
		diagnostics += len(c.Diagnostics)
	}
	if diagnostics == 0 {
		t.Fatal("lenient stream reported no diagnostics for a truncated word")
	}
	req.Strict = true
	for _, err := range p.Stream(context.Background(), req) {
		if !errors.Is(err, text.ErrPronunciation) {
			t.Fatalf("strict: got %v, want ErrPronunciation", err)
		}
	}
}
