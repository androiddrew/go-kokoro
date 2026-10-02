package kokoro_test

import (
	"context"
	"fmt"
	"log"
	"os"

	kokoro "github.com/androiddrew/go-kokoro"
	ort "github.com/yalue/onnxruntime_go"
)

// Synthesize phonemes and write a WAV file. The model and voices are downloads
// (see ASSETS.md); phonemes can come from github.com/androiddrew/go-g2p.
func Example() {
	engine, err := kokoro.New(kokoro.Config{
		ORTLibrary: "libonnxruntime.so", // OS loader name or a path to the library
		ModelPath:  "assets/kokoro/kokoro-v1.0.onnx",
		VoicesPath: "assets/kokoro/voices-v1.0.bin",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	result, err := engine.Synthesize(context.Background(), kokoro.Request{
		Phonemes: "həlˈO, wˈɜɹld!", Voice: "af_heart", Speed: 1, Trim: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Create("hello.wav")
	if err != nil {
		log.Fatal(err)
	}
	if err = kokoro.WriteWAV(f, result.Samples); err != nil {
		log.Fatal(err)
	}
	if err = f.Close(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%.2fs of audio\n", float64(len(result.Samples))/kokoro.SampleRate)
}

// An application that already initializes ONNX Runtime itself leaves
// ORTLibrary empty. The engine then uses that environment and never destroys it.
func Example_callerOwnedEnvironment() {
	ort.SetSharedLibraryPath("libonnxruntime.so")
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment() // runs after engine.Close

	engine, err := kokoro.New(kokoro.Config{
		ModelPath:  "assets/kokoro/kokoro-v1.0.onnx",
		VoicesPath: "assets/kokoro/voices-v1.0.bin",
		Provider:   kokoro.CUDA, // requires an ONNX Runtime build with CUDA
	})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()
	fmt.Println(engine.Voices())
}

// ListVoices reads the voice archive without loading ONNX Runtime.
func ExampleListVoices() {
	voices, err := kokoro.ListVoices("assets/kokoro/voices-v1.0.bin")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(voices), "voices, e.g.", voices[0])
}
