package kokoro

import "github.com/androiddrew/go-kokoro/internal/assets"

// ListVoices reads the voice archive and returns sorted voice names without
// initializing ONNX Runtime. The returned slice is owned by the caller.
func ListVoices(path string) ([]string, error) {
	voices, err := assets.LoadVoices(path)
	if err != nil {
		return nil, err
	}
	return voices.Names(), nil
}
