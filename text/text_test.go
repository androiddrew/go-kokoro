package text

import (
	"testing"

	g2p "github.com/androiddrew/go-g2p"
)

func TestPrepare(t *testing.T) {
	both := PrepOptions{Markdown: true, Normalize: true}
	for _, c := range []struct {
		in   string
		o    PrepOptions
		want string
	}{
		{"Meet at 10:30 AM on May 5, 2026.", PrepOptions{}, "Meet at 10:30 AM on May 5, 2026."},
		{"Meet at 10:30 AM on May 5, 2026.", PrepOptions{Normalize: true}, "Meet at ten thirty AM on May fifth, twenty twenty-six."},
		{"**Note:** it's [Kubernetes](/kˌubəɹnˈɛtiz/) v1.2.3, not `kubectl`.", both, "Note: it's [Kubernetes](/kˌubəɹnˈɛtiz/) v one point two point three, not kubectl."},
		{"## Steps\n\n1. Read pages 31-35\n2. It weighs 2.5 kg", both, "Steps. Read pages thirty-one to thirty-five. It weighs two point five kilograms."},
		{"Email hello@example.com", both, "Email H E L L O at E X A M P L E dot C O M"},
	} {
		if got := Prepare(c.in, c.o); got != c.want {
			t.Errorf("Prepare(%q, %+v)\n got: %q\nwant: %q", c.in, c.o, got, c.want)
		}
	}
}

func TestVoiceDialect(t *testing.T) {
	for voice, want := range map[string]g2p.Dialect{"af_heart": g2p.US, "am_adam": g2p.US, "bf_emma": g2p.GB, "bm_george": g2p.GB, "jf_alpha": "", "": ""} {
		got, ok := VoiceDialect(voice)
		if got != want || ok != (want != "") {
			t.Errorf("VoiceDialect(%q) = %q, %v", voice, got, ok)
		}
	}
}

func TestSpeakable(t *testing.T) {
	for phonemes, want := range map[string]bool{"həlˈO": true, "...": false, " , ! ": false, "": false} {
		if speakable(phonemes) != want {
			t.Errorf("speakable(%q) != %v", phonemes, want)
		}
	}
}

func TestNewFrontendRejectsUnknownFallback(t *testing.T) {
	if _, err := NewFrontend(FrontendConfig{Fallback: "festival"}); err == nil {
		t.Fatal("accepted an unknown fallback")
	}
}
