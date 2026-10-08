package kokoro

import (
	"reflect"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"
)

func TestStreamChunking(t *testing.T) {
	for _, test := range []struct {
		name, text string
		options    StreamOptions
		want       []string
	}{
		{"punctuation", "həlˈO, wˈɜɹld! həlˈO əɡˈɛn.", StreamOptions{FirstChunkPhonemes: 10}, []string{"həlˈO,", "wˈɜɹld! həlˈO əɡˈɛn."}},
		{"last phrase in budget", "a, b! c d e", StreamOptions{FirstChunkPhonemes: 7}, []string{"a, b!", "c d e"}},
		{"space at limit", "abc de fgh", StreamOptions{FirstChunkPhonemes: 3}, []string{"abc", "de fgh"}},
		{"word boundary", "abc defgh ij", StreamOptions{FirstChunkPhonemes: 6}, []string{"abc", "defgh ij"}},
		{"unicode hard split", "əɜɹˈəɜɹˈəɜɹ", StreamOptions{MaxChunkPhonemes: 4, FirstChunkPhonemes: 2}, []string{"əɜ", "ɹˈəɜ", "ɹˈəɜ", "ɹ"}},
		{"later chunk budget", "abc def ghi jkl mno", StreamOptions{MaxChunkPhonemes: 7, FirstChunkPhonemes: 3}, []string{"abc", "def ghi", "jkl mno"}},
		{"short input", "  həlˈO!  ", StreamOptions{FirstChunkPhonemes: 10}, []string{"həlˈO!"}},
		{"punctuation sequence", "a... b!!! c", StreamOptions{MaxChunkPhonemes: 5}, []string{"a...", "b!!!", "c"}},
		{"unicode whitespace", "ə\u2003ɜ\tɹ", StreamOptions{MaxChunkPhonemes: 2}, []string{"ə", "ɜ", "ɹ"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := splitStreamPhonemes(test.text, test.options)
			if err != nil || !reflect.DeepEqual(got, test.want) {
				t.Fatalf("got %q, %v; want %q", got, err, test.want)
			}
		})
	}
	for _, text := range []string{"  həlˈO ,wˈɜɹld!  ", strings.Repeat("ə", 1020), strings.Repeat("a, b! ", 200)} {
		want, err := SplitPhonemes(text)
		if err != nil {
			t.Fatal(err)
		}
		got, err := splitStreamPhonemes(text, StreamOptions{})
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("zero options changed original packing: %q, %v; want %q", got, err, want)
		}
	}
	for _, options := range []StreamOptions{
		{MaxChunkPhonemes: -1}, {MaxChunkPhonemes: MaxPhonemes + 1},
		{FirstChunkPhonemes: -1}, {FirstChunkPhonemes: MaxPhonemes + 1},
		{MaxChunkPhonemes: 10, FirstChunkPhonemes: 11},
	} {
		if _, err := splitStreamPhonemes("həlˈO", options); err == nil {
			t.Fatalf("accepted invalid options: %+v", options)
		}
	}
	for _, text := range []string{"", " \t\n", "\xff"} {
		if _, err := splitStreamPhonemes(text, StreamOptions{FirstChunkPhonemes: 10}); err == nil {
			t.Fatalf("accepted invalid input %q", text)
		}
	}
}

func FuzzStreamChunking(f *testing.F) {
	for _, text := range []string{"həlˈO. wˈɜɹld!", strings.Repeat("ə", 1500), "a  b , ; !", "   ", "a\u2003b"} {
		f.Add(text, uint16(80), uint16(160))
		f.Add(text, uint16(1), uint16(1))
	}
	f.Fuzz(func(t *testing.T, text string, first, maximum uint16) {
		options := StreamOptions{MaxChunkPhonemes: 1 + int(maximum)%MaxPhonemes}
		options.FirstChunkPhonemes = 1 + int(first)%options.MaxChunkPhonemes
		chunks, err := splitStreamPhonemes(text, options)
		if err != nil {
			if utf8.ValidString(text) && strings.TrimSpace(text) != "" {
				t.Fatal(err)
			}
			return
		}
		for i, chunk := range chunks {
			limit := options.MaxChunkPhonemes
			if i == 0 {
				limit = options.FirstChunkPhonemes
			}
			if chunk == "" || !utf8.ValidString(chunk) || utf8.RuneCountInString(chunk) > limit {
				t.Fatalf("invalid chunk %q with limit %d", chunk, limit)
			}
		}
		strip := func(text string) string {
			return strings.Map(func(r rune) rune {
				if unicode.IsSpace(r) {
					return -1
				}
				return r
			}, text)
		}
		if strip(strings.Join(chunks, "")) != strip(text) {
			t.Fatal("chunking lost or changed phonemes")
		}
	})
}

func TestSplitPhonemes(t *testing.T) {
	for _, n := range []int{1, 508, 509, 510, 511, 1018, 1531} {
		text := strings.Repeat("ə", n)
		chunks, err := SplitPhonemes(text)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Join(chunks, "") != text {
			t.Fatalf("omission at %d", n)
		}
		for _, c := range chunks {
			if len([]rune(c)) > MaxPhonemes || c == "" {
				t.Fatal("bad chunk")
			}
		}
	}
	text := strings.Repeat("həlˈO wˈɜɹld. ", 100)
	chunks, err := SplitPhonemes(text)
	if err != nil {
		t.Fatal(err)
	}
	strip := func(s string) string {
		return strings.Map(func(r rune) rune {
			if unicode.IsSpace(r) {
				return -1
			}
			return r
		}, s)
	}
	if strip(strings.Join(chunks, "")) != strip(text) {
		t.Fatal("lost phonemes")
	}
	for _, c := range chunks {
		if utf8.RuneCountInString(c) > MaxPhonemes {
			t.Fatal("oversized chunk")
		}
	}
	got, _ := SplitPhonemes("  həlˈO ,wˈɜɹld!  ")
	if len(got) != 1 || got[0] != "həlˈO, wˈɜɹld!" {
		t.Fatalf("punctuation packing: %q", got)
	}
	for _, input := range []string{"", " \t\n", "\xff"} {
		if _, err := SplitPhonemes(input); err == nil {
			t.Fatal("invalid input accepted")
		}
	}
}

func FuzzSplitPhonemes(f *testing.F) {
	for _, s := range []string{"həlˈO.", strings.Repeat("ə", 1500), "a  b , ; !", "   "} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		chunks, err := SplitPhonemes(text)
		if err != nil {
			return
		}
		for _, c := range chunks {
			if c == "" || !utf8.ValidString(c) || utf8.RuneCountInString(c) > MaxPhonemes {
				t.Fatalf("bad chunk %q", c)
			}
		}
		strip := func(s string) string {
			return strings.Map(func(r rune) rune {
				if unicode.IsSpace(r) {
					return -1
				}
				return r
			}, s)
		}
		if strip(strings.Join(chunks, "")) != strip(text) {
			t.Fatal("omitted phonemes")
		}
	})
}
