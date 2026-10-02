package kokoro

import (
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"
)

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
