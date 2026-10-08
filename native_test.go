package kokoro

import (
	"bytes"
	"os"
	"os/exec"
	"regexp"
	"testing"
	"time"
)

// nativeSubprocess runs the calling test in a fresh process and reports whether
// the caller is that process. ortenv retains the ONNX Runtime environment until
// process exit, so tests that need an uninitialized environment, or that must
// not leave one behind for others, each get their own process.
func nativeSubprocess(t *testing.T) bool {
	t.Helper()
	const marker = "KOKORO_TEST_SUBPROCESS"
	if os.Getenv(marker) == t.Name() {
		return true
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"-test.run=^" + regexp.QuoteMeta(t.Name()) + "$", "-test.count=1", "-test.v"}
	if deadline, ok := t.Deadline(); ok {
		args = append(args, "-test.timeout="+time.Until(deadline).String())
	}
	cmd := exec.CommandContext(t.Context(), executable, args...)
	cmd.Env = append(os.Environ(), marker+"="+t.Name())
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("native subprocess failed: %v\n%s", err, output)
	}
	// A child that matched no test or skipped also exits 0.
	if !bytes.Contains(output, []byte("--- PASS: "+t.Name()+" (")) {
		t.Fatalf("native subprocess did not pass %s:\n%s", t.Name(), output)
	}
	t.Logf("native subprocess:\n%s", output)
	return false
}
