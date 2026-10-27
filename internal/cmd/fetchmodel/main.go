// Command fetchmodel downloads the upstream C2K weights used by the tests
// and installs them only if their SHA-256 matches the pinned digest.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	modelURL    = "https://github.com/Patchethium/e2k/releases/download/0.4.0/model-c2k.safetensors"
	modelSHA256 = "8dc015ba1574b386cacf54d498e9d418e3c03fb7da58a65b88bc81e454705ca1"
)

func main() {
	out := flag.String("o", "testdata/model-c2k.safetensors", "destination path")
	flag.Parse()
	log.SetFlags(0)
	if err := fetch(*out); err != nil {
		log.Fatalf("fetchmodel: %v", err)
	}
}

func fetch(dst string) error {
	switch sum, err := fileSHA256(dst); {
	case err == nil && sum == modelSHA256:
		return nil
	case err != nil && !errors.Is(err, fs.ErrNotExist):
		return err
	}

	// The file is a few MB; the default client would wait forever on a stalled
	// connection.
	client := &http.Client{Timeout: 2 * time.Minute}
	resp, err := client.Get(modelURL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", modelURL, resp.Status)
	}

	// Download next to dst so the final rename is atomic.
	tmp, err := os.CreateTemp(filepath.Dir(dst), ".fetchmodel-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	defer tmp.Close()

	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(tmp, h), resp.Body); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != modelSHA256 {
		return fmt.Errorf("SHA-256 mismatch: got %s, want %s", got, modelSHA256)
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), dst)
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
