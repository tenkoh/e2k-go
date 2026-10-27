package e2k

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
)

type fixtureCase struct {
	Input  string                 `json:"input"`
	Output string                 `json:"output"`
	Trace  map[string][][]float64 `json:"trace"`
}

func readFixture(t *testing.T, path string) []fixtureCase {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cases []fixtureCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	return cases
}

func loadFile(t *testing.T, path string) *C2K {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	m, err := LoadC2K(f)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// loadReal loads the upstream weights, which are not in the repository.
func loadReal(t *testing.T) *C2K {
	t.Helper()
	const path = "testdata/model-c2k.safetensors"
	if _, err := os.Stat(path); err != nil {
		if os.Getenv("CI") == "true" {
			t.Fatalf("%s is required in CI: %v", path, err)
		}
		t.Skipf("%s not found, run `make fetch-model`", path)
	}
	return loadFile(t, path)
}

func TestTinyTrace(t *testing.T) {
	m := loadFile(t, "testdata/tiny-c2k.safetensors")
	for _, c := range readFixture(t, "testdata/tiny-c2k.json") {
		t.Run(fmt.Sprintf("%q", c.Input), func(t *testing.T) {
			got := map[string][][]float32{}
			out, err := m.convert(context.Background(), c.Input, func(layer string, rows ...[]float32) {
				var flat []float32
				for _, r := range rows {
					flat = append(flat, r...)
				}
				got[layer] = append(got[layer], flat)
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := firstDivergence(got, c.Trace); err != nil {
				t.Error(err)
			}
			if out != c.Output {
				t.Errorf("output = %q, want %q", out, c.Output)
			}
		})
	}
}

// firstDivergence compares traces in computation order and describes the
// first value that differs.
func firstDivergence(got map[string][][]float32, want map[string][][]float64) error {
	type call struct {
		layer string
		step  int
	}
	var order []call
	for _, l := range []string{"enc_fwd", "enc_rev", "enc_out"} {
		order = append(order, call{l, 0})
	}
	steps := max(len(want["logits"]), len(got["logits"]))
	for s := range steps {
		for _, l := range []string{"h1", "attn", "h2", "logits"} {
			order = append(order, call{l, s})
		}
	}
	for _, c := range order {
		g, w := got[c.layer], want[c.layer]
		if c.step >= len(g) || c.step >= len(w) {
			return fmt.Errorf("%s: %d calls, want %d", c.layer, len(g), len(w))
		}
		gv, wv := g[c.step], w[c.step]
		if len(gv) != len(wv) {
			return fmt.Errorf("%s step %d: %d values, want %d", c.layer, c.step, len(gv), len(wv))
		}
		for i := range wv {
			a, b := float64(gv[i]), wv[i]
			if !(math.Abs(a-b) <= 1e-4+1e-4*math.Abs(b)) {
				return fmt.Errorf("%s step %d index %d: got %v, want %v", c.layer, c.step, i, a, b)
			}
		}
	}
	return nil
}

func TestReal(t *testing.T) {
	m := loadReal(t)
	cases := readFixture(t, "testdata/real-c2k.json")
	for _, c := range cases {
		t.Run(fmt.Sprintf("%q", c.Input), func(t *testing.T) {
			if got := m.Convert(c.Input); got != c.Output {
				t.Errorf("Convert(%q) = %q, want %q", c.Input, got, c.Output)
			}
		})
	}
	if got := m.Convert("geogaddi"); got != "ジオガディ" {
		t.Errorf("Convert(geogaddi) = %q, want ジオガディ", got)
	}
}

func TestConcurrent(t *testing.T) {
	var inputs []string
	for _, c := range readFixture(t, "testdata/tiny-c2k.json") {
		inputs = append(inputs, c.Input)
	}
	models := map[string]*C2K{"tiny": loadFile(t, "testdata/tiny-c2k.safetensors")}
	if _, err := os.Stat("testdata/model-c2k.safetensors"); err == nil {
		models["real"] = loadFile(t, "testdata/model-c2k.safetensors")
		for _, c := range readFixture(t, "testdata/real-c2k.json") {
			inputs = append(inputs, c.Input)
		}
	}
	for name, m := range models {
		t.Run(name, func(t *testing.T) {
			want := make([]string, len(inputs))
			for i, s := range inputs {
				want[i] = m.Convert(s)
			}
			var wg sync.WaitGroup
			// 32 goroutines, each converting every 8th input, so every input
			// runs on 4 goroutines at once while keeping -race time low.
			for g := range 32 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for j := g % 8; j < len(inputs); j += 8 {
						if got := m.Convert(inputs[j]); got != want[j] {
							t.Errorf("Convert(%q) = %q, want %q", inputs[j], got, want[j])
						}
					}
				}()
			}
			wg.Wait()
		})
	}
}

func TestCancelledContext(t *testing.T) {
	m := loadFile(t, "testdata/tiny-c2k.safetensors")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.ConvertContext(ctx, "abc"); !errors.Is(err, context.Canceled) {
		t.Fatalf("ConvertContext with cancelled ctx: err = %v, want context.Canceled", err)
	}
}

// tinyParts splits the tiny model into its decoded header and its body.
func tinyParts(t *testing.T) (map[string]any, []byte) {
	t.Helper()
	raw, err := os.ReadFile("testdata/tiny-c2k.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	n := binary.LittleEndian.Uint64(raw)
	dec := json.NewDecoder(bytes.NewReader(raw[8 : 8+n]))
	dec.UseNumber()
	var header map[string]any
	if err := dec.Decode(&header); err != nil {
		t.Fatal(err)
	}
	return header, raw[8+n:]
}

func encode(t *testing.T, header map[string]any, body []byte) []byte {
	t.Helper()
	h, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	return withHeader(h, body)
}

func withHeader(h, body []byte) []byte {
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	return append(append(out, h...), body...)
}

func TestLoadMalformed(t *testing.T) {
	good, _ := os.ReadFile("testdata/tiny-c2k.safetensors")
	if _, err := LoadC2K(bytes.NewReader(good)); err != nil {
		t.Fatalf("unmodified tiny model: %v", err)
	}

	// edit returns the tiny model with its header changed by f.
	edit := func(f func(h map[string]any)) func(t *testing.T) []byte {
		return func(t *testing.T) []byte {
			h, body := tinyParts(t)
			f(h)
			return encode(t, h, body)
		}
	}
	tensorField := func(name, field string, v any) func(t *testing.T) []byte {
		return edit(func(h map[string]any) { h[name].(map[string]any)[field] = v })
	}
	metaField := func(key string, v any) func(t *testing.T) []byte {
		return edit(func(h map[string]any) {
			meta := h["__metadata__"].(map[string]any)
			if v == nil {
				delete(meta, key)
			} else {
				meta[key] = v
			}
		})
	}
	fixed := func(b []byte) func(*testing.T) []byte { return func(*testing.T) []byte { return b } }

	cases := map[string]func(t *testing.T) []byte{
		"empty":                  fixed(nil),
		"short length":           fixed(good[:4]),
		"huge header length":     fixed(binary.LittleEndian.AppendUint64(nil, math.MaxUint64)),
		"header past end":        fixed(withHeader([]byte(`{}`), nil)[:9]),
		"truncated body":         fixed(good[:len(good)-1]),
		"bad json":               fixed(withHeader([]byte(`{"a":`), nil)),
		"header not object":      fixed(withHeader([]byte(`[]`), nil)),
		"empty header":           fixed(withHeader([]byte(`{}`), nil)),
		"metadata not object":    edit(func(h map[string]any) { h["__metadata__"] = "x" }),
		"metadata not string":    metaField("sos_idx", 1),
		"metadata null":          metaField("eos_idx", nil),
		"missing in_table":       metaField("in_table", nil),
		"sos not integer":        metaField("sos_idx", "one"),
		"sos negative":           metaField("sos_idx", "-1"),
		"eos out of range":       metaField("eos_idx", "8"),
		"in_table too long":      metaField("in_table", "<pad>\x00<sos>\x00<eos>\x00a\x00b\x00c\x00 \x00'\x00d"),
		"out_table too short":    metaField("out_table", "<pad>\x00<sos>\x00<eos>"),
		"wrong dtype":            tensorField("fc.bias", "dtype", "F32"),
		"dtype not string":       tensorField("fc.bias", "dtype", 16),
		"negative shape":         tensorField("fc.bias", "shape", []any{-8}),
		"shape vs offsets":       tensorField("fc.bias", "shape", []any{9}),
		"shape overflow":         tensorField("fc.bias", "shape", []any{uint64(1) << 62, 4}),
		"offsets past body":      tensorField("fc.bias", "data_offsets", []any{len(good), len(good) + 16}),
		"offsets reversed":       tensorField("fc.bias", "data_offsets", []any{16, 0}),
		"offsets one entry":      tensorField("fc.bias", "data_offsets", []any{0}),
		"tensor not object":      edit(func(h map[string]any) { h["fc.bias"] = 1 }),
		"missing tensor":         edit(func(h map[string]any) { delete(h, "fc.bias") }),
		"transposed encoder_fc":  tensorField("encoder_fc.0.weight", "shape", []any{10, 8}),
		"bias as matrix":         tensorField("fc.bias", "shape", []any{8, 1}),
		"gru hh not 3h rows":     tensorField("encoder.weight_hh_l0", "shape", []any{25, 3}),
		"pre_decoder hh vs bias": tensorField("pre_decoder.weight_hh_l0", "shape", []any{12, 16}),
		"overlapping offsets":    tensorField("fc.bias", "data_offsets", []any{0, 16}),
		"extra tensor": edit(func(h map[string]any) {
			_, body := tinyParts(t)
			h["encoder.weight_ih_l1"] = map[string]any{"dtype": "F16", "shape": []any{0}, "data_offsets": []any{len(body), len(body)}}
		}),
	}
	for name, build := range cases {
		t.Run(name, func(t *testing.T) {
			m, err := LoadC2K(bytes.NewReader(build(t)))
			if err == nil {
				t.Fatalf("LoadC2K succeeded, model %v", m != nil)
			}
			if !strings.HasPrefix(err.Error(), "e2k: ") {
				t.Errorf("error %q lacks the e2k: prefix", err)
			}
		})
	}

	t.Run("aliased tensors stay small", func(t *testing.T) {
		// Many tensors over the same 1 MiB used to decode into gigabytes before
		// any error was reported.
		const n, size = 1000, 1 << 20
		h := map[string]any{}
		for i := range n {
			h[fmt.Sprint("t", i)] = map[string]any{"dtype": "F16", "shape": []any{size / 2}, "data_offsets": []any{0, size}}
		}
		file := encode(t, h, make([]byte, size))
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		if _, err := LoadC2K(bytes.NewReader(file)); err == nil {
			t.Fatal("LoadC2K succeeded")
		}
		runtime.ReadMemStats(&after)
		if got := after.TotalAlloc - before.TotalAlloc; got > 64<<20 {
			t.Errorf("LoadC2K allocated %d MiB for a %d MiB file", got>>20, len(file)>>20)
		}
	})

	t.Run("every truncation", func(t *testing.T) {
		for n := range len(good) {
			if _, err := LoadC2K(bytes.NewReader(good[:n])); err == nil {
				t.Fatalf("LoadC2K of the first %d bytes succeeded", n)
			}
		}
	})
}
