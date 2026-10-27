// Package e2k converts English spellings into katakana with the C2K model of
// Patchethium/e2k (https://github.com/Patchethium/e2k), using greedy decoding.
package e2k

import (
	"context"
	"fmt"
	"io"
	"maps"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// maxOutputLen is the number of symbols upstream decodes at most.
const maxOutputLen = 16

// C2K is a loaded character-to-katakana model.
//
// A C2K is immutable after LoadC2K and safe for concurrent use by multiple
// goroutines. Conversion cost grows with the input length, so callers that
// accept untrusted input should bound its length.
type C2K struct {
	sos, eos int
	inIndex  map[rune]int
	outTable []string

	eEmb, kEmb     matrix
	encFwd, encRev gru
	encFC          linear
	attn           mha
	pre, post      gru
	fc             linear
}

// LoadC2K reads a C2K model in safetensors format, such as model-c2k.safetensors
// from the upstream releases. The caller closes r.
func LoadC2K(r io.Reader) (*C2K, error) {
	tensors, meta, err := readSafetensors(r)
	if err != nil {
		return nil, fmt.Errorf("e2k: %w", err)
	}
	m, err := newC2K(tensors, meta)
	if err != nil {
		return nil, fmt.Errorf("e2k: %w", err)
	}
	return m, nil
}

// Convert returns the katakana for s. See ConvertContext.
func (m *C2K) Convert(s string) string {
	out, _ := m.ConvertContext(context.Background(), s)
	return out
}

// ConvertContext returns the katakana for s. Input is lowercased and
// characters outside the model's alphabet are dropped. The output has at most
// 16 symbols. It returns ctx.Err() if ctx is done before conversion finishes.
func (m *C2K) ConvertContext(ctx context.Context, s string) (string, error) {
	return m.convert(ctx, s, nil)
}

// traceFunc receives intermediate values for tests, under the layer names
// that testdata/gen.py records.
type traceFunc func(layer string, rows ...[]float32)

func (m *C2K) convert(ctx context.Context, s string, trace traceFunc) (string, error) {
	if trace == nil {
		trace = func(string, ...[]float32) {}
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}

	src := []int{m.sos}
	for _, r := range strings.ToLower(s) {
		if i, ok := m.inIndex[r]; ok {
			src = append(src, i)
		}
	}
	src = append(src, m.eos)

	h := make([]float32, m.encFwd.hidden())
	fwd := make([][]float32, len(src))
	for t, c := range src {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		h = m.encFwd.step(m.eEmb.row(c), h)
		fwd[t] = h
	}
	h = make([]float32, m.encRev.hidden())
	rev := make([][]float32, len(src))
	for t := len(src) - 1; t >= 0; t-- {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		h = m.encRev.step(m.eEmb.row(src[t]), h)
		rev[t] = h
	}
	trace("enc_fwd", fwd...)
	trace("enc_rev", rev...)

	enc := make([][]float32, len(src))
	keys := make([][]float32, len(src))
	values := make([][]float32, len(src))
	for t := range src {
		e := m.encFC.apply(append(append([]float32{}, fwd[t]...), rev[t]...))
		for i, v := range e {
			e[i] = tanh(v)
		}
		enc[t] = e
		keys[t] = m.attn.k.apply(e)
		values[t] = m.attn.v.apply(e)
	}
	trace("enc_out", enc...)

	var out strings.Builder
	prev := m.sos
	h1 := make([]float32, m.pre.hidden())
	h2 := make([]float32, m.post.hidden())
	for range maxOutputLen {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		h1 = m.pre.step(m.kEmb.row(prev), h1)
		a := m.attn.attend(h1, keys, values)
		h2 = m.post.step(append(append([]float32{}, h1...), a...), h2)
		logits := m.fc.apply(h2)
		trace("h1", h1)
		trace("attn", a)
		trace("h2", h2)
		trace("logits", logits)
		prev = argmax(logits)
		if prev == m.eos {
			break
		}
		// Like upstream, <pad> and <sos> are emitted as their literal table entries.
		out.WriteString(m.outTable[prev])
	}
	return out.String(), nil
}

// newC2K derives every dimension from the tensors and checks that they fit
// together, so inference on a loaded model cannot index out of range.
func newC2K(t map[string]tensor, meta map[string]string) (*C2K, error) {
	inTable, err := metaString(meta, "in_table")
	if err != nil {
		return nil, err
	}
	outTable, err := metaString(meta, "out_table")
	if err != nil {
		return nil, err
	}
	m := &C2K{inIndex: map[rune]int{}, outTable: strings.Split(outTable, "\x00")}
	in := strings.Split(inTable, "\x00")
	for i, s := range in {
		// Only single-character entries can match an input character.
		if r, size := utf8.DecodeRuneInString(s); size > 0 && size == len(s) {
			m.inIndex[r] = i
		}
	}
	if m.sos, err = metaIndex(meta, "sos_idx", len(in), len(m.outTable)); err != nil {
		return nil, err
	}
	if m.eos, err = metaIndex(meta, "eos_idx", len(in), len(m.outTable)); err != nil {
		return nil, err
	}

	l := loader{t: t}
	m.eEmb = l.matrix("e_emb.weight", len(in), -1)
	m.kEmb = l.matrix("k_emb.weight", len(m.outTable), -1)
	m.encFwd = l.gru("encoder.%s_l0", m.eEmb.cols)
	m.encRev = l.gru("encoder.%s_l0_reverse", m.eEmb.cols)
	h := m.encFwd.hidden()
	l.check(m.encRev.hidden() == h, "encoder directions have hidden sizes %d and %d", h, m.encRev.hidden())
	m.encFC = l.linear("encoder_fc.0.weight", "encoder_fc.0.bias", -1, 2*h)
	d := m.encFC.w.rows
	l.check(d > 0 && d%numHeads == 0, "attention dim %d is not a positive multiple of %d", d, numHeads)
	inProj := l.linear("attn.in_proj_weight", "attn.in_proj_bias", 3*d, d)
	m.attn.o = l.linear("attn.out_proj.weight", "attn.out_proj.bias", d, d)
	m.pre = l.gru("pre_decoder.%s_l0", m.kEmb.cols)
	l.check(m.pre.hidden() == d, "pre_decoder hidden size %d != attention dim %d", m.pre.hidden(), d)
	m.post = l.gru("post_decoder.%s_l0", 2*d)
	m.fc = l.linear("fc.weight", "fc.bias", len(m.outTable), m.post.hidden())
	if l.err == nil && len(l.t) > 0 {
		l.err = fmt.Errorf("unexpected tensor %q", slices.Min(slices.Collect(maps.Keys(l.t))))
	}
	if l.err != nil {
		return nil, l.err
	}
	m.attn.q = inProj.rows(0, d)
	m.attn.k = inProj.rows(d, 2*d)
	m.attn.v = inProj.rows(2*d, 3*d)
	return m, nil
}

func metaString(meta map[string]string, key string) (string, error) {
	s, ok := meta[key]
	if !ok {
		return "", fmt.Errorf("missing metadata %q", key)
	}
	return s, nil
}

// metaIndex parses an index that must be below every limit.
func metaIndex(meta map[string]string, key string, limits ...int) (int, error) {
	s, err := metaString(meta, key)
	if err != nil {
		return 0, err
	}
	i, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("metadata %q: %w", key, err)
	}
	for _, n := range limits {
		if i < 0 || i >= n {
			return 0, fmt.Errorf("metadata %q = %d is out of range", key, i)
		}
	}
	return i, nil
}

// loader looks up tensors by name and shape, removing each one it returns so
// that leftovers can be reported as unexpected. After the first error it keeps
// that error and returns zero values, so newC2K checks it once at the end.
type loader struct {
	t   map[string]tensor
	err error
}

func (l *loader) check(ok bool, format string, args ...any) {
	if !ok && l.err == nil {
		l.err = fmt.Errorf(format, args...)
	}
}

// get returns the data and shape of the named tensor, which must have the
// given shape, where -1 matches any size.
func (l *loader) get(name string, shape ...int) ([]float32, []int) {
	zero := make([]int, len(shape))
	if l.err != nil {
		return nil, zero
	}
	t, ok := l.t[name]
	if !ok {
		l.check(false, "missing tensor %q", name)
		return nil, zero
	}
	delete(l.t, name)
	match := len(t.shape) == len(shape)
	for i := 0; match && i < len(shape); i++ {
		match = shape[i] == -1 || shape[i] == t.shape[i]
	}
	if !match {
		l.check(false, "tensor %q has shape %v, want %v", name, t.shape, shape)
		return nil, zero
	}
	return t.data, t.shape
}

func (l *loader) matrix(name string, rows, cols int) matrix {
	data, shape := l.get(name, rows, cols)
	return matrix{rows: shape[0], cols: shape[1], data: data}
}

func (l *loader) linear(weight, bias string, out, in int) linear {
	w := l.matrix(weight, out, in)
	b, _ := l.get(bias, w.rows)
	return linear{w: w, b: b}
}

func (l *loader) gru(pattern string, in int) gru {
	name := func(s string) string { return fmt.Sprintf(pattern, s) }
	hh := l.matrix(name("weight_hh"), -1, -1)
	h := hh.cols
	l.check(hh.rows == 3*h, "tensor %q has %d rows, want %d", name("weight_hh"), hh.rows, 3*h)
	bhh, _ := l.get(name("bias_hh"), 3*h)
	return gru{
		ih: l.linear(name("weight_ih"), name("bias_ih"), 3*h, in),
		hh: linear{w: hh, b: bhh},
	}
}
