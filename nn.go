package e2k

import "math"

// matrix is a row-major rows x cols matrix.
type matrix struct {
	rows, cols int
	data       []float32
}

func (m matrix) row(i int) []float32 {
	return m.data[i*m.cols : (i+1)*m.cols]
}

// linear computes w x + b, like torch.nn.Linear on a single vector.
type linear struct {
	w matrix
	b []float32
}

func (l linear) apply(x []float32) []float32 {
	y := make([]float32, l.w.rows)
	for i := range y {
		var s float32
		for j, v := range l.w.row(i) {
			s += v * x[j]
		}
		y[i] = s + l.b[i]
	}
	return y
}

// gru is a single torch.nn.GRU cell, gates stacked in r, z, n order.
type gru struct {
	ih, hh linear
}

func (g gru) hidden() int { return g.hh.w.cols }

func (g gru) step(x, h []float32) []float32 {
	gi := g.ih.apply(x)
	gh := g.hh.apply(h)
	n := len(h)
	out := make([]float32, n)
	for i := range out {
		r := sigmoid(gi[i] + gh[i])
		z := sigmoid(gi[n+i] + gh[n+i])
		c := tanh(gi[2*n+i] + r*gh[2*n+i])
		out[i] = (1-z)*c + z*h[i]
	}
	return out
}

const numHeads = 4

// mha is torch.nn.MultiheadAttention with numHeads heads, for one query.
type mha struct {
	q, k, v, o linear
}

// attend attends q over keys and values, which are already projected.
func (a mha) attend(q []float32, keys, values [][]float32) []float32 {
	q = a.q.apply(q)
	dh := len(q) / numHeads
	scale := float32(math.Sqrt(float64(dh)))
	o := make([]float32, len(q))
	w := make([]float32, len(keys))
	for h := range numHeads {
		lo, hi := h*dh, (h+1)*dh
		var sum float32
		for t, k := range keys {
			var s float32
			for j := lo; j < hi; j++ {
				s += q[j] * k[j]
			}
			// No max subtraction: upstream exponentiates the raw scores.
			w[t] = exp(s / scale)
			sum += w[t]
		}
		for t, v := range values {
			p := w[t] / sum
			for j := lo; j < hi; j++ {
				o[j] += p * v[j]
			}
		}
	}
	return a.o.apply(o)
}

// argmax returns the first maximum, or the first NaN like numpy.argmax.
func argmax(x []float32) int {
	best := 0
	for i, v := range x {
		if v != v {
			return i
		}
		if v > x[best] {
			best = i
		}
	}
	return best
}

func exp(x float32) float32     { return float32(math.Exp(float64(x))) }
func tanh(x float32) float32    { return float32(math.Tanh(float64(x))) }
func sigmoid(x float32) float32 { return 1 / (1 + exp(-x)) }

// rows returns output rows [lo, hi) of l, sharing its storage.
func (l linear) rows(lo, hi int) linear {
	return linear{
		w: matrix{rows: hi - lo, cols: l.w.cols, data: l.w.data[lo*l.w.cols : hi*l.w.cols]},
		b: l.b[lo:hi],
	}
}
