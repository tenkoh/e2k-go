package e2k

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

// TestTinyEdge pins upstream's numerics that ordinary weights never reach:
// softmax overflowing to NaN because it skips max subtraction, argmax picking
// the first NaN, and argmax keeping the first of tied logits. gen.py checks
// that the fixture exercises each of them.
func TestTinyEdge(t *testing.T) {
	m := loadFile(t, "testdata/tiny-edge-c2k.safetensors")
	for _, c := range readFixture(t, "testdata/tiny-edge-c2k.json") {
		t.Run(fmt.Sprintf("%q", c.Input), func(t *testing.T) {
			if got := m.Convert(c.Input); got != c.Output {
				t.Errorf("Convert(%q) = %q, want %q", c.Input, got, c.Output)
			}
		})
	}
}

// stepCtx counts Err calls and reports Canceled once left reaches zero.
// A negative left never cancels.
type stepCtx struct {
	context.Context
	left, calls int
}

func (c *stepCtx) Err() error {
	c.calls++
	if c.left == 0 {
		return context.Canceled
	}
	c.left--
	return nil
}

func TestContextCheckedEachStep(t *testing.T) {
	m := loadFile(t, "testdata/tiny-c2k.safetensors")
	const input = "ab"
	srcLen := len(input) + 2 // with <sos> and <eos>

	run := func(left int) (*stepCtx, int, error) {
		ctx := &stepCtx{Context: context.Background(), left: left}
		steps := 0
		_, err := m.convert(ctx, input, func(layer string, _ ...[]float32) {
			if layer == "logits" {
				steps++
			}
		})
		return ctx, steps, err
	}

	ctx, steps, err := run(-1)
	if err != nil {
		t.Fatal(err)
	}
	// Once before starting, then once per encoder step in each direction and
	// once per decoder step.
	if want := 1 + 2*srcLen + steps; ctx.calls != want {
		t.Fatalf("ctx.Err called %d times, want %d", ctx.calls, want)
	}

	for left := range ctx.calls {
		_, got, err := run(left)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled at check %d: err = %v, want context.Canceled", left, err)
		}
		if want := max(0, left-1-2*srcLen); got != want {
			t.Errorf("cancelled at check %d: ran %d decoder steps, want %d", left, got, want)
		}
	}
}
