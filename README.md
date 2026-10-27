# e2k-go

Translate English into katakana notation in pure Go. This is a partial port of
[Patchethium/e2k](https://github.com/Patchethium/e2k): only the C2K model
(spelling to katakana) with greedy decoding, CPU only, standard library only.

## Model weights

The weights are not included in this repository because they may be derived
from a CC-BY-SA dictionary. Check the upstream terms before redistributing them.

Download them from the upstream release:

- URL: https://github.com/Patchethium/e2k/releases/download/0.4.0/model-c2k.safetensors
- SHA-256: `8dc015ba1574b386cacf54d498e9d418e3c03fb7da58a65b88bc81e454705ca1`

## Usage

```go
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/tenkoh/e2k-go"
)

func main() {
	f, err := os.Open("model-c2k.safetensors")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	m, err := e2k.LoadC2K(f)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(m.Convert("geogaddi")) // ジオガディ
}
```

A loaded `*e2k.C2K` is safe for concurrent use. Conversion cost grows with the
input length, so bound the input length yourself when it comes from untrusted
sources. `ConvertContext` stops early when its context is cancelled.

## Development

- `make fetch-model` downloads the weights into `testdata/` and verifies their SHA-256.
- `make check` runs the formatters, linters and tests.
- `uv run testdata/gen.py tiny` regenerates the tiny model and its trace fixture
  from the upstream Python implementation. `uv run testdata/gen.py edge`
  regenerates a variant of it that overflows the softmax, produces NaN logits
  and ties logits exactly. `uv run testdata/gen.py real` regenerates the
  expected outputs of the real model.
