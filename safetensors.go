package e2k

import (
	"cmp"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
)

// maxHeaderLen bounds the JSON header. The C2K header is a few KiB.
const maxHeaderLen = 1 << 20

// tensor is a decoded F16 tensor, widened to float32.
type tensor struct {
	shape []int
	data  []float32
}

type tensorInfo struct {
	Dtype       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

// readSafetensors parses a safetensors stream into its tensors and string metadata.
func readSafetensors(r io.Reader) (map[string]tensor, map[string]string, error) {
	var lenBuf [8]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return nil, nil, fmt.Errorf("reading header length: %w", err)
	}
	n := binary.LittleEndian.Uint64(lenBuf[:])
	if n > maxHeaderLen {
		return nil, nil, fmt.Errorf("header length %d exceeds %d", n, maxHeaderLen)
	}
	raw := make([]byte, n)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, nil, fmt.Errorf("reading header: %w", err)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw, &header); err != nil {
		return nil, nil, fmt.Errorf("parsing header: %w", err)
	}

	meta := map[string]string{}
	if m, ok := header["__metadata__"]; ok {
		var fields map[string]any
		if err := json.Unmarshal(m, &fields); err != nil || fields == nil {
			return nil, nil, errors.New("__metadata__ is not an object")
		}
		for k, v := range fields {
			s, ok := v.(string)
			if !ok {
				return nil, nil, fmt.Errorf("metadata %q is not a string", k)
			}
			meta[k] = s
		}
		delete(header, "__metadata__")
	}

	infos := make(map[string]tensorInfo, len(header))
	ranges := make([][2]uint64, 0, len(header))
	for name, msg := range header {
		info, err := parseTensorInfo(msg)
		if err != nil {
			return nil, nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		infos[name] = info
		ranges = append(ranges, [2]uint64(info.DataOffsets))
	}
	// Like the safetensors spec, require the tensors to tile the body. Overlapping
	// ranges would let a small file decode into many times its size.
	slices.SortFunc(ranges, func(a, b [2]uint64) int { return cmp.Or(cmp.Compare(a[0], b[0]), cmp.Compare(a[1], b[1])) })
	var bodyLen uint64
	for _, rg := range ranges {
		if rg[0] != bodyLen {
			return nil, nil, fmt.Errorf("tensor data at %d, want %d: overlapping or gapped data_offsets", rg[0], bodyLen)
		}
		bodyLen = rg[1]
	}

	// LimitReader keeps a lying header from forcing one huge allocation.
	body, err := io.ReadAll(io.LimitReader(r, int64(bodyLen)))
	if err != nil {
		return nil, nil, fmt.Errorf("reading body: %w", err)
	}
	if uint64(len(body)) != bodyLen {
		return nil, nil, fmt.Errorf("body is %d bytes, header needs %d", len(body), bodyLen)
	}

	tensors := make(map[string]tensor, len(infos))
	for name, info := range infos {
		raw := body[info.DataOffsets[0]:info.DataOffsets[1]]
		t := tensor{shape: make([]int, len(info.Shape)), data: make([]float32, len(raw)/2)}
		for i, d := range info.Shape {
			t.shape[i] = int(d)
		}
		for i := range t.data {
			t.data[i] = f16ToF32(binary.LittleEndian.Uint16(raw[2*i:]))
		}
		tensors[name] = t
	}
	return tensors, meta, nil
}

func parseTensorInfo(msg json.RawMessage) (tensorInfo, error) {
	var info tensorInfo
	if err := json.Unmarshal(msg, &info); err != nil {
		return info, err
	}
	if info.Dtype != "F16" {
		return info, fmt.Errorf("unsupported dtype %q", info.Dtype)
	}
	if len(info.DataOffsets) != 2 {
		return info, errors.New("data_offsets must have two entries")
	}
	start, end := info.DataOffsets[0], info.DataOffsets[1]
	if start > end || end > math.MaxInt64 {
		return info, errors.New("bad data_offsets")
	}
	count := uint64(1)
	for _, d := range info.Shape {
		if d != 0 && count > (end-start)/d {
			return info, errors.New("shape does not match data_offsets")
		}
		count *= d
	}
	if count*2 != end-start {
		return info, errors.New("shape does not match data_offsets")
	}
	return info, nil
}

func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	e := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch e {
	case 0:
		v := float32(mant) / (1 << 24)
		if sign != 0 {
			v = -v
		}
		return v
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	default:
		return math.Float32frombits(sign | (e+127-15)<<23 | mant<<13)
	}
}
