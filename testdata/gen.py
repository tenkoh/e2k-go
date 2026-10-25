# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""Generates the test fixtures from the upstream reference implementation.

    uv run testdata/gen.py tiny   # tiny-c2k.safetensors and tiny-c2k.json
    uv run testdata/gen.py real   # real-c2k.json, needs testdata/model-c2k.safetensors
"""

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = "2d4aabf8000e69f9c70378e41f0cd17abe8be9f1"
UPSTREAM_URL = f"https://raw.githubusercontent.com/Patchethium/e2k/{UPSTREAM_COMMIT}/src/e2k/inference.py"
UPSTREAM_SHA256 = "26b06bf15444487c5a989924744814f136f5165d75920009af133fca0edc3815"
MODEL_SHA256 = "8dc015ba1574b386cacf54d498e9d418e3c03fb7da58a65b88bc81e454705ca1"

HERE = Path(__file__).resolve().parent

TINY_CASES = [
    "",
    "a",
    "abc",
    "ABC",
    "cab ba",
    "a'b",
    "xyz!",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "日本a",
    "Ab C'",
    "cc",
    "ba",
]

REAL_CASES = [
    "geogaddi",
    "GeoGaddi",
    "GEOGADDI",
    "computer",
    "coffee",
    "bed",
    "cookie",
    "check",
    "violin",
    "vegas",
    "tokyo",
    "music",
    "shop",
    "jazz",
    "hello",
    "apple",
    "game",
    "water",
    "new york",
    "ice cream",
    "rock'n'roll",
    "don't",
    "e-mail",
    "café",
    "ＡＢＣ",
    "İstanbul",
    "Kelvin",
    "new\tyork",
    "123",
    "!!!",
    "",
    "supercalifragilisticexpialidocious",
    "antidisestablishmentarianism",
]


def load_upstream():
    with urllib.request.urlopen(UPSTREAM_URL) as resp:
        src = resp.read()
    if hashlib.sha256(src).hexdigest() != UPSTREAM_SHA256:
        sys.exit("upstream inference.py does not match the pinned SHA-256")
    path = Path(tempfile.mkdtemp()) / "e2k_inference.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("e2k_inference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_safetensors(path: Path, tensors: dict, metadata: dict):
    header = {"__metadata__": metadata}
    body = b""
    for name in sorted(tensors):
        data = tensors[name].astype("<f2").tobytes()
        header[name] = {
            "dtype": "F16",
            "shape": list(tensors[name].shape),
            "data_offsets": [len(body), len(body) + len(data)],
        }
        body += data
    raw = json.dumps(header, ensure_ascii=False).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + body)


def tiny_weights():
    rng = np.random.default_rng(1)
    in_table = ["<pad>", "<sos>", "<eos>", "a", "b", "c", " ", "'"]
    out_table = ["<pad>", "<sos>", "<eos>", "ア", "イ", "ウ", "ー", "ッ"]
    # distinct sizes so a swapped dimension fails to load or diverges
    e, h, d, k, q = 6, 5, 8, 7, 9
    shapes = {
        "e_emb.weight": [len(in_table), e],
        "k_emb.weight": [len(out_table), k],
        "encoder_fc.0.weight": [d, 2 * h],
        "encoder_fc.0.bias": [d],
        "attn.in_proj_weight": [3 * d, d],
        "attn.in_proj_bias": [3 * d],
        "attn.out_proj.weight": [d, d],
        "attn.out_proj.bias": [d],
        "fc.weight": [len(out_table), q],
        "fc.bias": [len(out_table)],
    }
    for prefix, inp, hid in [
        ("encoder.%s_l0", e, h),
        ("encoder.%s_l0_reverse", e, h),
        ("pre_decoder.%s_l0", k, d),
        ("post_decoder.%s_l0", 2 * d, q),
    ]:
        shapes[prefix % "weight_ih"] = [3 * hid, inp]
        shapes[prefix % "weight_hh"] = [3 * hid, hid]
        shapes[prefix % "bias_ih"] = [3 * hid]
        shapes[prefix % "bias_hh"] = [3 * hid]
    tensors = {n: rng.normal(0, 1.2, s).astype(np.float16) for n, s in shapes.items()}
    metadata = {
        "in_table": "\0".join(in_table),
        "out_table": "\0".join(out_table),
        "sos_idx": "1",
        "eos_idx": "2",
    }
    return tensors, metadata


def trace(mod, c2k, word):
    """Runs `c2k(word)` and records the intermediate tensors of every step."""
    s2s = c2k.s2s
    rec = {}

    def add(name, v):
        rec.setdefault(name, []).append(np.asarray(v, np.float32).ravel().tolist())

    def wrap(obj, attr, record):
        orig = getattr(obj, attr)

        def f(*args):
            out = orig(*args)
            record(args, out)
            return out

        setattr(obj, attr, f)
        return orig

    saved = [
        (s2s.encoder, "forward", wrap(s2s.encoder, "forward", lambda a, o: add("enc_fwd", o[0]))),
        (s2s.encoder_reverse, "forward", wrap(s2s.encoder_reverse, "forward", lambda a, o: add("enc_rev", o[0]))),
        (s2s.attn, "memory", wrap(s2s.attn, "memory", lambda a, o: add("enc_out", a[0]))),
        (s2s.pre_decoder, "forward", wrap(s2s.pre_decoder, "forward", lambda a, o: add("h1", o[1]))),
        (s2s.attn, "forward", wrap(s2s.attn, "forward", lambda a, o: add("attn", o))),
        (s2s.post_decoder, "forward", wrap(s2s.post_decoder, "forward", lambda a, o: add("h2", o[1]))),
    ]
    greedy = mod.greedy
    mod.greedy = lambda x: (add("logits", x), greedy(x))[1]
    try:
        out = c2k(word)
    finally:
        mod.greedy = greedy
        for obj, attr, _ in saved:
            delattr(obj, attr)
    return out, rec


def gen_tiny(mod):
    model = HERE / "tiny-c2k.safetensors"
    write_safetensors(model, *tiny_weights())
    c2k = mod.C2K(path=str(model))
    cases = []
    for word in TINY_CASES:
        out, rec = trace(mod, c2k, word)
        cases.append({"input": word, "output": out, "trace": rec})
    dump(HERE / "tiny-c2k.json", cases)


def gen_real(mod):
    model = HERE / "model-c2k.safetensors"
    if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA256:
        sys.exit(f"{model} does not match the pinned SHA-256, run `make fetch-model`")
    c2k = mod.C2K(path=str(model))
    dump(HERE / "real-c2k.json", [{"input": w, "output": c2k(w)} for w in REAL_CASES])


def dump(path: Path, cases):
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    cmds = {"tiny": gen_tiny, "real": gen_real}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](load_upstream())
