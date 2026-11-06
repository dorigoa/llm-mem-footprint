#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convertitore di formati in virgola mobile usati negli LLM.

DIREZIONE IN AVANTI  (numero -> bit):
    python3 float_formats.py 0,15625
    python3 float_formats.py -0.7

STAMPA: FP32, BF16, FP16, FP8-E4M3, FP8-E5M2, INT8 (con scala 1/127),
        MXFP4 (E2M1 + scala E8M0 per blocco), NVFP4 (E2M1 + scala UE4M3).

DIREZIONE INVERSA  (bit -> numero):
    python3 float_formats.py -f fp16  "0 01101 0100011111"
    python3 float_formats.py -f bf16  0b0011111000010000
    python3 float_formats.py -f mxfp4 1101 01111100      # 4 bit dati + 8 bit scala
    python3 float_formats.py -f nvfp4 1101_01111100
    python3 float_formats.py -f int8  11001010            # complemento a 2, scala 1/127
    python3 float_formats.py -f fp32  0_01111100_01000000000000000000000

I bit possono essere scritti con spazi, '_' o '0b'; il numero decimale
accetta sia la virgola sia il punto.
"""

import argparse
import math
import sys
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Descrittori dei formati in virgola mobile (sign / exp / mantissa)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FloatFormat:
    name: str
    exp_bits: int
    man_bits: int
    bias: int
    has_inf: bool = True    # pattern exp-tutti-1, mant=0  -> infinito
    has_nan: bool = True    # pattern exp-tutti-1, mant!=0 -> NaN


FORMATS = {
    "fp32": FloatFormat("fp32", 8, 23, 127),
    "bf16": FloatFormat("bf16", 8, 7, 127),
    "fp16": FloatFormat("fp16", 5, 10, 15),
    "e4m3": FloatFormat("e4m3", 4, 3, 7, has_inf=False, has_nan=True),   # FP8 "e4m3fn"
    "e5m2": FloatFormat("e5m2", 5, 2, 15),                               # FP8 OCP
    "e2m1": FloatFormat("e2m1", 2, 1, 1, has_inf=False, has_nan=False),  # FP4 OCP/NV
}

INT8_DEFAULT_SCALE = 1.0 / 127.0
E8M0_NAN = 255           # campo scala E8M0 (MX) per 2^(255-127): NaN
E8M0_BIAS = 127


# --------------------------------------------------------------------------
# Arrotaldamento round-to-nearest-even
# --------------------------------------------------------------------------
def round_rne(y: float) -> int:
    """Arrotonda y >= 0 al piu' vicino intero, .5 -> pari."""
    f = math.floor(y)
    r = y - f
    if r > 0.5 or (r == 0.5 and f % 2 == 1):
        return f + 1
    return f


# --------------------------------------------------------------------------
# Codifica / decodifica generica float
# --------------------------------------------------------------------------
def encode_float(x: float, fmt: FloatFormat) -> int:
    if math.isnan(x) or math.isinf(x):
        raise ValueError("NaN/Inf non possono essere codificati")
    mb = fmt.man_bits
    maxfield = (1 << fmt.exp_bits) - 1
    sign = 1 if (x < 0 or (x == 0 and math.copysign(1.0, x) < 0)) else 0
    a = abs(float(x))

    if a == 0:
        field, mant = 0, 0
    else:
        m, e2 = math.frexp(a)       # a = m * 2**e2  con 0.5 <= m < 1
        E = e2 - 1                  # esponente IEEE: a = 1,f * 2**E
        if E < 1 - fmt.bias:        # subnormale
            shift = 1 - fmt.bias
            mant = round_rne(a * 2.0 ** (mb - shift))
            field = 0
            if mant >= 1 << mb:     # diventato normale dopo l'arrotondamento
                field, mant = 1, 0
        else:
            mant = round_rne(a * 2.0 ** (mb - E)) - (1 << mb)   # toglie l'1 implicito
            field = E + fmt.bias
            if mant >= 1 << mb:         # carry dalla mantissa -> esponente +1
                field += 1
                mant = 0

    # overflow
    if fmt.has_inf:
        if field >= maxfield:                    # -> +-Inf
            field, mant = maxfield, 0
    else:
        # saturazione al finito massimo (comportamento tipico di e4m3 ed e2m1)
        max_mant_top = (1 << mb) - 2 if fmt.has_nan else (1 << mb) - 1
        if field > maxfield or (field == maxfield and mant > max_mant_top):
            field, mant = maxfield, max_mant_top

    return (sign << (fmt.exp_bits + mb)) | (field << mb) | mant


def decode_float(bits: int, fmt: FloatFormat) -> float:
    mb = fmt.man_bits
    sign = (bits >> (fmt.exp_bits + mb)) & 1
    field = (bits >> mb) & ((1 << fmt.exp_bits) - 1)
    mant = bits & ((1 << mb) - 1)
    maxfield = (1 << fmt.exp_bits) - 1

    if field == maxfield and fmt.has_inf and mant == 0:
        v = math.inf
    elif field == maxfield and fmt.has_nan and mant != 0:
        v = math.nan
    elif field == 0:
        v = mant * 2.0 ** (1 - fmt.bias - mb)                     # subnormale
    else:
        v = (1.0 + mant / 2.0 ** mb) * 2.0 ** (field - fmt.bias)  # normale
    return -v if sign else v


# --------------------------------------------------------------------------
# FormatI a blocchi per pesi LLM: MXFP4 e NVFP4
# --------------------------------------------------------------------------
def enc_mxfp4(x: float):
    """MXFP4 (OCP MX): dato E2M1 + scala di blocco E8M0 = 2^(f-127), blocco 32.

    Scegliamo la scala potenza di 2 massima tale che |x|/s <= 6,
    cioe' s = 2^ceil(log2(|x|/6))  (cosi' |x|/s sta in (3, 6]).
    """
    if x == 0:
        return 0, E8M0_BIAS                      # dati 0, scala 1.0
    r = abs(x) / 6.0
    m, e2 = math.frexp(r)
    e = e2 - 1 if m == 0.5 else e2               # minimo intero con 2**e >= r
    while 2.0 ** e < r:
        e += 1
    e = max(-127, min(127, e))                   # limiti del campo E8M0
    s = 2.0 ** e
    data = encode_float(x / s, FORMATS["e2m1"])
    return data, e + E8M0_BIAS


def dec_mxfp4(data_bits: int, scale_field: int) -> float:
    if scale_field == E8M0_NAN:
        return math.nan
    return decode_float(data_bits, FORMATS["e2m1"]) * 2.0 ** (scale_field - E8M0_BIAS)


def enc_nvfp4(x: float):
    """NVFP4 (NVIDIA): dato E2M1 + scala di blocco UE4M3 (senza segno), blocco 16.

    (Piu' una scala globale FP32 per tutto il tensore, qui assunta = 1.0.)
    Scegliamo il valore E4M3 minimo s tale che |x|/s <= 6 (=> |x|/s non travalica).
    """
    e4m3 = FORMATS["e4m3"]
    if x == 0:
        return 0, 0x38                           # UE4M3 per 1.0
    target = abs(x) / 6.0
    sbits = encode_float(target, e4m3)           # RNE
    while decode_float(sbits, e4m3) < target and sbits < 0x7E:
        sbits += 1                               # arrotonda per eccesso: niente overflow
    if sbits == 0:
        sbits = 1                                # scala subnormale minima, non zero
    s = decode_float(sbits, e4m3)
    data = encode_float(x / s, FORMATS["e2m1"])  # eventualmente saturo a 6*s
    return data, sbits


def dec_nvfp4(data_bits: int, scale_bits: int) -> float:
    if (scale_bits & 0x7F) == 0x7F:              # 0x7F = NaN per E4M3 senza segno
        return math.nan
    s = decode_float(scale_bits & 0x7F, FORMATS["e4m3"])
    return decode_float(data_bits, FORMATS["e2m1"]) * s


# --------------------------------------------------------------------------
# INT8 (quantizzazione simmetrica di pesi: w ~= q * scala)
# --------------------------------------------------------------------------
def enc_int8(x: float, scale: float) -> int:
    q = round_rne(abs(x) / scale)
    if x < 0:
        q = -q
    q = max(-128, min(127, q))
    return q & 0xFF


def dec_int8(bits: int, scale: float) -> float:
    q = bits - 256 if bits >= 128 else bits
    return q * scale


# --------------------------------------------------------------------------
# Utility di visualizzazione / parsing
# --------------------------------------------------------------------------
def bits_grouped(bits: int, width: int) -> str:
    s = format(bits, f"0{width}b")
    parts = []
    i = width
    while i > 0:
        j = max(0, i - 4)
        parts.append(s[j:i])
        i = j
    return "_".join(reversed(parts))


def annotate(bits: int, fmt: FloatFormat) -> str:
    w = 1 + fmt.exp_bits + fmt.man_bits
    s = format(bits, f"0{w}b")
    return f"S={s[0]}  E={''.join(s[1:1+fmt.exp_bits])}  M={''.join(s[1+fmt.exp_bits:])}"


def it_num(v: float) -> str:
    """Numero in notazione decimale con la virgola."""
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if v == 0:
        return "0"
    s = f"{v:.17g}"
    # notazione posizionale se l'esponente e' ragionevole, altrimenti scientifica
    if -4 <= math.floor(math.log10(abs(v))) <= 16:
        from decimal import Decimal
        s = format(Decimal(s), "f")
    else:
        s = f"{float(s):e}"
    return s.replace(".", ",").rstrip("0").rstrip(",") if "," in s and "e" not in s else s.replace(".", ",")


def parse_bits(text: str) -> str:
    t = text.lower().replace("0b", "").replace(" ", "").replace("_", "").replace(",", "").replace(".", "")
    if any(c not in "01" for c in t):
        raise SystemExit(f"errore: '{text}' non e' una stringa di bit valida")
    return t


def parse_number(text: str) -> float:
    return float(text.replace(",", "."))


# --------------------------------------------------------------------------
# Direzioni
# --------------------------------------------------------------------------
def encode_all(x: float, int8_scale: float):
    print(f"Numero in input: {it_num(x)}   (exatto in doppio: {x!r})")
    print("=" * 78)

    for key in ("fp32", "bf16", "fp16", "e4m3", "e5m2", "e2m1"):
        fmt = FORMATS[key]
        bits = encode_float(x, fmt)
        back = decode_float(bits, fmt)
        err = "" if back == x else f"   err = {back - x:+.3e}"
        label = {"e4m3": "FP8 E4M3 (e4m3fn)", "e5m2": "FP8 E5M2 (OCP)",
                 "e2m1": "FP4 E2M1 (senza scala)"}.get(key, key.upper())
        print(f"{label:26s} {bits_grouped(bits, 1 + fmt.exp_bits + fmt.man_bits):>12s}  [{annotate(bits, fmt)}]")
        print(f"{'':26s} -> {it_num(back)}{err}")

    # INT8
    bits = enc_int8(x, int8_scale)
    back = dec_int8(bits, int8_scale)
    print(f"{'INT8 (scala ' + format(int8_scale, '.6g').replace('.', ',') + ')':26s} {bits_grouped(bits, 8)}  [complemento a 2, q={bits - 256 if bits >= 128 else bits}]")
    print(f"{'':26s} -> {it_num(back)}   err = {back - x:+.3e}")

    # MXFP4
    data, sf = enc_mxfp4(x)
    back = dec_mxfp4(data, sf)
    print(f"{'MXFP4 (blocco 32)':26s} dati={format(data, '04b')}  scala E8M0={format(sf, '08b')} (=2^{sf - E8M0_BIAS})")
    print(f"{'':26s} -> {it_num(back)}   err = {back - x:+.3e}")
    d1 = encode_float(x, FORMATS["e2m1"])
    print(f"{'':26s} (con scala forzata 1.0: dati={format(d1, '04b')} -> {it_num(decode_float(d1, FORMATS['e2m1']))})")

    # NVFP4
    data, sb = enc_nvfp4(x)
    back = dec_nvfp4(data, sb)
    print(f"{'NVFP4 (blocco 16)':26s} dati={format(data, '04b')}  scala UE4M3={format(sb & 0x7F, '07b')}0 (= {it_num(decode_float(sb & 0x7F, FORMATS['e4m3']))})")
    print(f"{'':26s} -> {it_num(back)}   err = {back - x:+.3e}")
    print(f"{'':26s} nota: NVFP4 ha anche una scala globale FP32 per il tensore (qui 1,0)")


ALIASES = {
    "fp32": "fp32", "float32": "fp32", "ieee754": "fp32",
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "half": "fp16", "float16": "fp16",
    "e4m3": "e4m3", "fp8": "e4m3", "fp8_e4m3": "e4m3", "e4m3fn": "e4m3", "fp8fn": "e4m3",
    "e5m2": "e5m2", "fp8_e5m2": "e5m2",
    "e2m1": "e2m1", "fp4": "e2m1", "fp4_e2m1": "e2m1",
    "mxfp4": "mxfp4", "mx_fp4": "mxfp4",
    "nvfp4": "nvfp4", "nv_fp4": "nvfp4",
    "int8": "int8", "q8_0": "int8", "w8": "int8", "w4": "mxfp4",
}

WIDTHS = {"fp32": 32, "bf16": 16, "fp16": 16, "e4m3": 8, "e5m2": 8,
          "e2m1": 4, "int8": 8, "mxfp4": (4, 12), "nvfp4": (4, 12)}


def decode_one(fmt_name: str, bits_text: str, int8_scale: float):
    fmt = ALIASES.get(fmt_name.lower().replace("-", "_"))
    if fmt is None:
        raise SystemExit(f"errore: formato '{fmt_name}' sconosciuto "
                         f"(validi: {', '.join(sorted(set(ALIASES)))})")
    b = parse_bits(bits_text)

    if fmt in WIDTHS:
        w = WIDTHS[fmt]
        ok = (len(b) == w) if isinstance(w, int) else (len(b) in w)
        if not ok:
            exp = f"{w} bit" if isinstance(w, int) else " o ".join(f"{x} bit" for x in w)
            raise SystemExit(f"errore: {fmt} richiede {exp}, ricevuti {len(b)} bit")
    bits = int(b, 2)

    if fmt in FORMATS:
        f = FORMATS[fmt]
        v = decode_float(bits, f)
        print(f"{fmt.upper():8s} bit = {bits_grouped(bits, len(b))}  [{annotate(bits, f)}]")
    elif fmt == "int8":
        v = dec_int8(bits, int8_scale)
        print(f"{'INT8':8s} bit = {bits_grouped(bits, 8)}  [complemento a 2, q={bits - 256 if bits >= 128 else bits}, scala {it_num(int8_scale)}]")
    elif fmt == "mxfp4":
        if len(b) == 12:
            d, s = bits >> 8, bits & 0xFF
        else:
            d, s = bits, E8M0_BIAS
        v = dec_mxfp4(d, s)
        print(f"{'MXFP4':8s} dati={format(d, '04b')} (E2M1)  scala={format(s, '08b')} (E8M0 = 2^{s - E8M0_BIAS})")
    elif fmt == "nvfp4":
        if len(b) == 12:
            d, s = bits >> 8, bits & 0x7F
        else:
            d, s = bits, 0x38
        v = dec_nvfp4(d, s)
        print(f"{'NVFP4':8s} dati={format(d, '04b')} (E2M1)  scala={format(s, '07b')} (UE4M3 = {it_num(decode_float(s, FORMATS['e4m3'])) if s != 0x7F else 'NaN'})")

    print(f"          ->  {it_num(v)}")
    return v


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Converti numeri decimali <-> formati FP32/BF16/FP16/FP8/INT8/MXFP4/NVFP4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("DIREZIONE IN AVANTI")[0])
    ap.add_argument("valore", nargs="+", help="numero decimale (|x|<1) da convertire, oppure stringa di bit con -f")
    ap.add_argument("-f", "--format", help="formato della stringa di bit (decode): "
                                           "fp32, bf16, fp16, e4m3, e5m2, e2m1, int8, mxfp4, nvfp4")
    ap.add_argument("--int8-scale", type=float, default=INT8_DEFAULT_SCALE,
                    help="scala usata per INT8 (default 1/127 = %(default).6f)")
    args = ap.parse_args([a.replace(",", ".") for a in sys.argv[1:]])
    args.valore = " ".join(args.valore) if len(args.valore) == 1 else "".join(args.valore)

    if args.format:
        decode_one(args.format, args.valore, args.int8_scale)
    else:
        try:
            x = parse_number(args.valore)
        except ValueError:
            raise SystemExit(f"errore: '{args.valore}' non e' un numero")
        if abs(x) >= 1:
            print(f"avviso: |{it_num(x)}| >= 1 (il formato e' pensato per |x| < 1)", file=sys.stderr)
        encode_all(x, args.int8_scale)


if __name__ == "__main__":
    main()
