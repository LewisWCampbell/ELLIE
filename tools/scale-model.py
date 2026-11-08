#!/usr/bin/env python3
"""Cost model behind the call-center scaling study.

Every rate is a named constant so a footnote in the study can point at it, and
so a price change is a one-line edit and a rerun rather than a document that
quietly goes stale. All figures are US list prices in us-central1 as published
in August 2026, before any negotiated discount, which makes the output a floor
rather than a forecast.

  python3 tools/scale-model.py
"""


# ---- rates (US, us-central1, list prices) -----------------------------------
TWILIO_INBOUND_MIN      = 0.0085     # programmable voice, inbound to a local number
TWILIO_SIP_INBOUND_MIN  = 0.0034     # elastic SIP trunking, inbound to local
TWILIO_NUMBER_MO        = 1.15

DG_STT_MULTI_MIN        = 0.0058     # nova-3 multilingual streaming
DG_STT_MONO_MIN         = 0.0048     # nova-3 monolingual streaming
DG_TTS_PER_1K_CHARS     = 0.030      # aura-2

OPUS5_IN_M, OPUS5_OUT_M = 5.00, 25.00
HAIKU_IN_M, HAIKU_OUT_M = 1.00, 5.00
CACHE_READ_MULT         = 0.10       # cached input ~= 0.1x input

GEMINI_LIVE_IN_M        = 3.00       # audio input tokens
GEMINI_LIVE_OUT_M       = 12.00      # audio output tokens
GEMINI_AUDIO_TOK_PER_S  = 25.0

CR_VCPU_S               = 0.000024   # request-based
CR_GIB_S                = 0.0000025
CR_VCPU_S_ALLOC         = 0.000018   # always-allocated (min-instance)
CR_GIB_S_ALLOC          = 0.000002
CR_REQ_M                = 0.40

H100_VM_HR              = 11.06      # a3-highgpu-1g, 1x H100 80GB, on demand
A100_40_HR              = 3.67       # a2-highgpu-1g
L4_HR                   = 0.70       # g2-standard-4
CR_L4_HR                = 0.672      # Cloud Run L4, no zonal redundancy ($0.0001867/s)
CUD_3YR                 = 0.45       # pay ~45% of on demand
SPOT_H100_HR            = 3.30

GCS_STD_GB_MO           = 0.020
SECRET_VERSION_MO       = 0.06
LOGGING_GIB             = 0.50
ARTIFACT_REG_GB_MO      = 0.10
EGRESS_GB               = 0.12

# ---- conversation shape ------------------------------------------------------
SPOKEN_CHARS_PER_MIN    = 900        # ~150 wpm
ASSISTANT_TALK_SHARE    = 0.50
TURNS_PER_MIN           = 2.0
CACHED_IN_TOK           = 6000       # system + knowledge, cached
FRESH_IN_TOK            = 400
OUT_TOK                 = 120

def llm_per_min(inp, out):
    per_turn = (CACHED_IN_TOK * inp * CACHE_READ_MULT / 1e6
                + FRESH_IN_TOK * inp / 1e6
                + OUT_TOK * out / 1e6)
    return per_turn * TURNS_PER_MIN

def tts_per_min():
    return SPOKEN_CHARS_PER_MIN * ASSISTANT_TALK_SHARE / 1000 * DG_TTS_PER_1K_CHARS

def gemini_live_per_min():
    inp = 60 * GEMINI_AUDIO_TOK_PER_S * GEMINI_LIVE_IN_M / 1e6
    out = 60 * ASSISTANT_TALK_SHARE * GEMINI_AUDIO_TOK_PER_S * GEMINI_LIVE_OUT_M / 1e6
    return inp + out

def cloudrun_bridge_per_min(vcpu=0.5, gib=0.5):
    return 60 * (vcpu * CR_VCPU_S + gib * CR_GIB_S)

def gpu_per_stream_min(vm_hr, streams, util, discount=1.0):
    return vm_hr * discount / streams / 60 / util

print("=== per call-minute, compute only ===")
rows = [
  ("Deepgram cascade + Opus 5", DG_STT_MULTI_MIN + tts_per_min() + llm_per_min(OPUS5_IN_M, OPUS5_OUT_M)),
  ("Deepgram cascade + Haiku 4.5", DG_STT_MULTI_MIN + tts_per_min() + llm_per_min(HAIKU_IN_M, HAIKU_OUT_M)),
  ("Gemini Live (managed duplex)", gemini_live_per_min()),
]
for n, v in rows: print(f"  {n:<34} ${v:.4f}")
print()
print(f"  {'  of which STT':<34} ${DG_STT_MULTI_MIN:.4f}")
print(f"  {'  of which TTS':<34} ${tts_per_min():.4f}")
print(f"  {'  of which LLM (Opus 5)':<34} ${llm_per_min(OPUS5_IN_M, OPUS5_OUT_M):.4f}")
print(f"  {'  of which LLM (Haiku 4.5)':<34} ${llm_per_min(HAIKU_IN_M, HAIKU_OUT_M):.4f}")
print(f"  bridge (Cloud Run, per min)        ${cloudrun_bridge_per_min():.5f}")
print()
print("=== PersonaPlex GPU cost per stream-minute ===")
for label, hr, disc in [("H100 on demand", H100_VM_HR, 1.0),
                        ("H100 3yr CUD", H100_VM_HR, CUD_3YR),
                        ("H100 spot", SPOT_H100_HR, 1.0),
                        ("A100 40GB on demand", A100_40_HR, 1.0),
                        ("L4 (Cloud Run)", CR_L4_HR, 1.0)]:
    line = f"  {label:<22}"
    for streams in (2, 4, 8, 16):
        line += f"  s={streams}: ${gpu_per_stream_min(hr, streams, 0.65, disc):.4f}"
    print(line)
print("  (at 65% utilisation)")

import math
print()
print("=== volume tiers ===")
BUSY_WALL_MIN = 8 * 22 * 60      # business-hours wall clock per month
PEAK_FACTOR   = 2.5
AVG_CALL_MIN  = 3.5
UTIL          = 0.65
STREAMS_PER_GPU = 8

def tier(monthly_min):
    avg = monthly_min / BUSY_WALL_MIN
    peak = math.ceil(avg * PEAK_FACTOR)
    calls = monthly_min / AVG_CALL_MIN
    return avg, max(peak, 1), calls

VOLUMES = [1_000, 10_000, 100_000, 1_000_000]
for v in VOLUMES:
    avg, peak, calls = tier(v)
    gpus_peak = math.ceil(peak / STREAMS_PER_GPU)
    print(f"  {v:>9,} min/mo  {calls:>8,.0f} calls  avg concurrent {avg:6.1f}  peak {peak:>4}  GPUs for peak {gpus_peak:>3}")

print()
print("=== fixed monthly (footnote lines) ===")
def fixed_costs(monthly_min, peak, gpu_fleet, gke=False):
    calls = monthly_min / AVG_CALL_MIN
    f = {}
    f["Twilio numbers (1)"] = TWILIO_NUMBER_MO
    # bridge floor: 1 always-on instance per 25 concurrent calls, min 1
    inst = max(1, math.ceil(peak / 25))
    f[f"Cloud Run min-instances ({inst} x 2 vCPU / 2 GiB)"] = (
        inst * 730 * 3600 * (2 * CR_VCPU_S_ALLOC + 2 * CR_GIB_S_ALLOC))
    f["Secret Manager (7 secrets + access)"] = 7 * SECRET_VERSION_MO + 0.03 * (calls / 10000)
    log_gib = monthly_min * 2 / 1024 / 1024 * 1024  # ~2 KB per call-minute -> MB -> GiB
    log_gib = monthly_min * 2 / 1_048_576
    f["Cloud Logging (beyond 50 GiB free)"] = max(0.0, log_gib - 50) * LOGGING_GIB
    f["GCS call archive (8 KB/call, 12 mo retained)"] = calls * 12 * 8 / 1_048_576 * GCS_STD_GB_MO
    f["Artifact Registry (2 GiB image)"] = max(0.0, 2 - 0.5) * ARTIFACT_REG_GB_MO
    f["Egress to Twilio (mu-law 8 kHz, 480 KB/min)"] = monthly_min * 480 / 1_048_576 * EGRESS_GB
    if gke:
        f["GKE cluster management"] = 0.10 * 730
        f["Internal load balancer"] = 18.00
        f["GPU node boot disks (100 GB SSD each)"] = gpu_fleet * 100 * 0.17
    return f

for v in VOLUMES:
    avg, peak, calls = tier(v)
    print(f"\n  --- {v:,} min/mo, peak {peak} concurrent ---")
    for k, val in fixed_costs(v, peak, math.ceil(peak/STREAMS_PER_GPU), gke=True).items():
        print(f"    {k:<48} ${val:>10,.2f}")
