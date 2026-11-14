"""Fail fast on a sick GPU before spending hours on it.

A Spot A100 once returned NaN probabilities, tripped a device-side assert and
then dropped off the machine mid-session, so every result of that session was
suspect. This runs a few seconds of known computations and exits non-zero if
the GPU is missing, reports ECC/Xid trouble, or computes wrong numbers."""
import subprocess
import sys

import torch


def main() -> int:
    problems = []
    if not torch.cuda.is_available():
        print("gpu_check: FAIL no CUDA GPU visible")
        return 1
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,ecc.errors.uncorrected.volatile.total,temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    print("gpu_check:", q.stdout.strip())
    try:
        ecc = q.stdout.strip().split(",")[1].strip()
        if ecc not in ("0", "[N/A]", "N/A"):
            problems.append(f"uncorrected ECC errors: {ecc}")
    except IndexError:
        problems.append("nvidia-smi gave no answer")
    d = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True)
    xid = [l for l in d.stdout.splitlines() if "NVRM: Xid" in l]
    if xid:
        problems.append(f"{len(xid)} Xid error(s) in the kernel log, last: {xid[-1][-160:]}")
    torch.manual_seed(0)
    for dtype, tol in ((torch.float32, 5e-3), (torch.bfloat16, 3e-2)):  # float32 may run as TF32
        a, b = torch.randn(4096, 4096), torch.randn(4096, 4096)
        ref = (a.double() @ b.double())
        for _ in range(5):
            got = (a.to("cuda", dtype) @ b.to("cuda", dtype)).double().cpu()
            if not torch.isfinite(got).all():
                problems.append(f"{dtype} matmul produced non-finite values")
                break
            err = ((got - ref).norm() / ref.norm()).item()
            if err > tol:
                problems.append(f"{dtype} matmul relative error {err:.2e} > {tol}")
                break
    x = torch.randn(32004, device="cuda", dtype=torch.bfloat16)
    if not torch.isfinite(torch.softmax(x.float(), -1)).all():
        problems.append("softmax produced non-finite values")
    torch.cuda.synchronize()
    if problems:
        for p in problems:
            print("gpu_check: FAIL", p)
        return 1
    print("gpu_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
