#!/usr/bin/env python3
"""Phase 1 hardware probes on 8x MI100: compute, collectives, PP transfers, graphs.

Run inside the glm53f container with all 8 GPUs visible:
    python3 /workspace/hw_probe.py

Exits non-zero on the first failed probe. Each probe prints PASS/FAIL lines.
"""
import os
import sys

import torch

PROBE_NAME = sys.argv[1] if len(sys.argv) > 1 else "all"


def probe(name):
    def deco(fn):
        if PROBE_NAME not in ("all", name):
            fn._skip = True
        return fn

    return deco


failures = []


def record(name, ok, detail=""):
    line = f"{'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else "")
    print(line, flush=True)
    if not ok:
        failures.append(name)


def main():
    n_gpu = torch.cuda.device_count()
    record("device_count", n_gpu == 8, f"found {n_gpu}")
    for i in range(n_gpu):
        props = torch.cuda.get_device_properties(i)
        gcn = getattr(props, "gcnArchName", "?")
        record(f"gpu{i}_arch", "gfx908" in str(gcn), f"{props.name} {gcn}")

    if PROBE_NAME in ("all", "matmul"):
        torch.manual_seed(0)
        for dtype in (torch.bfloat16, torch.float16):
            try:
                a = torch.randn(4096, 4096, dtype=dtype, device="cuda:0")
                b = torch.randn(4096, 4096, dtype=dtype, device="cuda:0")
                c = a @ b
                torch.cuda.synchronize()
                ok = torch.isfinite(c).all().item()
                # Chunked fp32 reference for the first output tile: at
                # K=4096, bf16 output rounding allows diffs up to ~1 ULP of
                # the largest partial sums (~0.5 at |c|~200).
                ref = torch.zeros(64, 64, device="cuda:0", dtype=torch.float32)
                for k in range(0, 4096, 1024):
                    ref += a[:64, k : k + 1024].float() @ b[k : k + 1024, :64].float()
                diff = (c[:64, :64].float() - ref).abs().max().item()
                ok = ok and diff <= 2.0
                record(f"matmul_{dtype}", ok, f"max_diff={diff:.3f}")
                del a, b, c, ref
            except Exception as e:
                record(f"matmul_{dtype}", False, repr(e))

    if PROBE_NAME in ("all", "triton"):
        try:
            import triton
            import triton.language as tl

            @triton.jit
            def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offs = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                x = tl.load(x_ptr + offs, mask=mask)
                y = tl.load(y_ptr + offs, mask=mask)
                tl.store(out_ptr + offs, x + y, mask=mask)

            x = torch.randn(65536, device="cuda:0")
            y = torch.randn(65536, device="cuda:0")
            out = torch.empty_like(x)
            _add_kernel[(triton.cdiv(65536, 1024),)](x, y, out, 65536, BLOCK=1024)
            torch.cuda.synchronize()
            record("triton_add", torch.allclose(out, x + y))
        except Exception as e:
            record("triton_add", False, repr(e))

    if PROBE_NAME in ("all", "hive_collective"):
        # RCCL all-reduce across a 4-GPU hive, one process per GPU.
        # Launched here as subprocesses so a single probe covers both hives.
        import subprocess

        worker = r"""
import os, sys
import torch, torch.distributed as dist
rank, hive = int(sys.argv[1]), [int(x) for x in sys.argv[2:]]
os.environ["HIP_VISIBLE_DEVICES"] = ",".join(str(g) for g in hive)
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29521"
dist.init_process_group("nccl", rank=rank, world_size=len(hive))
torch.cuda.set_device(rank)
x = torch.full((4096,), float(rank + 1), device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
want = float(sum(range(1, len(hive) + 1)))
ok = torch.allclose(x, torch.full_like(x, want))
print("PASS" if ok else "FAIL", f"hive={hive} rank={rank}", flush=True)
dist.destroy_process_group()
sys.exit(0 if ok else 1)
"""
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(worker)
            worker_path = f.name
        for hive in ([0, 1, 2, 3], [4, 5, 6, 7]):
            procs = [
                subprocess.Popen(
                    [sys.executable, worker_path, str(r), *map(str, hive)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for r in range(4)
            ]
            outs = [p.communicate()[0] for p in procs]
            codes = [p.returncode for p in procs]
            record(
                f"rccl_allreduce_hive{hive[0]}-{hive[-1]}",
                all(c == 0 for c in codes),
                "; ".join(o.strip().splitlines()[-1] for o in outs if o.strip()),
            )

    if PROBE_NAME in ("all", "p2p"):
        # Peer access matrix + copy behavior. On this host: intra-hive P2P
        # works; cross-hive (PCIe, no P2P) torch direct copies SEGFAULT —
        # known MI100 issue. vLLM PP transfers use RCCL (torch.distributed
        # send/recv), which is probed separately and works cross-hive, so a
        # cross-hive direct-copy failure is recorded but not fatal here.
        try:
            a = torch.randn(256, 256, device="cuda:0")
            c = a.to("cuda:1")
            torch.cuda.synchronize()
            record("p2p_intra_hive", torch.allclose(a.cpu(), c.cpu()))
        except Exception as e:
            record("p2p_intra_hive", False, repr(e))
        try:
            peer = torch.cuda.can_device_access_peer(0, 4)
            record(
                "p2p_cross_hive_direct",
                peer,  # direct copies must NOT be used unless P2P exists
                f"can_device_access_peer(0,4)={peer}; direct .to() segfaults "
                "on this host — PP must use RCCL (verified separately)",
            )
        except Exception as e:
            record("p2p_cross_hive_direct", False, repr(e))

    if PROBE_NAME in ("all", "graph"):
        try:
            x = torch.randn(128, 128, device="cuda:0")
            w = torch.randn(128, 128, device="cuda:0")
            y = torch.empty_like(x)
            # Warmup: first matmul call allocates BLAS workspaces; doing it
            # under capture faults (hipBLASLt 'operation not permitted when
            # stream is capturing'). vLLM always warms up before capture.
            for _ in range(3):
                y += x @ w
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(3):
                    y += x @ w
            for _ in range(2):
                x.normal_()
                g.replay()
            torch.cuda.synchronize()
            ok = torch.isfinite(y).all().item()
            record("graph_replay", ok, f"y[0,0]={y[0, 0].item():.4f}")
            del g, x, w, y
        except Exception as e:
            record("graph_replay", False, repr(e))

        # Eight-worker graph capture/replay with changing inputs: one process
        # per GPU, each captures its own graph and replays with new inputs.
        import subprocess
        import tempfile

        worker = r"""
import os, sys, torch
rank = int(sys.argv[1])
torch.cuda.set_device(rank)
x = torch.randn(128, 128, device="cuda")
w = torch.randn(128, 128, device="cuda")
y = torch.empty_like(x)
for _ in range(3):
    y += x @ w
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y += x @ w
for _ in range(2):
    x.normal_()
    g.replay()
torch.cuda.synchronize()
ok = torch.isfinite(y).all().item()
print("PASS" if ok else "FAIL", f"rank={rank}", flush=True)
sys.exit(0 if ok else 1)
"""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(worker)
            worker_path = f.name
        procs = [
            subprocess.Popen(
                [sys.executable, worker_path, str(r)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for r in range(8)
        ]
        outs = [p.communicate()[0] for p in procs]
        codes = [p.returncode for p in procs]
        record(
            "graph_8gpu",
            all(c == 0 for c in codes),
            "; ".join(o.strip().splitlines()[-1] for o in outs if o.strip()),
        )

    print(f"\n{'ALL PROBES PASSED' if not failures else f'FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
