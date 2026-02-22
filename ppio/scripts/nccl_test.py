#!/usr/bin/env python3
"""Multi-rank NCCL test with RoCE transport and second process group creation."""
import os, time, torch, torch.distributed as dist, datetime

os.environ["NCCL_IB_DISABLE"] = "0"
os.environ["NCCL_IB_GID_INDEX"] = "3"
os.environ["NCCL_IB_HCA"] = "mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11"
os.environ["NCCL_SOCKET_IFNAME"] = "b_manage0"
os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "WARN")

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
print(f"Rank {rank}: Initializing (local_rank={local_rank}, world_size={world_size})...", flush=True)

dist.init_process_group(
    backend="nccl", rank=rank, world_size=world_size,
    timeout=datetime.timedelta(seconds=300)
)
print(f"Rank {rank}: Init complete!", flush=True)

# Test 1: allreduce on default process group
t = torch.ones(1024, device=f"cuda:{local_rank}") * rank
dist.all_reduce(t)
expected = world_size * (world_size - 1) / 2
print(f"Rank {rank}: allreduce sum = {t[0].item()} (expected {expected})", flush=True)

# Test 2: create second process group (simulates FSDP creating sub-groups)
print(f"Rank {rank}: Creating second process group...", flush=True)
pg2 = dist.new_group(list(range(world_size)))
t2 = torch.ones(1024, device=f"cuda:{local_rank}") * rank
dist.all_reduce(t2, group=pg2)
print(f"Rank {rank}: second group allreduce sum = {t2[0].item()}", flush=True)

# Test 3: bandwidth
big = torch.ones(64 * 1024 * 1024, device=f"cuda:{local_rank}")
torch.cuda.synchronize()
start = time.time()
for _ in range(3):
    dist.all_reduce(big)
torch.cuda.synchronize()
elapsed = (time.time() - start) / 3
bw = 0.256 / elapsed  # 256MB
print(f"Rank {rank}: avg 256MB allreduce in {elapsed:.3f}s = {bw:.1f} GB/s", flush=True)

dist.destroy_process_group()
if rank == 0:
    print("SUCCESS: All ranks completed!", flush=True)
