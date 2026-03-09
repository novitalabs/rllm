# Megatron Backend for MoE Training

## 模型: Qwen3-30B-A3B
- 30.1B 总参数, 128 experts/layer, 8 active/token, 3B active params
- 48 layers, hidden_size=2048
- vocab_size=151936

## 硬件: 2x8 H200 (140 GB HBM3e)
- 节点: 10.83.115.18, 10.83.115.21
- 网络: 8x 400G RoCE v2 (mlx5_0/1/2/5/6/7/8/11)
- K8s StatefulSet: `deepswe-30b-0` (.21, head), `deepswe-30b-1` (.18, worker)

---

## 并行策略演进

### 方案一: TP=2 EP=4 SP=true DP=2 (首次成功)
- Layout per node: [TP0,EP0] [TP1,EP0] [TP0,EP1] [TP1,EP1] ...
- 每 GPU 持有 32 experts (128/4)
- 注意力层通过 TP=2 切分
- **成功通过 actor update**, update_actor: 6.9 min (FSDP 的 6.9x 加速)
- 缺点: rollout 因 enforce_eager 慢 (67 min)

### 方案二: TP=1 EP=8 SP=false DP=2 (当前方案, 已成功)
- Layout per node: [EP0] [EP1] [EP2] [EP3] [EP4] [EP5] [EP6] [EP7]
- 每 GPU 持有 16 experts (128/8)
- 注意力层完全复制在每个 GPU
- **关键**: 需要 `max_response_length=8192` + `use_fused_kernels=True`
- **Peak memory: 124.38 GB** (余量 15 GB)
- **update_actor: 4.5 min** (比 TP=2 EP=4 的 6.9 min 快 35%)
- **Step 1 总时间: 66.5 min** (rollout 60 min + update 4.5 min + log_prob 2 min)

---

## 当前配置 (方案二, 已验证成功)

### 训练脚本: `k8s/scripts/train_deepswe_30b_moe.sh`

关键参数:
```
data.max_prompt_length=4096
data.max_response_length=8192          # 16384→12288→8192 (逐步降低)
actor.ppo_max_token_len_per_gpu=12288  # 必须 >= max_prompt + max_response
actor.ppo_mini_batch_size=4
actor.ppo_micro_batch_size_per_gpu=1
use_fused_kernels=True                 # 避免 clone() 内存开销

# Actor Megatron
actor.megatron.tensor_model_parallel_size=1
actor.megatron.expert_model_parallel_size=8
actor.megatron.sequence_parallel=false
actor.megatron.use_distributed_optimizer=true
actor.megatron.param_offload=true
actor.megatron.optimizer_offload=true
actor.megatron.recompute_granularity=full
actor.megatron.recompute_method=uniform
actor.megatron.recompute_num_layers=48
actor.megatron.attention_backend=flash

# Ref Megatron
ref.megatron.tensor_model_parallel_size=1
ref.megatron.expert_model_parallel_size=8
ref.megatron.param_offload=true

# Rollout (vLLM)
rollout.tensor_model_parallel_size=4
rollout.mode=async
rollout.enforce_eager=True
rollout.gpu_memory_utilization=0.15
rollout.n=8
```

### RL 超参
```
algorithm.adv_estimator=rloo
algorithm.kl_ctrl.kl_coef=0.001
actor.optim.lr=1e-6
actor.clip_ratio_high=0.28
actor.use_kl_loss=False
actor.entropy_coeff=0.0
actor.loss_agg_mode=seq-mean-token-sum
data.train_batch_size=16
trainer.nnodes=2
```

### Agent / SWE 配置
```
rllm.env.name=swe
rllm.env.env_args.backend=docker
rllm.env.env_args.delete_image=True
rllm.agent.name=sweagent
rllm.agent.max_steps=30
rllm.agent.trajectory_timeout=3600
rllm.agent.overlong_filter=True
```

### Step 1 性能指标
```
perf/max_memory_allocated_gb: 124.38
perf/max_memory_reserved_gb:  125.11
perf/mfu/actor:               1.63%
timing_s/collect_trajectory:  3597s (60 min)
timing_s/old_log_prob:        122.8s
timing_s/update_actor:        269.7s (4.5 min)
timing_s/step:                3989.9s (66.5 min)
critic/score/mean:            0.039 (5/128 成功)
response_length/mean:         7577.8
response_length/clip_ratio:   57%
```

---

## 依赖包

### Docker 镜像: `rllm-deepswe:latest`
基于 `vllm/vllm-openai:v0.10.2`, 包含:
- megatron-core==0.16.0
- mbridge==0.15.1
- transformer-engine-cu12==2.12.0, transformer-engine==2.12.0
- transformer-engine-torch==2.12.0 (源码编译, 需 cudnn.h)
- flash-attn (Hopper)
- verl==0.6.1 (含 optimizer.py 和 rl_dataset.py 补丁)

### TE-torch 编译注意事项
```bash
CUDNN_INC=$(python3 -c "from pathlib import Path; import nvidia.cudnn; print(Path(nvidia.cudnn.__path__[0])/'include')")
CUDNN_LIB=$(python3 -c "from pathlib import Path; import nvidia.cudnn; print(Path(nvidia.cudnn.__path__[0])/'lib')")
MAX_JOBS=$(nproc) CPLUS_INCLUDE_PATH="${CUDNN_INC}" C_INCLUDE_PATH="${CUDNN_INC}" LIBRARY_PATH="${CUDNN_LIB}" \
pip install --no-build-isolation transformer-engine-torch==2.12.0
```

### verl 补丁 (两处)
1. **optimizer.py**: megatron-core 0.16 API 变更, 见 `k8s/scripts/patch_optimizer.py`
2. **rl_dataset.py**: extra_info 字段为 JSON string 而非 dict, 需反序列化

---

## NCCL 配置 (RDMA)
```bash
export NCCL_CUMEM_ENABLE=0       # 禁用 cuMem (避免 NVLS 分配问题)
export NCCL_NVLS_ENABLE=0        # 显式禁用 NVLS
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=b_manage0
export GLOO_SOCKET_IFNAME=b_manage0
```

---

## K8s 部署要点

### Docker 镜像分发
- K8s 用 containerd, 不是 Docker daemon
- `docker save | docker load` 只更新 Docker daemon, K8s 不可见
- 正确方式: `docker save rllm-deepswe:latest | ssh node 'ctr -n k8s.io images import --all-platforms -'`
- StatefulSet 设 `imagePullPolicy: IfNotPresent`

### 代码同步
```bash
rsync -avz /root/develop/ref/rllm/ root@10.83.115.21:/root/develop/ref/rllm/
rsync -avz /root/develop/ref/rllm/ root@10.83.115.18:/root/develop/ref/rllm/
```

### Entrypoint 流程 (`k8s/scripts/entrypoint.sh`)
1. 环境变量 + PYTHONPATH
2. verl patches (幂等)
3. dockerd (DinD for SWE containers)
4. Ray cluster (head: --head, worker: --address)
5. 等待所有节点就绪
6. 启动训练脚本 (head only)

### Ray 端口规划 (hostNetwork 模式)
```
worker ports: 30000-39999
metrics: 20100
runtime-env-agent: 20400
dashboard-agent-grpc: 20200
dashboard-agent-listen: 20300
```
