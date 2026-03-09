# MoE 训练问题记录

## 问题 1: Actor Update OOM (已解决)

### 现象
Actor update 阶段在 `transformer_engine grouped_linear forward` (MoE expert FC1) 中 OOM:
```
torch.OutOfMemoryError: Tried to allocate 260.00 MiB.
GPU 0 total 139.80 GiB, PyTorch allocated 143.61 GiB
```
Peak memory 超过 GPU 物理容量 3+ GiB.

### OOM 位置
```
megatron/core/transformer/moe/experts.py:670 → linear_fc1
transformer_engine/pytorch/module/grouped_linear.py:188 → torch.empty()
```

### 尝试过的解法

#### TP=2 EP=4 方案 (成功)
| 参数 | 值 | 结果 |
|------|-----|------|
| max_response_length | 16384 | OOM (191 GiB) |
| + param_offload + optimizer_offload | | OOM (176.82 GiB) |
| + ref.param_offload | | OOM (161.89 GiB) |
| + gpu_memory_utilization 0.3→0.2 | | OOM (147.90 GiB) |
| + gpu_memory_utilization 0.2→0.15 | | OOM (141.38 GiB, 差 78 MiB) |
| + enforce_eager=True | | **成功** (释放 106 MiB CUDA graphs 内存) |

#### TP=1 EP=8 方案 (成功)
| 尝试 | max_response | ppo_max_token | gpu_mem_util | 结果 |
|------|-------------|---------------|-------------|------|
| #1 | 16384 | 21000 | 0.15 | OOM: 145.21 GiB allocated, 差 304 MiB |
| #2 | 16384 | 16000 | 0.15 | AssertionError: max_token_len < max_seq_len(20480) |
| #3 | 16384 | 20480 | 0.15 | expandable_segments 与 vLLM 冲突 |
| #4 | 16384 | 20480 | 0.10 | vLLM KV cache 不够: "No available memory" |
| #5 | 12288 | 16384 | 0.15 | OOM: 143.61 GiB allocated, 差 260 MiB |
| #6 | 8192 | 12288 | 0.15 | **成功!** peak 124.38 GB, update_actor 4.5 min |

### 解法总结
- `max_response_length=8192` (从 16384 逐步降低)
- `ppo_max_token_len_per_gpu=12288` (= max_prompt 4096 + max_response 8192)
- `use_fused_kernels=True` (避免 clone(), 节省显存)
- 三者缺一不可: 仅降低 response_length 到 12288 仍 OOM, 需要配合 fused_kernels

---

## 问题 2: expandable_segments 与 vLLM 不兼容

### 现象
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
→ RuntimeError: Expandable segments are not compatible with memory pool
```

### 原因
vLLM 使用自定义 CUDA memory pool, 与 PyTorch 的 expandable_segments 冲突.

### 解法
必须设 `expandable_segments:False` (或不设此选项).

---

## 问题 3: ppo_max_token_len_per_gpu 约束

### 现象
```
AssertionError: max_token_len=16000 < max_seq_len=20480
```

### 原因
`ppo_max_token_len_per_gpu` 必须 >= `max_prompt_length + max_response_length`.
因为 `use_dynamic_bsz=True` 下, 每个 GPU 至少要能放下一个完整序列.

### 解法
降低 `max_response_length` 才能降低 `ppo_max_token_len_per_gpu`.

---

## 问题 4: verl optimizer.py 不兼容 megatron-core 0.16

### 现象
megatron-core 0.16.0 修改了 optimizer API, verl 0.6.1 未适配.

### 解法
运行 `k8s/scripts/patch_optimizer.py` 在容器内修补.
现已做进 Dockerfile, entrypoint.sh 里也保留幂等补丁.

---

## 问题 5: verl rl_dataset.py JSON extra_info

### 现象
parquet 文件中 extra_info 字段存为 JSON string, verl 期望 dict.
`.get()` 调用在 string 上报错.

### 解法
补丁: 检测到 string 时 `json.loads()`.
已做进 Dockerfile 和 entrypoint.sh.

---

## 问题 6: Ray env 传播

### 现象
`ray start` 启动 worker 后, 训练脚本设置的环境变量不会传播到 worker.
导致 NCCL 配置缺失, 通信失败.

### 解法
所有环境变量必须在 `ray start` **之前**设置, 写在 entrypoint.sh 最前面.

---

## 问题 7: kubectl exec 中 PYTHONPATH 丢失

### 现象
```
kubectl exec ... -- nohup bash -c 'python3 -m rllm.trainer...'
→ ModuleNotFoundError: No module named 'rllm.trainer'
```

### 原因
nohup exec 不继承 pod 内的 shell 环境变量.

### 解法
在 exec 命令中显式传递所有需要的环境变量.

---

## 问题 8: Docker 镜像更新不生效 (containerd vs Docker daemon)

### 现象
`docker save | ssh docker load` 后, K8s pod 仍用旧镜像.

### 原因
K8s 用 containerd 管理镜像, `docker load` 只更新 Docker daemon 的 image store.
`imagePullPolicy: IfNotPresent` 下, containerd 里有旧镜像就不会更新.

### 解法
```bash
docker save rllm-deepswe:latest | ssh root@NODE 'ctr -n k8s.io images import --all-platforms -'
```

---

## 问题 9: vLLM KV Cache 不够

### 现象
`gpu_memory_utilization=0.10` 时:
```
No available memory for the cache blocks
```

### 原因
0.10 (14 GB) 扣除 vLLM 模型权重后, KV cache 空间不足.

### 解法
最低 `gpu_memory_utilization=0.15`. 更低就 KV cache 不够用.

---

## 问题 10: NCCL NVLS 分配问题

### 现象
NCCL 初始化失败, 报 NVLS 相关 OOM 或 invalid argument.

### 解法
```bash
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
```

---

## 问题 11: Docker build 路径问题

### 现象
SSH 到远程节点 build 时:
```
lstat k8s: no such file or directory
```

### 原因
SSH 默认目录是 `/root`, 不是项目目录. 相对路径无效.

### 解法
使用绝对路径:
```bash
docker build -f /root/develop/ref/rllm/k8s/Dockerfile.k8s /root/develop/ref/rllm
```

---

## 问题 12: pip 下载慢

### 现象
容器内 pip install 非常慢, 超时失败.

### 解法
在 Dockerfile 中配置 pip 镜像 (清华/中科大/华为):
```bash
mkdir -p /root/.pip && printf '[global]\nindex-url = http://pypi.tuna.tsinghua.edu.cn/simple/\n...' > /root/.pip/pip.conf
```

---

## 问题 13: transformer-engine-torch 编译慢

### 现象
TE-torch 源码编译默认单核, 在容器里要 30+ 分钟.

### 解法
设置 `MAX_JOBS=$(nproc)` 充分利用多核, 编译时间降到 ~5 分钟.
