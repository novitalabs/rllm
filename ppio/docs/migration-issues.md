# 集群迁移问题记录：4节点(.18/.21/.22/.23) → 2节点(.14/.17)

迁移日期：2026-03-02

## 背景

将 DeepSWE 32B RL 训练从 4 节点 K8s 集群（.18/.21/.22/.23, 32 GPUs）迁移到 2 节点（.14/.17, 16 GPUs）。目标节点此前由 **sealos** 管理 K8s，残留了大量非标准配置。

---

## Issue 1: kubeadm init 超时 — Calico iptables 规则阻塞 localhost 流量

**现象**: `kubeadm init` 在 `wait-control-plane` 阶段卡死 4 分钟后报错：
```
The HTTP call equal to 'curl -sSL http://127.0.0.1:10248/healthz' returned error:
  Get "http://127.0.0.1:10248/healthz": context deadline exceeded
```

kubelet 进程已运行且绑定 10248 端口（`ss -tlnp` 确认），kube-apiserver/etcd/scheduler/controller-manager 容器均已启动，但 `curl http://127.0.0.1:10248/healthz` 返回 HTTP 000（无响应）。

**根因**: 目标节点之前运行 sealos 管理的 K8s，残留了 **Calico iptables 规则**和 **IPVS 配置**。`iptables -L INPUT` 第一条就是 `cali-INPUT` 链，拦截所有入站流量。加上 `KUBE-IPVS-OUT-FILTER` 在 OUTPUT 链中，以及 `KUBE-FIREWALL` 规则，共同导致 localhost 流量被阻塞。

关键规则：
```
Chain INPUT: → cali-INPUT (pos 1, catches everything)
Chain OUTPUT: → cali-OUTPUT → KUBE-IPVS-OUT-FILTER → KUBE-FIREWALL
KUBE-FIREWALL: -j DROP (block incoming localnet connections, conntrack !RELATED,ESTABLISHED,DNAT)
KUBE-IPVS-FILTER: -j REJECT (conntrack NEW, match-set KUBE-IPVS-IPS)
```

**解决**: 彻底清除所有残留网络规则：
```bash
# 清除 iptables (legacy + nftables)
iptables -F && iptables -X
iptables -t nat -F && iptables -t nat -X
iptables -t mangle -F && iptables -t mangle -X
iptables -t raw -F && iptables -t raw -X
nft flush ruleset

# 清除 IPVS
ipvsadm --clear

# 清除 ipsets (Calico 创建的)
for set in $(ipset list -n); do ipset destroy $set; done

# 清除 Calico 虚拟网络接口
ip link delete tunl0 2>/dev/null
ip link delete vxlan.calico 2>/dev/null
ip link delete kube-ipvs0 2>/dev/null
```

**两台节点都需执行。**

---

## Issue 2: CNI 插件缺失 — `/opt/cni/bin/` 为空

**现象**: kubelet 持续报错：
```
"Container runtime network not ready" networkReady="NetworkReady=false reason:NetworkPluginNotReady
  message:Network plugin returns error: cni plugin not initialized"
```

`/opt/cni/bin/` 和 `/etc/cni/net.d/` 均为空目录。

**根因**: sealos 自行管理 CNI（Calico），卸载/清理后 CNI 二进制文件也没了。标准 kubeadm 安装会自带 CNI 插件，但这些节点是 sealos 预置的，从未单独安装过 CNI plugins。

**解决**: 从已有节点(.18)复制 CNI 插件：
```bash
# 在 .18 上打包
ssh 10.83.115.18 "tar czf /tmp/cni-plugins.tgz -C /opt/cni/bin ."

# 分发到 .14 和 .17
scp 10.83.115.18:/tmp/cni-plugins.tgz /tmp/
for node in 10.83.115.14 10.83.115.17; do
    scp /tmp/cni-plugins.tgz $node:/tmp/
    ssh $node "mkdir -p /opt/cni/bin && tar xzf /tmp/cni-plugins.tgz -C /opt/cni/bin/"
done
```

也可从 GitHub 下载：`https://github.com/containernetworking/plugins/releases/download/v1.6.1/cni-plugins-linux-amd64-v1.6.1.tgz`

---

## Issue 3: 僵尸 sandbox 无法删除 — CNI 未初始化的循环依赖

**现象**: 残留了一个 3 周前的 nvidia-device-plugin sandbox (`a226f059a73f3`)，`crictl rmp -f` 删不掉：
```
rpc error: code = Unknown desc = failed to forcibly stop sandbox "a226f059a73f3...":
  failed to destroy network for sandbox "a226f059a73f3...": cni plugin not initialized
```

sandbox 需要 CNI 来拆除网络命名空间，但 CNI 未初始化（因为 `/etc/cni/net.d/` 没有配置文件）。CNI 配置文件要等 Flannel 安装后才有，Flannel 安装要等 kubeadm init，kubeadm init 又被这个 sandbox 的清理循环卡住。

**解决**: 写一个临时 CNI 配置，重启 containerd，sandbox 自然消失：
```bash
# 写临时 CNI 配置
cat > /etc/cni/net.d/10-bridge.conflist <<EOF
{
  "cniVersion": "1.0.0",
  "name": "bridge",
  "plugins": [
    {"type": "bridge", "bridge": "cni0", "isGateway": true, "ipMasq": true,
     "ipam": {"type": "host-local", "subnet": "10.244.0.0/24", "routes": [{"dst": "0.0.0.0/0"}]}},
    {"type": "loopback"}
  ]
}
EOF

# 重启 containerd（3周前的sandbox状态在重启后被清理）
systemctl restart containerd
sleep 3

# 确认 sandbox 已消失
crictl pods  # 应为空

# 删除临时配置（Flannel 会安装自己的）
rm -f /etc/cni/net.d/10-bridge.conflist
```

---

## Issue 4: image-cri-shim 劫持镜像引用 — sealos 残留组件

**现象**: containerd 配置中 `sandbox_image = "sealos.hub:5000/pause:3.10"`，且 kubelet 启动参数包含 `--image-service-endpoint=unix:///var/run/image-cri-shim.sock`。`sealos.hub` 解析到 `10.83.115.10`（一台无关的节点），虽然镜像已缓存在本地 containerd 中。

**根因**: sealos 部署了 `image-cri-shim` 服务，作为 CRI 图像服务的中间层，将标准镜像引用（如 `registry.k8s.io/*`）重写为 `sealos.hub:5000/*`。禁用 sealos 后该 shim 仍在运行。

**解决**: 四步清理：

```bash
# 1. 停止并禁用 image-cri-shim 服务
systemctl stop image-cri-shim
systemctl disable image-cri-shim

# 2. 从 kubelet 配置中移除 shim 引用
sed -i 's|--image-service-endpoint=unix:///var/run/image-cri-shim.sock||g' \
    /etc/systemd/system/kubelet.service.d/10-kubeadm.conf
systemctl daemon-reload

# 3. 修复 containerd 配置中的 sandbox_image
sed -i 's|sealos.hub:5000/pause:3.10|registry.k8s.io/pause:3.10|g' /etc/containerd/config.toml
sed -i '/sealos.hub:5000/d' /etc/containerd/config.toml
systemctl restart containerd

# 4. 修复 crictl 配置（也指向了 shim）
cat > /etc/crictl.yaml <<EOF
runtime-endpoint: unix:///var/run/containerd/containerd.sock
image-endpoint: unix:///var/run/containerd/containerd.sock
timeout: 10
EOF
```

**注意**: .17 也有同样问题，需在两个节点上都执行。

---

## Issue 5: Docker 镜像只在 Docker store，K8s 找不到

**现象**: StatefulSet pod 创建后报 `ImagePullBackOff`：
```
Failed to pull image "docker.io/library/rllm-deepswe:latest":
  pull access denied, repository does not exist or may require authorization
```

但 `docker images` 能看到镜像。

**根因**: K8s 使用 containerd 而非 Docker 作为 CRI。`docker load` 将镜像导入 Docker 的镜像存储，而 K8s 的 kubelet 通过 containerd 拉取镜像，两者是独立的存储。containerd 的 `k8s.io` namespace 中没有该镜像。

**解决**: 从 Docker 导出并导入 containerd：
```bash
docker save rllm-deepswe:latest | ctr -n k8s.io images import --base-name docker.io/library/rllm-deepswe -

# 验证
crictl images | grep rllm
```

**两台节点都需执行。** 后续部署建议直接用 `ctr` 导入，跳过 Docker。

---

## Issue 6: Ray GCS 崩溃 — 残留 session 状态冲突

**现象**: Head pod 启动后 Ray 报 `Ray runtime started`，但 2 秒后 GCS 崩溃：
```
ray_syncer_bidi_reactor_base.h:190: Check failed: !msg->node_id().empty()
```

Worker 无法连接到 head:6379。`ss -tlnp | grep 6379` 无输出（GCS 已不在）。

**根因**: `/tmp/ray/` 目录下残留了之前失败的 kubeadm init 尝试中的 Ray session 数据。新的 GCS 启动后读取了旧 session 的持久化状态，收到了一个 `node_id` 为空的消息，触发了 assert 失败。

**解决**:
```bash
# 在两台节点上清除 Ray 状态
ssh 10.83.115.14 "rm -rf /tmp/ray/"
ssh 10.83.115.17 "rm -rf /tmp/ray/"

# 重启所有 pods（同时删除，不要 rolling）
kubectl delete pods -n deepswe --all
```

**教训**: 每次重建 K8s 集群或重启训练前，都要清理 `/tmp/ray/`。

---

## Issue 7: entrypoint.sh EXPECTED_NODES 未更新

**现象**: Head pod 的 phase-3 等待 4 个 Ray 节点，但只有 2 个节点：
```
[phase-3] Waiting for nodes: 2/4 (60s)...
```

等待 300 秒后才跳过，浪费 5 分钟。

**根因**: `k8s/scripts/entrypoint.sh` 中 `EXPECTED_NODES=4` 硬编码，迁移时忘记改为 `2`。

**解决**: `entrypoint.sh` 第 166 行：`EXPECTED_NODES=4` → `EXPECTED_NODES=2`

---

## Issue 8: Docker Hub Rate Limit — 冷缓存 + 无认证

**现象**: 训练启动后所有 trajectory 的 Docker 容器创建失败：
```
toomanyrequests: You have reached your unauthenticated pull rate limit.
AttributeError: 'NoneType' object has no attribute 'id'  # container = None
```

registry mirror 已配置但仅缓存了 8 个仓库，大部分是 cache miss。

**根因**: 两个问题叠加：
1. 新部署的 registry mirror 缓存为空（冷启动），每个 cache miss 都需要从 Docker Hub 拉取
2. `dockerhub-creds` K8s secret 不存在（旧集群的 secret 没有迁移），DinD 和 registry mirror 都以匿名身份访问 Docker Hub（100 pulls/6h 限制）

**解决**:

```bash
# 1. 创建 K8s secret（从 .14 上 ~/.docker/config.json 获取凭证）
kubectl create secret generic dockerhub-creds -n deepswe \
    --from-literal=username=novitalabs \
    --from-literal=token=<REDACTED>

# 2. 给 registry mirror 也加上 DockerHub 认证（提升 upstream 拉取限额）
# 在 k8s/registry/config.yml 的 proxy 部分添加：
proxy:
  remoteurl: https://registry-1.docker.io
  username: novitalabs
  password: <REDACTED>

# 3. 更新并重启 registry mirror
scp config.yml 10.83.115.14:/data/registry-cache/config.yml
ssh 10.83.115.14 "docker restart registry-mirror"

# 4. 重启训练 pods（让 DinD 读取新的 DOCKERHUB_* 环境变量）
kubectl delete pods -n deepswe --all
```

**说明**: Pod entrypoint 已有 DockerHub 登录逻辑（当 `DOCKERHUB_USERNAME` 和 `DOCKERHUB_TOKEN` env 存在时自动 `docker login`），只需确保 K8s secret 存在即可。

---

## Issue 9: .17 代码目录不存在

**现象**: `rsync` 失败：
```
rsync: [Receiver] mkdir "/root/develop/ref/rllm" failed: No such file or directory (2)
```

**根因**: .17 是新节点，从未有过 `/root/develop/ref/` 目录。

**解决**: 先建目录再同步：
```bash
ssh 10.83.115.17 "mkdir -p /root/develop/ref"
rsync -az /root/develop/ref/rllm/ 10.83.115.17:/root/develop/ref/rllm/
```

---

## Issue 10: 旧节点残留进程未清理

**现象**: 迁移到新集群后，旧的 4 台节点上仍有大量 Ray 进程在运行（raylet, WorkerDict, vLLMHttpServer, dashboard agent 等），占用 GPU 显存和 CPU。.23 上还有 230+ 个僵尸进程（来自更早的训练 run）。

**根因**: 之前只删除了 K8s pods，但这些节点上的 K8s 集群已停用（pods 运行在旧的 K8s 集群中，而我们在新节点上建了新集群）。旧 pods 被删除时，Ray 进程是通过 entrypoint.sh 的 bash 启动的，K8s 删除 pod 后 containerd 杀掉了容器，但由于 `hostNetwork: true`，Ray 的子进程直接运行在 host 上未被完全清理。

**解决**:
```bash
for node in 10.83.115.18 10.83.115.21 10.83.115.22 10.83.115.23; do
    ssh $node 'ray stop --force 2>/dev/null; rm -rf /tmp/ray/'
done
```

**注意**: 僵尸进程（zombie/defunct）无法被 kill，需要父进程退出后由 init 回收。不占 CPU/GPU 资源，无害。

---

## 迁移清单总结

基于以上问题，sealos 节点迁移到标准 kubeadm 的完整清理步骤：

```bash
# === 在每个目标节点上执行 ===

# 1. 停止 sealos 残留服务
systemctl stop image-cri-shim && systemctl disable image-cri-shim

# 2. 清理 kubelet 配置
sed -i 's|--image-service-endpoint=unix:///var/run/image-cri-shim.sock||g' \
    /etc/systemd/system/kubelet.service.d/10-kubeadm.conf
systemctl daemon-reload

# 3. 修复 containerd 配置
sed -i 's|sealos.hub:5000/pause:3.10|registry.k8s.io/pause:3.10|g' /etc/containerd/config.toml
sed -i '/sealos.hub:5000/d' /etc/containerd/config.toml
systemctl restart containerd

# 4. 修复 crictl 配置
cat > /etc/crictl.yaml <<EOF
runtime-endpoint: unix:///var/run/containerd/containerd.sock
image-endpoint: unix:///var/run/containerd/containerd.sock
timeout: 10
EOF

# 5. kubeadm reset
kubeadm reset -f
rm -rf /etc/kubernetes/manifests/* /etc/kubernetes/pki/* /var/lib/etcd/

# 6. 清除所有 iptables/nftables/IPVS 规则
iptables -F && iptables -X
iptables -t nat -F && iptables -t nat -X
iptables -t mangle -F && iptables -t mangle -X
nft flush ruleset
ipvsadm --clear
for set in $(ipset list -n); do ipset destroy $set; done

# 7. 清除 Calico 网络接口
ip link delete tunl0 2>/dev/null
ip link delete vxlan.calico 2>/dev/null
ip link delete kube-ipvs0 2>/dev/null

# 8. 安装 CNI 插件（如果 /opt/cni/bin/ 为空）
# tar xzf cni-plugins.tgz -C /opt/cni/bin/

# 9. 清除 Ray 状态
rm -rf /tmp/ray/

# === 然后正常 kubeadm init ===
```
