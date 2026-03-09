# Local Docker Registry

## 架构概览

本地 Docker registry 使用 **mergerfs** 将多个节点的存储合并为单一虚拟卷，对 registry 进程透明。registry 读写统一通过 `/data/registry-union` 进行，mergerfs 自动路由到各实际存储节点。

```
训练 Pod (DinD daemon)
    │  registry-mirrors: http://10.83.115.14:5000
    ▼
docker-registry-plain (port 5000, on .14)
    │  rootdirectory: /data/registry-union
    ▼
mergerfs union  (/data/registry-union)
    ├── /data/registry-cache/data    ← .14 本地 NVMe
    ├── /data/registry-nfs-10        ← .10 via NFS  (/data/registry-blobs on .10)
    ├── /data/registry-nfs-12        ← .12 via NFS  (/data/registry-blobs on .12)
    └── /data/registry-nfs-17        ← .17 via NFS  (/data/registry-blobs on .17)
```

## 存储节点

| 节点 | 路径 | 挂载方式 | 容量 | 已用 | 占比 | Blob Buckets |
|------|------|----------|------|------|------|-------------|
| .14 (本地) | `/data/registry-cache/data` | 本地 NVMe | 7.0T | 2.3T | 33% | 65 (0a, c0-ff) |
| .10 | `/data/registry-nfs-10` | NFS (`/data/registry-blobs`) | 7.0T | 3.1T | 44% | 85 (0b-3f, 80-9f) |
| .12 | `/data/registry-nfs-12` | NFS (`/data/registry-blobs`) | 7.0T | 4.8T | 69% | 32 (40-5f) |
| .17 | `/data/registry-nfs-17` | NFS (`/data/registry-blobs`) | 7.0T | 2.1T | 30% | 78 (00-09, 60-7f, a0-bf) |
| **合计 (union)** | `/data/registry-union` | mergerfs | **28T** | **12.3T** | **44%** | **260** |

> 数据截至 2026-03-09。Blob buckets 指 `docker/registry/v2/blobs/sha256/` 下的 hex prefix 目录。

## 镜像内容

10 个 repos（namanjain12 namespace），共 4622 个 tags，全部 SWE-bench r2e-gym 环境镜像：

| Repo | Tags |
|------|------|
| pandas_final | 1460 |
| numpy_final | 781 |
| pillow_final | 624 |
| orange3_final | 486 |
| aiohttp_final | 300 |
| tornado_final | 261 |
| scrapy_final | 215 |
| pyramid_final | 192 |
| datalad_final | 182 |
| coveragepy_final | 121 |

## 写入路由规则

mergerfs 使用 **`category.create=mfs`（Most Free Space）** 策略：

- 新 blob 写入时，自动选空闲空间最多的 branch
- mergerfs 自动选空闲最多的 branch（当前 .17 > .10 > .14 > .12）
- `minfreespace=100G`：某 branch 剩余 < 100G 时停止向它写入，自动切换到另一个
- 读取使用 `category.search=ff`（first-found），按 branch 顺序查找

## 配置文件

### /etc/fstab（.14 上）

```
# Registry pool: NFS from .10/.12/.17 + mergerfs union
10.83.115.10:/data/registry-blobs  /data/registry-nfs-10  nfs  rw,hard,nointr,rsize=1048576,wsize=1048576,tcp,nofail,x-systemd.automount  0  0
10.83.115.12:/data/registry-blobs  /data/registry-nfs-12  nfs  rw,hard,nointr,rsize=1048576,wsize=1048576,tcp,nofail,x-systemd.automount  0  0
10.83.115.17:/data/registry-blobs  /data/registry-nfs-17  nfs  rw,hard,nointr,rsize=1048576,wsize=1048576,tcp,nofail,x-systemd.automount  0  0

# mergerfs union: 合并本地 + 3 个 NFS 节点
/data/registry-cache/data:/data/registry-nfs-10:/data/registry-nfs-12:/data/registry-nfs-17  /data/registry-union  fuse.mergerfs  defaults,allow_other,use_ino,cache.files=partial,dropcacheonclose=true,category.create=mfs,category.search=ff,minfreespace=100G,nofail  0  0
```

### /etc/systemd/system/docker-registry-plain.service（.14 上）

```ini
[Unit]
Description=Docker Registry (plain, union-backed)
After=docker.service data-registry\x2dnfs\x2d17.mount data-registry\x2dunion.mount
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre=-/usr/bin/docker stop registry-plain
ExecStartPre=-/usr/bin/docker rm registry-plain
ExecStart=/usr/bin/docker run -d --restart=always --name registry-plain \
  -v /data/registry-union:/var/lib/registry \
  -v /data/registry-cache/config-plain.yml:/etc/docker/registry/config.yml:ro \
  --network=host registry:2
ExecStop=/usr/bin/docker stop registry-plain

[Install]
WantedBy=multi-user.target
```

### /data/registry-cache/config-plain.yml

```yaml
version: 0.1
log:
  level: warn
storage:
  filesystem:
    rootdirectory: /var/lib/registry
  delete:
    enabled: true
http:
  addr: 0.0.0.0:5000
  headers:
    X-Content-Type-Options: [nosniff]
```

### /etc/exports（.10/.12/.17 上）

```
/data/registry-blobs 10.83.115.14(rw,sync,no_subtree_check,no_root_squash)
```

## 训练 Pod 配置

`k8s/statefulset.yaml` 中：

```yaml
- name: REGISTRY_MIRROR
  value: "http://10.83.115.14:5000"
```

`k8s/scripts/entrypoint.sh` 用此变量生成 `/etc/docker/daemon.json`：

```json
{
  "registry-mirrors": ["http://10.83.115.14:5000"],
  "insecure-registries": ["10.83.115.14:5000"],
  "storage-driver": "overlay2"
}
```

## 常用操作

### 查看各节点占用

```bash
# union 视图（总量）
df -h /data/registry-union

# 各 branch 分别占用
df -h /data/registry-cache/data /data/registry-nfs-10 /data/registry-nfs-12 /data/registry-nfs-17

# 各 repo 的 tag 数
for repo in /data/registry-union/docker/registry/v2/repositories/namanjain12/*/; do
    echo "$(basename $repo): $(ls $repo/_manifests/tags/ | wc -l) tags"
done
```

### 启动 / 重启 registry

```bash
# 在 .14 上
systemctl start docker-registry-plain
systemctl restart docker-registry-plain

# 验证
curl -s http://localhost:5000/v2/_catalog | python3 -m json.tool
```

### 拉取缺失镜像到 registry

```bash
# 在 .17 上（通过 proxy 从 Docker Hub 拉取，推送到 .14:5000）
bash k8s/scripts/pull-missing-to-registry.sh 10.83.115.14:5000
```

### 向 registry 推送本地镜像

```bash
# 在 .14 上
bash k8s/scripts/push-to-registry.sh localhost:5000 namanjain12
# 或并行版本
bash k8s/scripts/push-to-registry-fast.sh localhost:5000 16 namanjain12
```

### Garbage collect（回收已删除镜像的 blob）

```bash
# 停止 registry 后 GC（需停服务避免并发写入）
systemctl stop docker-registry-plain
docker run --rm \
  -v /data/registry-union:/var/lib/registry \
  -v /data/registry-cache/config-plain.yml:/etc/docker/registry/config.yml:ro \
  registry:2 garbage-collect /etc/docker/registry/config.yml --delete-untagged
systemctl start docker-registry-plain
```

## 扩容：追加新存储节点

当前已有 .14(本地)/.10/.12/.17 共 4 节点。以新增 `.18` 节点为例：

### 1. 在 .18 上创建目录并配置 NFS

```bash
mkdir -p /data/registry-blobs
# 追加到 /etc/exports
echo '/data/registry-blobs 10.83.115.14(rw,sync,no_subtree_check,no_root_squash)' >> /etc/exports
exportfs -ra
systemctl enable nfs-server && systemctl start nfs-server
```

### 2. 在 .14 上挂载并更新 mergerfs

```bash
# 创建挂载点
mkdir -p /data/registry-nfs-18

# 追加到 /etc/fstab
echo '10.83.115.18:/data/registry-blobs  /data/registry-nfs-18  nfs  rw,hard,nointr,rsize=1048576,wsize=1048576,tcp,nofail,x-systemd.automount  0  0' >> /etc/fstab

# 更新 mergerfs 行：在分号间追加新 branch
# 将原来的:
#   /data/registry-cache/data:/data/registry-nfs-10:/data/registry-nfs-12:/data/registry-nfs-17  /data/registry-union  fuse.mergerfs ...
# 改为:
#   /data/registry-cache/data:/data/registry-nfs-10:/data/registry-nfs-12:/data/registry-nfs-17:/data/registry-nfs-18  /data/registry-union  fuse.mergerfs ...
```

### 3. 重新挂载 mergerfs

```bash
systemctl daemon-reload
umount /data/registry-union
mount /data/registry-union
# 重启 registry
systemctl restart docker-registry-plain
```

新节点自动加入写入轮换，mergerfs `mfs` 策略会优先写入空闲空间最多的节点。

## Docker Hub 镜像缓存（pull-through）

**注意**：`docker-registry-plain` (port 5000) 是纯本地存储，**不从 Docker Hub 自动缓存**。

当前方案：先在有 proxy 的节点（.17）上运行 `pull-missing-to-registry.sh` 从 Docker Hub 主动拉取，再推送到 .14:5000。
