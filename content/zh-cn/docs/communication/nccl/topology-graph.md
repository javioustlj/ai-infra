---
title: Topology 与图搜索
description: NCCL 如何把硬件拓扑变成 ring/tree channel。
weight: 4
---

NCCL 在初始化时不直接用固定的通信模式，而是**先把硬件抽象成图，再在图上搜索最优的 ring/tree/collnet/nvls 拓扑**，最后落成 channel。这套机制让 NCCL 能适应从单机 8 卡到跨节点大规模的各种拓扑。本文基于 `v2.30.4-1` 的 `src/graph/` 目录。

## 三个层次，一句话区分

```text
硬件 / XML / 插件
  → ncclTopoSystem         "机器能怎么走"（物理拓扑图）
  → ncclTopoCompute()      搜索 → ncclTopoGraph   "NCCL 选择怎么走"（算法拓扑）
  → comm->channels         "运行时按哪个邻居发收"（实际连接）
```

## 节点与边

`ncclTopoSystem`（`src/graph/topo.h:147`）持有全系统图。节点类型（`topo.h:43`）：

| 类型 | 含义 |
|---|---|
| `GPU(0)` | GPU，带 rank、compCap、pciPath、gdrSupport |
| `PCI(1)` | PCIe bridge/switch |
| `NVS(2)` | NVSwitch |
| `CPU(3)` | NUMA 域 |
| `NIC(4)` | 网卡 |
| `NET(5)` | 网络插件设备（IB/RoCE 等） |
| `GIN(6)` | GPU-Initiated Network 设备 |
| `DEV(7)` | 其他设备 |

`NCCL_TOPO_MAX_NODES=640`（`include/graph.h:106`，注释说支持到 576 GPU，如 NVLD144）。每个 `ncclTopoNode`（`topo.h:94`）有 `nlinks` + `links[NCCL_TOPO_MAX_LINKS]`（`ncclTopoLink{type,bw,remNode}`）和到各类型节点的 `paths[NCCL_TOPO_NODE_TYPES]`（`ncclTopoLinkList{count,bw,list[]}`）。

链路类型（`topo.h:55`）：`LINK_LOC / NVL / C2C / PCI / SYS / NET`。

## 路径类型：GPU 之间走多远

`ncclTopoComputePaths`（`src/graph/paths.cc:685`）对每对 GPU/GPU-NIC 算一条路径类型，这是选 transport 和判断 P2P 可行性的依据。路径类型（`include/graph.h:111`）：

| 类型 | 含义 | P2P? |
|---|---|---|
| `PATH_LOC(0)` | 自身 | — |
| `PATH_NVL(1)` | NVLink 直连 | ✓ |
| `PATH_NVB(2)` | 经中间 GPU 的 NVLink | ✓ |
| `PATH_C2C(3)` | CPU↔GPU C2C | ✓ |
| `PATH_PIX(4)` | 同一 PCIe bridge | ✓ |
| `PATH_PXB(5)` | 多级 PCIe switch，不跨 PHB | ✓ |
| `PATH_P2C(6)` | GPU↔NIC 经 C2C+PCIe | — |
| `PATH_PXN(7)` | GPU↔NIC 经中间 GPU（rail-local） | — |
| `PATH_PHB(8)` | 经 PCIe Host Bridge / CPU | 受限 |
| `PATH_SYS(9)` | 跨 NUMA / UPI / QPI | 通常不可 P2P |
| `PATH_NET(10)` | 网络 | — |
| `PATH_DIS(11)` | 不连通 | — |

`ncclTopoCheckP2p`（`paths.cc:282`）返回两 rank 间的路径类型，直接决定能否用 P2P transport。`NCCL_P2P_LEVEL` 环境变量可强制拉低可用层级。`ncclTopoCheckGdr`/`ncclTopoIsGdrAvail`（`paths.cc:441/518`）判断 GPUDirect RDMA 可行性；`ncclTopoCheckMNNVL`（`:412`）判多节点 NVLink。

## 图的构建

```text
ncclTopoGetSystem(comm, &comm->topo)    src/graph/topo.cc:1521
  ├─ 读 XML 拓扑（/var/run/nvidia-topologyd/virtualTopology.xml 或 NCCL_TOPO_FILE）
  ├─ 填本机 GPU 节点
  ├─ 各节点 XML 经 bootstrap allgather 汇总
  ├─ ncclTopoAddGpu/AddNic/AddPci/AddCpu/AddNvLinks/AddPciLinks/AddC2c/AddNet/AddGin
  └→ ncclTopoSystem
```

XML 节点上限 `NCCL_TOPO_XML_MAX_NODES=256`，graph XML 上限 `NCCL_GRAPH_XML_MAX_NODES=65536`（可 dump/load 复现拓扑）。带宽常量在 `topo.h:28`（如 `P9_BW=32.0`、`NET_BW=12` 表示 100Gbit、`INTEL_P2P_OVERHEAD(bw)=bw*6/5`）。

## 图搜索：把拓扑变成算法拓扑

`ncclTopoCompute`（`src/graph/search.cc`，声明 `graph.h:178`）是搜索入口。初始化时 `initTransportsRank` 对四种 `ncclTopoGraph` 各调一次（`init.cc:1138-1184`）：

```text
ring graph            id=0  pattern=NCCL_TOPO_PATTERN_RING
tree graph            id=1  pattern=NCCL_TOPO_PATTERN_BALANCED_TREE
collnet chain graph   id=2  (config.collnetEnable 时)
collnet direct graph  id=4
nvls graph            id=3  pattern=NCCL_TOPO_PATTERN_NVLS
```

`ncclTopoGraph`（`include/graph.h:158`）的关键输出字段：

```text
nChannels             搜出来的 channel 数
bwIntra / bwInter     节点内/跨节点带宽
latencyInter
typeIntra / typeInter 路径类型
sameChannels / crossNic
intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES]  每 channel 的本地 rank 顺序
inter[MAXCHANNELS*2]                     每 channel 的两端网络点
```

搜索核心是 `ncclTopoSearchRec`（`search.cc`），递归枚举 GPU 顺序，在路径类型约束下最大化带宽：

```text
ncclTopoSearchInit            search.cc:39    初始化
ncclTopoCompute(topo, graph)  search.cc:1023
  └─ ncclTopoSearchRec        递归
       ├─ ncclTopoSearchRecGpu   search.cc:578   GPU 序列搜索
       │    └─ ncclTopoSearchNextGpuSort  search.cc:244  候选下一 GPU 排序
       └─ ncclTopoSearchRecNet    search.cc:671   网络段搜索
ncclTopoSearchTryNvls          search.cc:388    NVLS 分支
ncclTopoSearchTryCollnetDirect search.cc:352   CollNet Direct 分支
ncclTopoCompareGraphs          search.cc:421    保留更优图
ncclTopoReplayGetGpu           search.cc:325    图复用（相似拓扑）
```

搜索目标是找一条能最大化瓶颈带宽的环/树排列。`ncclTopoSearchParams`（`search.cc:772`）按 pattern 给出 `backToNet/backToFirstRank` 等约束。

### Ring 与 Tree 构造

- Ring：`ncclBuildRings`（`src/graph/rings.cc:29`）按 prev/next 生成 `nrings` 条环序。
- Tree：`ncclGetBtree` / `ncclGetDtree`（`src/graph/trees.cc:32/89`）建二叉/双二叉树。`NCCL_MAX_TREE_ARITY=3`，顶层 arity 2（`include/device.h:188`）。

### 网卡选择

`ncclTopoSelectNets`（`search.cc:510`）选跨节点用的网卡。两种偏好：

- `ncclTopoPrefNetsGpuFirst`（`:443`）：GPU → NIC（rail-local，常见于 HPC）
- `ncclTopoPrefNetsChannelFirst`（`:478`）：NIC → GPU

`PXN`（PATH_PXN）允许 GPU 经中间 GPU 到 NIC，提升 rail-local 架构的吞吐。NIC 融合由 `ncclTopoNetInfo.mergeLevel/forceMerge` 控制，把多网卡合成一个逻辑 NET 节点。

### NVLS / CollNet 分支

- NVLS（NVLink SHARP）：`ncclTopoSearchTryNvls` 在 NVSwitch 的 GPU 组上搜可做多播/归约的 channel，要求 CUDA 12.1+。
- CollNet（SHARP）：`ncclTopoSearchTryCollnetDirect` 把归约下推到交换机，分 chain（经 CPU proxy）和 direct（GPU 直接）两种。

## 从图到 channel

搜索完成后，`ncclTopoPreset`（`src/init.cc:1241`）和 `ncclTopoPostset`（`init.cc:1433`）把搜索结果落成 channel：

```text
ncclTopoPreset    init.cc:1241
  填 ncclTopoRanks (graph.h:184) per channel:
    ringRecv/Send/Prev/Next, treeToParent/Child0/Child1,
    nvlsHeads, nvlsHeadNum
ncclTopoPostset   init.cc:1433
  跨节点 allgather → 写 channel->ring.userRanks 和 tree children
```

随后 `setupChannel`（`init.cc:762`）把 `ring->userRanks[]` / `rankToIndex[]` 填进去，`ncclTransportRingConnect` / `ncclTransportTreeConnect` 据此建连（见 [transport](transport)）。

## 代价模型与算法选择

图搜索产出的是"能怎么走"，运行时选哪个算法走由代价模型定。`src/graph/tuning.cc` 负责建模：

```text
ncclTopoInitTunerConstants   tuning.cc:231   按 compCap 填 baseLatencies/hwLatencies/llMaxBws
ncclTopoTuneModel            tuning.cc:238   填 comm->bandwidths[func][algo][proto]
ncclTopoGetAlgoTime          tuning.cc:593
  time = lat*latCount + nBytes/(1000*bw)
  latCount = numPipeOps (ring) 或 DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS)
  (有 tree/allreduce、ring plateau、NVLS_TREE 的修正因子)
```

`comm->bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS]`（`comm.h:639`）是建好的查找表。运行时 `topoGetAlgoInfo`（`enqueue.cc:1940`）用它选最小时间的 (algo,proto)，并按数据量调 channel 数（小消息少 channel）。

常量来自 `ncclTunerConstants_v5_t`（`include/plugin/tuner/tuner_v5.h:34`），按 Volta/Ampere/Hopper/Blackwell 分档。有 tuner 插件时 `comm->tuner->getCollInfo` 可覆盖（见 [plugin-abi](plugin-abi)）。

## 调试与可观测

- `NCCL_DEBUG=INFO` 打印 `topo` 子系统日志，能看到搜索选了什么路径类型、几个 channel。
- `NCCL_DEBUG_SUBSYS=INIT,GRAPH` 限定子系统。
- 代价表可用 `NCCL_ALGO`/`NCCL_PROTO` 过滤，强制选某算法/协议验证。
- topology XML 可通过 `NCCL_TOPO_FILE` 指定外部文件，复现/对比不同集群的选路。

## 小结

- 拓扑层三件套：`ncclTopoSystem`（物理图）→ `ncclTopoGraph`（算法拓扑）→ `channels`（连接）。
- 路径类型（NVL/PIX/PXB/PHB/SYS/...）是选 transport 和判 P2P 的核心。
- 图搜索在路径约束下最大化带宽，支持 ring/tree/collnet/nvls 四种 pattern，按 clique 规模选多 root/bootstrap 错峰。
- 代价模型（`tuning.cc` 的 `time = lat*latCount + nBytes/bw`）在运行时按数据量选 algo/proto 和 channel 数。
- 理解这一层就能解释 `NCCL_DEBUG=INFO` 里的路径选择日志和为什么同样代码在不同集群上跑出不同带宽。

进一步：channel 建连状态机见 [Transport 与建连](transport)。
