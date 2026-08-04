---
title: 架构设计
description: NCCL 的整体分层与从 host API 到 GPU kernel 的执行模型。
weight: 1
---

理解 NCCL 的架构，关键是建立一条主线：一次 `ncclAllReduce` 如何从用户进程的 API 调用，变成 GPU 上跑的 collective kernel，中间经历了哪些层、哪些线程、哪些数据结构。本文基于 NCCL `v2.30.4-1` 源码梳理这条主线。

## 分层总览

NCCL 在工程上可以切成六层，由上到下：

```text
┌─────────────────────────────────────────────┐
│ 1. Host API 层    src/collectives.cc         │  ncclAllReduce/AllGather/...
│                   src/group.cc               │  ncclGroupStart/End
├─────────────────────────────────────────────┤
│ 2. 入队/调度层    src/enqueue.cc             │  taskAppend / scheduleCollTasksToPlan
├─────────────────────────────────────────────┤
│ 3. Communicator   src/init.cc                │  ncclCommInitRank → initTransportsRank
│                   src/bootstrap.cc           │  rank 互发现
├─────────────────────────────────────────────┤
│ 4. 拓扑图         src/graph/                 │  topo / paths / search
├─────────────────────────────────────────────┤
│ 5. Transport      src/transport/             │  p2p / shm / net / net_ib / nvls / coll_net
│   + Proxy 线程    src/proxy.cc               │  CPU proxy 推进网络传输
├─────────────────────────────────────────────┤
│ 6. Device Kernel  src/device/                │  ncclDevKernel_* / ncclKernelMain
└─────────────────────────────────────────────┘
```

一个容易踩的坑：不要按目录顺序读 NCCL。第 4、5 层交织很深（拓扑决定 transport 选型，transport 又反过来影响图搜索的代价模型），按执行链路读才不会迷路。

## 一次 AllReduce 的端到端流程

这是贯穿全节的基准流程，记住它就够了：

```text
ncclAllReduce(sendbuff, recvbuff, count, ...)
  │  仅填充 ncclInfo{func=AllReduce, chunkSteps, sliceSteps}
  ▼
ncclEnqueueCheck(&info)              src/enqueue.cc:3016
  │  隐式 ncclGroupStartInternal() → CommCheck → ArgsCheck
  ▼
taskAppend(comm, info)               src/enqueue.cc:2920
  │  AllReduce 走 collTaskAppend    src/enqueue.cc:2580
  │  → 分配 ncclTaskColl，插入 planner.collSorter（按大小分桶）
  │  → ncclGroupCommJoin 把 comm 挂进 thread-local group 头
  │  （此时任务只入队，未调度）
  ▼
（隐式）ncclGroupEndInternal()        src/group.cc:753
  │  depth 归零触发 groupLaunch → doLaunches  src/group.cc:307
  ▼
ncclLaunchPrepare(comm)              src/enqueue.cc:1513
  │  ncclTasksRegAndEnqueue：按(func,op,datatype) 聚合 → ncclGetAlgoInfo
  │  → scheduleCollTasksToPlan：把任务分布到 channel，生成 ncclKernelPlan
  ▼
ncclLaunchKernelBefore_NoUncapturedCuda  src/enqueue.cc:1670
  │  uploadWork：work 结构写进 device 可见的 work fifo / kernel args
  ▼
ncclLaunchKernel                     src/enqueue.cc:1683
  │  cuLaunchKernelEx 启动 ncclDevKernel_<coll>_<ty>_<redop>_<algo>_<proto>
  ▼
ncclLaunchKernelAfter_NoCuda         src/enqueue.cc:1784
  │  hostStreamPlanTask → uploadProxyOps + ncclProxyStart
  │  把 proxy op 投递到 ncclProxyOpsPool，唤醒 CPU proxy 推进网络收发
  ▼
（GPU）ncclKernelMain                 src/device/common.h:346
  │  blockIdx.x ↔ channel; warp 加载 comm/channel/work 到 shmem
  │  循环执行 RunWorkBatch<coll,ty,redop,algo,proto>
  │  ring: directSend → directRecvReduceDirectSend ×(k-2) → ... → directRecv
```

几个关键认知：

- **API 层极薄**。`ncclAllReduce` 本体（`src/collectives.cc:113`）只是填一个 `ncclInfo` 结构然后调 `ncclEnqueueCheck`。所有 collective 都是这个模式（AllGather/Reduce/Send/Recv 在 `collectives.cc:86-260`）。
- **调度被 group 边界驱动**。即使你不显式写 `ncclGroupStart/End`，`ncclEnqueueCheck` 也会隐式包一层 group（`enqueue.cc:3029/3053`）。真正的任务调度和 kernel launch 发生在 `ncclGroupEndInternal` 里，不是发生在 API 调用瞬间。
- **任务和 launch 是两个阶段**。`taskAppend` 只入队（进 `collSorter`），`ncclLaunchPrepare` 才做算法选择 + channel 分配 + plan 生成。这让你可以在一个 group 里攒多个 collective，统一调度。
- **CPU 和 GPU 协同**。kernel 在 GPU 上跑 ring/tree 的数据搬运和归约；CPU proxy 线程负责网络（IB/RoCE/socket）那段数据收发，因为 GPU 不能直接驱动网卡的 WQ/CQ。两者通过 `ncclProxyOpsPool`（一段共享内存 ring）交接工作。

## 六层职责详解

### 1. Host API 层

入口在 `src/collectives.cc`。每个 API 填 `ncclInfo` 后调 `ncclEnqueueCheck`。`ncclInfo` 携带 func、sendbuff/recvbuff、count、datatype、op、chunkSteps、sliceSteps 等。group 语义在 `src/group.cc`：`ncclGroupStart` 只是把 thread-local `ncclGroupDepth++`（`include/group.h:90`），`ncclGroupEnd` 在 depth 归零时才真正触发 launch。

### 2. 入队/调度层（enqueue.cc）

`src/enqueue.cc` 是 NCCL 的"大脑"，超过 3000 行，但主线清晰：

- `taskAppend`（`:2920`）：区分 p2p（Send/Recv → `p2pTaskAppend`）和 collective（→ `collTaskAppend`）。
- `collTaskAppend`（`:2580`）：分配 `ncclTaskColl`，算 `trafficBytes`，插进 `planner.collSorter`（按 `trafficBytes` 分桶的大顶堆，`comm.h:359`），把 comm 挂进 group。
- `ncclTasksRegAndEnqueue`（`:299`）：排空 `collSorter`，按 `(func,op,datatype)` 分 bin，相邻 4× 大小内的任务聚合，对每个聚合调 `ncclGetAlgoInfo` 选算法/协议。
- `ncclGetAlgoInfo`（`:2036`）：建 `collCostTable[ALGO][PROTO]`，有 tuner 插件就走插件，否则走 `topoGetAlgoInfo`（基于拓扑带宽/延迟的代价模型）。产出 `algorithm`/`protocol`/`nMaxChannels`/`nWarps`。
- `scheduleCollTasksToPlan`（`:572`）：调度核心。把任务按 channel 切成 `ncclDevWorkColl`，`calcCollChunking` 算 chunkSize/pattern，`ncclAddProxyOpIfNeeded` 把 proxy op 入 `plan.channels[c].proxyOpQueue`，设 `plan->kernelFn = ncclDevKernelForFunc[devFuncId]`。

注意：v2.30 里 `src/scheduler/` 目录只有 `allgatherv_sched.cc` 和 `symmetric_sched.cc`，通用 collective 调度是内联在 `enqueue.cc` 里的，不是一个独立模块。

### 3. Communicator 与 bootstrap

`ncclComm`（`include/comm.h:523`）是核心句柄，字段极多（后面 [communicator](communicator) 一文逐块讲）。初始化主流程 `initTransportsRank`（`src/init.cc:931`）的顺序就是 NCCL 的"启动剧本"：

```text
ncclTopoGetSystem        构建拓扑图
ncclTopoComputePaths     算 GPU/NIC 间路径矩阵
ncclTopoTrimSystem       裁剪不可达节点
ncclTopoSearchInit       准备图搜索
ncclTopoComputeCommCPU   选 proxy CPU
ncclTopoCompute(ring/tree/collnet/nvls)   搜索各算法的 channel 拓扑
setupChannel × nChannels                 把搜索结果落成 channel
ncclTransportRingConnect / TreeConnect / NvlsSetup   建连
ncclProxyConnect                         起 proxy 连接
```

在这一切之前，rank 之间要先能互相找到——这是 `src/bootstrap.cc` 干的事。rank 0 起一个 root 监听线程（`bootstrapCreateRoot`），所有 rank 向 root 报到，root 把每个 rank 和它的 ring 前/后继配对（`bootstrapRoot:353-365`），随后 rank 之间用 socket 或网络插件建立 bootstrap ring，再做 ring allgather 交换彼此的 P2P/proxy 地址（`ringAllInfo:611`）。bootstrap 完成后才有真正的 NCCL communicator。

### 4. 拓扑图（src/graph/）

把硬件抽象成图：节点是 GPU/CPU/NIC/NVSwitch/GIN，边带带宽和路径类型。`ncclTopoSystem` 是全系统图，`ncclTopoGraph` 是某算法（ring/tree/collnet/nvls）搜出来的 channel 拓扑。

路径类型（`src/graph/paths.cc`）刻画 GPU-GPU "走多远"：NVL（NVLink 直连）、PIX（同 PCIe bridge）、PXB（同 PCIe Switch 多级）、PHB（同 Host 但跨 PCIe Root）、SYS（跨 NUMA/跨节点）。NCCL 用这些类型判断 P2P 可行性和代价。图搜索（`src/graph/search.cc` 的 `ncclTopoSearchRec`）在路径类型约束下枚举 ring/tree 顺序，最大化带宽。详见 [topology-graph](topology-graph)。

### 5. Transport + Proxy

Transport 是"数据怎么搬"的抽象，分几类（`src/transport/`）：

| Transport | 场景 | 文件 |
|---|---|---|
| P2P | 同机 GPU 间，NVLink/PCIe 直传 | `p2p.cc` |
| SHM | 同机跨进程，共享内存 | `shm.cc` |
| Net | 跨节点，IB/RoCE/socket | `net.cc` + `net_ib/` |
| NVLS | NVSwitch 组播 (multimem) | `nvls.cc` |
| CollNet | SHARP/交换机内归约 | `coll_net.cc` |

`ncclTransportP2pConnect`（`src/transport.cc:45`）只设 `connectSend/connectRecv` 位掩码，真正建连是 `ncclTransportP2pSetup`（`:119`）：对每个 peer 的每个 channel，`selectTransport` 选 transport，双方交换 `ncclConnect` blob，再循环调 `transport->connect` 直到 `connected=1`。

CPU proxy（`src/proxy.cc`）存在的原因：**GPU kernel 无法直接操作网卡**。跨节点的数据必须由 CPU 线程经 RDMA/socket 收发。proxy 把"网络这一段"从 kernel 里剥出来：kernel 在 GPU 显存的 ring buffer 里读写，proxy 线程负责把 ring buffer 的数据经网卡发出去 / 从网卡收进来填进 ring buffer。主线程通过 `ncclProxyOpsPool`（一段 per-local-peer 的共享内存 ring）把 `ncclProxyOp` 投递给 progress 线程，progress 线程调对应 transport 的 `proxyProgress` 回调推进。详见 [proxy](proxy)。

### 6. Device Kernel

GPU 侧入口是 `ncclDevKernel_*`（`src/device/common.cu:23` + `common.h:420` 的 `DEFINE_ncclDevKernel` 宏），每个特化 kernel 对应一组 `<coll,ty,redop,algo,proto>`，例如 `ncclDevKernel_AllReduce_Sum_*_Ring_LL128`。host 在 `scheduleCollTasksToPlan` 里通过 `ncclDevKernelForFunc[devFuncId]` 选定一个。

kernel 主体 `ncclKernelMain`（`device/common.h:346`）做的事：

1. `blockIdx.x` ↔ channel（grid.x = `countOneBits(channelMask)`，block i 处理第 i 个置位的 channel）。
2. 第一个 warp 把 `ncclKernelComm` 拷进 shmem，第二个 warp 拷当前 channel 的 `ncclDevChannel`，其余 warp 协作 `loadWorkBatchToShmem`。
3. 循环读 `ncclDevWorkBatch`，调 `RunWorkBatch<...>::run`，按协议（Simple/LL/LL128）和算法（Ring/Tree/Collnet/Nvls）执行实际的 send/recv/reduce。
4. 每个 batch 后跟 `nextBatchIx`，-1 结束。

Ring AllReduce 的核心是 k-step ring（`src/device/all_reduce.h:14` 的 `runRing`）：`directSend → directRecvReduceDirectSend ×(k-2) → directRecvReduceCopyDirectSend(postOp) → directRecvCopyDirectSend ×(k-2) → directRecv`。详见 [kernel](kernel)。

## 为什么这么设计

几个值得记住的设计取舍：

- **channel 多路并行**：一个 collective 不只用一条 ring，而是拆到 `nChannels` 条 channel 上并行，每条 channel 独立建连、独立 proxy、独立 kernel block。这样能同时打满多条 NVLink/网卡，也利于和计算重叠。
- **算法 × 协议正交**：算法（Ring/Tree/Collnet/Nvls）决定拓扑形状，协议（Simple/LL/LL128）决定单步数据格式和同步粒度。正交组合让 NCCL 能覆盖从 8B 小消息（LL/LL128 低延迟）到 GB 大消息（Simple 高带宽）的全谱。
- **proxy 与 kernel 解耦**：GPU 做归约和 NVLink/SHM 搬运，CPU 做网络搬运，两者通过 ring buffer + OpsPool 流水线。代价是 CPU 要参与，好处是 GPU 不必碰网卡 API，且网络栈可独立演进（插件化）。
- **group 批量化**：调度延迟摊到一个 group 的多个 collective 上，相邻任务还能聚合（`ncclTasksRegAndEnqueue` 的 4× 聚合），减少 kernel launch 数和 proxy op 数。

## 进一步阅读

- 调用链逐函数追踪：[源码分析：AllReduce 调用链](allreduce-flow)
- communicator 与 group 语义：[Communicator 与 Group](communicator)
- 拓扑如何变 channel：[Topology 与图搜索](topology-graph)
- 建连状态机：[Transport 与建连](transport)
- proxy 内部：[Proxy 线程](proxy)
- kernel 执行模型：[Device Kernel](kernel)
- 读源码的方法论：[源码阅读路线](reading-guide)
