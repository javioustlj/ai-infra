---
title: Communicator 结构与字段地图
description: ncclComm 结构、channel、rank 模型与 device 侧镜像；group 并发语义与线程模型见 group-comm 专文。
weight: 3
---

`ncclComm` 是 NCCL 的核心句柄，几乎所有状态挂在它上面。本文基于 `v2.30.4-1` 源码拆解 `ncclComm` 的字段布局、channel 模型、device 侧镜像。group 的提交模型、join/leave 哨兵语义、非阻塞串行约束和**线程模型硬约束**是理解 NCCL 并发的钥匙，单独成文见 [Group 提交模型与 Communicator 线程模型](group-comm)——这里只保留与 `ncclComm` 字段直接相关的部分。

> **硬约束（先记）**：一个 communicator 在任意时刻只能被一个线程驱动。`ncclComm` 不是线程安全的，多线程必须各用各的 comm，不能跨线程共享同一个 comm（尤其不能在有未关闭 group 的情况下交叉使用）。违反它不会报错，而是产生静默的孤儿任务与 data race。原因详见 [group-comm 专文](group-comm#跨线程共享-comm-会怎样)。

## ncclComm 的字段地图

`struct ncclComm` 定义在 `src/include/comm.h:523`，字段非常多，但可以划成几块来记：

### 身份与拓扑

| 字段 | 含义 |
|---|---|
| `rank` / `nRanks` | 全局 rank 与总 rank 数 |
| `cudaDev` / `nvmlDev` / `compCap` / `busId` | 本 rank 绑定的 GPU |
| `node` / `nNodes` | 节点编号与节点数 |
| `localRank` / `localRanks` | 节点内本地 rank 与本节点 rank 数 |
| `rankToNode[]` / `rankToLocalRank[]` / `localRankToRank[]` | rank ↔ 节点/本地 rank 的互查表 |
| `topo` (`ncclTopoSystem*`) | 拓扑图（见 [topology-graph](topology-graph)） |
| `graphs[NCCL_NUM_ALGORITHMS]` | 每个算法搜出的 `ncclTopoGraph` |

这些字段在 `initTransportsRank`（`src/init.cc:931`）里填好，决定后续一切 channel 和 transport 选型。

### channel

```text
channels[MAXCHANNELS]   comm.h:534   MAXCHANNELS = 64   include/device.h:88
nChannels               实际建连的 channel 数
collChannels            入队用 channel 数
nvlsChannels            NVLS 可用 channel 数
p2pnChannels / p2pnChannelsPerPeer   p2p 专用 channel
```

每个 `ncclChannel`（`include/comm.h:147`）是一条独立的通信通道，自带 ring/tree/collnet/nvls 拓扑和一组 peer 连接器。channel 数由代价模型决定（小消息少用 channel，详见 [allreduce-flow](allreduce-flow) 的 `topoGetAlgoInfo`）。

### 内存与任务池

```text
memPermanent / memScoped          ncclMemoryStack，永久/按 group 分帧
memPool_ncclTaskColl/P2p/Bcast/Rma/ProxyOp/KernelPlan   各类对象的池
planner (ncclKernelPlanner)       调度状态：collSorter/collTaskQueue/planQueue...
```

`memScoped` 是 group 语义的关键：每个 group push 一个栈帧，这一批的 `ncclTaskColl`/`ncclKernelPlan` 全从该帧分配，group 结束整帧回收，无需逐个 free。`ncclMemoryStack` 这套 LIFO 帧级内存池的完整拆解（frame/hunk/unhunk 三层结构、Push/Pop 成片回收、慢路径与复用优化）见 [数据结构剖析：ncclMemoryStack](data-structures/memory-stack)。

那六个 `memPool_*` 是 `ncclMemoryPool`——建在 `memPermanent` 之上的 per-type 空闲链表，让单个任务对象能"用完放回、下次复用"而不必每 group 重新分配。它为什么必须以 `memPermanent` 为后端、Cell 类型擦除怎么工作，见 [数据结构剖析：ncclMemoryPool](data-structures/memory-pool)。

### 连接与 proxy

```text
connectSend[] / connectRecv[]     每 rank 的位掩码，标记待建连的 channel
peerInfo (ncclPeerInfo[])         每 rank 的 hostHash/pidHash/cudaDev/busId
gproxyConn                        全局 proxy 连接
proxyState (ncclProxyState*)      CPU proxy 线程状态（见 [proxy](proxy)）
bootstrap (void*)                 bootstrap 阶段的状态
```

### device 镜像

```text
devComm (ncclKernelComm*)         device 侧 communicator 指针
workFifoBuf / workFifoBufDev / workFifoProduced / workFifoConsumed
                                  work fifo（host+device+计数器）
workArgsBytes / workFifoBytes     容量
```

`devComm` 实际指向一个 `ncclKernelCommAndChannels`（`include/device.h:461`）的 `comm` 成员，这个结构后跟 `ncclDevChannel channels[MAXCHANNELS]` 数组——**host 的 channel 信息在 launch 前要镜像到 device 可见的内存**，kernel 直接读。

### 异步与终止

```text
asyncResult / abortFlag / abortFlagDev / childAbortFlag
destroyFlag / revokedFlag
config (ncclConfig_t)             blocking / cgaClusterSize / CTAPolicy / ...
groupNext[ncclGroupTaskTypeNum]   group 链表指针（0x1 哨兵=不在 group）
groupJob                          当前 group job
```

## Channel 模型

一个 channel 不是单条物理链路，而是**一组逻辑连接 + 拓扑形状**：

```text
ncclChannel (include/comm.h:147)
├── peers (ncclChannelPeer**)       每 rank 一个，host 侧连接器
├── devPeers / devPeersHostPtr      device 镜像
├── ring   (ncclRing)               prev/next/userRanks/rankToIndex
├── tree   (ncclTree)               depth/up/down[NCCL_MAX_TREE_ARITY=3]
├── collnetChain (ncclTree)
├── collnetDirect (ncclDirect)      depth/out/nHeads/headRank/heads[]/up[]/down[]
├── nvls   (ncclNvls)               out/nHeads/headRank/up[32]/down/treeUp/treeDown[3]
├── id
└── workFifoProduced
```

`ncclChannelPeer`（`include/device.h:229`）有 `send[NCCL_MAX_CONNS]` / `recv[NCCL_MAX_CONNS]` 两个连接器，`NCCL_MAX_CONNS=2`：connIndex 0 给 collective/ring/tree，connIndex 1 给 p2p。`ncclConnector`（device.h:158）持有 `transportComm`（vtable 指针）、`proxyConn`、`conn`（`ncclConnInfo`，含 `buffs[NCCL_NUM_PROTOCOLS]`、`tail/head` fifo 计数器、`stepSize`、`connFifo`）。

建一个 channel 的代价不小：每 peer 每 protocol 一个 ring buffer、一套连接器、可能的 proxy 连接。所以 channel 数是 `nChannels` 而非 `MAXCHANNELS`——按需开。

### setupChannel

`initTransportsRank` 在图搜索完成、`ncclTopoPreset`/`ncclTopoPostset` 填好 rank 顺序后，循环调 `setupChannel`（`src/init.cc:762`）：

```text
setupChannel(comm, c, rank, nranks, rings+c*nranks)
  ├─ initChannel(comm, channelId)   src/channel.cc:17  懒分配 peers/ring 数组
  ├─ ring->index, ring->userRanks[], ring->rankToIndex[]  从图搜索结果填
  └─ （nPeers = nRanks + 1(collnet) + nvlsRanks）
```

随后 `ncclTransportRingConnect` / `ncclTransportTreeConnect` / NVLS/CollNet setup / `ncclProxyConnect` 才真正建连（见 [transport](transport)）。

## Device 侧镜像

GPU kernel 不能 dereference host 指针，所以 NCCL 把 communicator 切成两层：

```text
ncclKernelCommAndChannels (include/device.h:461)
├── comm   (ncclKernelComm)
│    rank / nRanks / node / nNodes
│    buffSizes[NCCL_NUM_PROTOCOLS] / p2pChunkSize
│    abortFlag
│    channels  → 指向下面的 channels 数组
│    rankToLocalRank[]
│    workStarted / workCompleted   per-channel profiler 计数
└── channels[MAXCHANNELS] (ncclDevChannel)
     peers (ncclDevChannelPeer**) → send/recv[2] ncclConnInfo（精简版）
     ring / tree / collnetChain / collnetDirect / nvls
     workFifoDone (device 写，标记处理到哪)
     workCounter
```

`comm->devComm = &ncclKernelCommAndChannels::comm`（`include/comm.h:~672`）。kernel 启动时第一个 warp 把 `ncclKernelComm` 拷进 shmem，第二个 warp 拷当前 channel 的 `ncclDevChannel`，然后才开始干活（详见 [kernel](kernel)）。

## Group 语义（摘要）

group 是 NCCL 的批量执行单元，把"入队"和"提交"分离：group 内调 collective 只走 `taskAppend → collTaskAppend` 把任务挂进 planner，**调度和 launch 全部推迟到 `ncclGroupEndInternal`**。这带来批量调度（`ncclTasksRegAndEnqueue` 按 `(func,op,datatype)` 分 bin 并 4× 聚合，少 launch）、跨 comm 对齐（`doLaunches` 按 clique 分轮交错 launch）和统一错误回滚三个好处。

group 状态是 `thread_local` 的（`src/group.cc` 的 `ncclGroupDepth` / `ncclGroupError` / `ncclGroupCommHead[]` / `ncclGroupBlocking`），`ncclGroupStart` 只 `ncclGroupDepth++`，`ncclGroupEnd` 仅在 depth 归零时触发——所以是可嵌套计数器，隐式 group 对显式 group 透明。`ncclGroupTaskType`（`include/comm.h:506`）：`Collective=0`、`SymRegister=1`。

阻塞 vs 非阻塞由首个参与 comm 的 `config.blocking` 决定，全 group 必须一致：阻塞在当前线程跑 `groupLaunch`；非阻塞给每个参与 comm 设 `ncclInProgress`、起独立 `std::thread` 跑 `groupLaunchNonBlocking`，用户轮询 `ncclCommGetAsyncError`。

group 的提交模型、join/leave 哨兵 `0x1` 的真实语义（**幂等守卫，不是互斥锁**）、非阻塞串行约束、校验三档，以及跨线程共享 comm 的静默错误推演，单独成文见 [Group 提交模型与 Communicator 线程模型](group-comm)。那里有两条最容易被忽视的约束：

> 1. **一个 communicator 任意时刻只能被一个线程驱动**（comm 不是线程安全的，跨线程共享会静默产生孤儿任务与 data race）。
> 2. **非阻塞模式下，同一 comm 的上一个 group 完成前不能提交下一个**（`ncclCommEnsureReady` 会以 `ncclInvalidArgument` 拒绝）。

### 与 split/grow 的关系

`ncclCommSplit`（`init.cc`）复用 parent 的 `sharedRes`（含 `proxyState`、`peers`、`deviceStream`），新 comm 通过 `ncclAtomicRefCountIncrement` 共享 `ncclChannelPeer`，只新开自己需要的 channel 子集。`bootstrapSplit`（`bootstrap.cc:866`）用 parent 的 bootstrap 环分发新 handle。grow 走 `bootstrapInit` + `isGrow=true`。Split comms 共享 proxy 线程，省了重新建连的开销。

## bootstrap：rank 互发现

在 communicator 真正建连之前，rank 之间要互相找到。`src/bootstrap.cc` 做这件事，特点是用一个临时 socket 环（或网络插件）把 rank 拉到一起，完成后这套 bootstrap 通道会被复用做后续的 transport 建连信息交换。

```text
ncclGetUniqueId(commId)            bootstrap.cc:426
  rank 0: bootstrapCreateRoot(handle)  起监听线程 bootstrapRoot
  其他: 若 NCCL_COMM_ID 则解析 host:port，否则用 handle

bootstrapInit(nHandles, handles, comm, parent)   bootstrap.cc:674
  ├─ 分配 bootstrapState，comm->bootstrap = state
  ├─ 起 ring 邻居监听 socket + root 回连 socket
  ├─ sendToRoot(extInfo)            bootstrap.cc:764
  │    root 把每个 rank 与其 ring 前/后继配对 (bootstrapRoot:353-365)
  ├─ 建立 ring: state->ring.send/recv（socket 或 netRingConnect）
  ├─ ringAllInfo                     bootstrap.cc:611
  │    打包 (peerP2pAddress, peerProxyAddress, peerProxyUDS, rasRank)
  │    ring allgather → 每个 rank 拿到所有 rank 的 P2P/proxy 地址
  └─ ncclProxyInit(...)              bootstrap.cc:844
       装好本 rank 的 proxy 服务监听
```

之后所有 rank 间信息交换（transport 建连、`ncclConnect` blob 交换）都用 `bootstrapSend/Recv`（`bootstrap.cc:964/1058`，连 `peerP2pAddresses[peer]`，带 tag 区分消息）。`bootstrapBarrier`（`:1196`）用 dissemination 算法（对每个 mask 与 `rank±mask` 收发 4 字节）。

多 root（`nroots>1`）支持大规模：rank r 联系 `rootIdFromRank(r)`，每个 root 管一段连续 rank，缓解单 root 瓶颈。`StaggerThreshold=256` 以上错峰连接，避免 thundering herd。

## 小结

- `ncclComm` 是个"大杂烩"句柄，但结构上清晰：身份/拓扑、channel、内存池、连接、device 镜像、group/异步。
- channel 是逻辑通道，自带多种拓扑形状，按需开 `nChannels` 条，每条独立 proxy/连接器。
- host 侧 channel 状态在 launch 前镜像到 `ncclKernelCommAndChannels`，kernel 直接读 device 内存。
- group 把"入队"和"提交"分离，支持批量调度、跨 comm 对齐、嵌套与回滚，是 NCCL 性能和正确性的核心机制。
- bootstrap 用临时 socket 环（或网络插件）把各 rank 拉到一起，完成后复用做 transport 建连的信息交换。

进一步：建连状态机见 [Transport 与建连](transport)，proxy 内部见 [Proxy 线程](proxy)。
