---
title: Device Kernel
description: GPU 侧 collective kernel 的执行模型与协议。
weight: 7
---

GPU 侧的 collective kernel 是 NCCL 的"出力点"：ring/tree 的数据搬运和归约都在这里发生。本文基于 `v2.30.4-1` 的 `src/device/`，拆解 kernel 的入口、block↔channel 映射、work 循环、协议原语，以及 GPU 与 CPU proxy 的协同。

## 入口 kernel

```text
ncclDevKernel_Generic(ncclDevKernelArgs4K args4K)   src/device/common.cu:23
  → ncclKernelMain<-1, RunWorkNop>(&args4K.args)
```

Generic kernel 是"通配"入口，靠运行时分派。但 NCCL 主要是用**特化 kernel**——通过 `DEFINE_ncclDevKernel` 宏（`src/device/common.h:420`）为每组 `<coll, ty, redop, algo, proto>` 预编译一个 `__global__` 函数：

```text
DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId)
  → __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K const args4K) {
        ncclKernelMain<specializedFnId,
                       RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args);
    }
```

例如 `ncclDevKernel_AllReduce_Sum_*_Ring_LL128`。host 在 `scheduleCollTasksToPlan` 里通过 `ncclDevKernelForFunc[task->devFuncId]`（`enqueue.cc:787`）选定一个。特化的好处：模板常量折叠，算法/协议分支在编译期消除，寄存器/shmem 用得更紧。特化表由 `src/device/generate.py:232` 生成。

## ncclKernelMain 的执行步骤

`ncclKernelMain`（`src/device/common.h:346`）是每个 CTA（thread block）的执行主体：

```text
1. 把 kernel args 拷进 shmem
2. block ↔ channel 映射
     n = popcll(channelMask & ((1<<tid)-1));
     if (blockIdx.x == n) channelId = tid;
   (grid.x = countOneBits(channelMask)，block i 处理第 i 个置位的 channel)
3. warp0: ncclKernelComm → shmem
   warp1: ncclDevChannel → shmem
   其余 warp: loadWorkBatchToShmem(blockIdx.x)
4. 循环直到 nextBatchIx == -1:
     profiler(START)
     SpecializedRunWorkBatch().run()   (或 ncclDevFuncTable[funcId]())
     profiler(STOP)
     loadWorkBatchToShmem(nextBatchIx)
```

### CTA 与 CGA

CTA（Cooperative Thread Array）就是 CUDA thread block。NCCL **一个 block 服务一个 channel**，所以 grid.x = 活跃 channel 数。sm90+ 可选 CGA（Cooperative Grid Array，即 Thread Block Cluster）：`cuLaunchKernelEx` 设 `config.cgaClusterSize`（`enqueue.cc:~1710`），把多个 CTA 编进一个 cluster，支持跨 SM 同步和数据交换（`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`）。这对 NVLS/LSA 等需要跨 SM 协作的算法有用。

### loadWorkBatchToShmem

`loadWorkBatchToShmem`（`common.h:135`）按 `workStorageType` 决定怎么读 `ncclDevWorkBatch`：

- `Args`：`ld.param` 从 kernel args（4KB 内联，最快）
- `Fifo`/`Persistent`：`ld.global` 从 work fifo

`ncclDevWorkBatch`（`include/device.h:381`）用 `offsetBase` + `offsetBitset`（64-bit，每个置位 bit 对应一个 work 结构在 `offsetBase + i*workSize`）紧凑编码一批 work。解码后把每个 `ncclDevWorkColl`/`P2p`/`Bcast`/`CollReg` 拷进 shmem。`nextBatchIx`（`common.h:244`）指向下一批，-1 结束。

## RunWorkBatch

`RunWorkBatch<Fn,T,RedOp,Algo,Proto>::run`（`common.h:282`）遍历 shmem 里的 work slot，对每个调 `RunWorkColl<...>().run(tid, subtn, work)`（`common.h:308`）：

```text
subtn = work->nWarps * WARP_SIZE   每个 work 参与的线程数
相邻 work 的 nWarps 变化时插 __syncthreads   common.h:300
```

`ncclDevWorkColl`（`include/device.h:262`）是每个 channel 拿到的"任务卡"：

```text
channelLo/channelHi (u8 each)   该 work 用哪些 channel
nWarps                            参与线程组数
flags: redOpArgIsPtr / regUsed / netRegUsed / oneNode / direct / isOneRPN / profilerEnabled
root
recvbuff / sendbuff
sendbuffOffset / recvbuffOffset
sendbuffRmtAddrs / recvbuffRmtAddrs   (网络 offload 时远端地址)
cbd { countLo, countMid, countHi, chunkGrainsLo/Mid/Hi }   连续字节分布
  或 collnet { count, chunkCount }
redOpArg
```

### CBD 切分

数据按 `cbd`（Continuous Byte Distribution）在 channel 间切分：`ncclCollCbdPart`（`include/device.h:333`）根据 `channelId` 和 `cbd` 算出本 channel 处理的 `(partOffset, partCount, chunkCount)`。中间 channel 处理 `countMid`，首/尾 channel 处理 `countLo`/`countHi`。grain size 由协议决定（`ncclProtoGrainSize`，`device.h:325`）：

| 协议 | grain size |
|---|---|
| LL | 16 字节 |
| LL128 | `WARP_SIZE*8/16*15*8` ≈ 1920 字节 |
| SIMPLE | 512 字节 |

## 协议

三种协议决定单步数据格式和同步粒度。原语分别在 `src/device/prims_simple.h` / `prims_ll.h` / `prims_ll128.h`。

### Simple

大 buffer + `head`/`tail` 计数器。每步搬 `stepSize` 字节（`buffSizes[SIMPLE]/NCCL_STEPS`）。适合大消息，带宽优先。`MaxGroupWidth=2`。

### LL（Low Latency）

buffer 一半当数据、一半当 flag。每条 `ncclLLFifoLine`（`device.h:76`）是 128-bit `{data, flag}`，靠 flag 同步。`MaxGroupWidth=1`。grain 16 字节，适合小消息低延迟。

### LL128

128 字节 line（`NCCL_LL128_LINESIZE=128`），每 line 16 个 uint64，其中 15 个数据 + 1 个 flag（`NCCL_LL128_DATAELEMS=15`/`LINEELEMS=16`）。`bytePerStep = WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD/...`。比 LL 带宽高，仍保持低延迟。`MaxGroupWidth` 较大。需要 Volta+（128-bit 原子）。

### 线程数

`include/device.h:95`：`NCCL_MAX_NTHREADS=640`、`NCCL_MIN_NTHREADS=128`、`NCCL_SIMPLE_MAX_NTHREADS=512`、`NCCL_LL_MAX_NTHREADS=512`、`NCCL_LL128_MAX_NTHREADS=640`。Tree 用满 640 线程；Ring Simple 加 1 个 sync warp，Tree 加 4 个（`enqueue.cc:2025`）。

## 算法实现

各算法特化分散在 `src/device/all_reduce.h`、`all_gather.h`、`reduce.h`、`reduce_scatter.h`、`broadcast.h`、`sendrecv.h`。以 AllReduce 为例（`all_reduce.h`）：

### Ring

`runRing`（`all_reduce.h:14`）构造 `Primitives<T,RedOp,FanSymmetric<1>,Direct=1,Proto,0>(tid,nthreads,&ring->prev,&ring->next,sendbuff,recvbuff,...)`，跑经典 k-step ring：

```text
directSend                                  发 k-1 步
→ directRecvReduceDirectSend ×(k-2)         收+归约+发
→ directRecvReduceCopyDirectSend(postOp)    末尾带 postOp
→ directRecvCopyDirectSend ×(k-2)           回程
→ directRecv                                收尾
```

`Direct=1` 表示用 GPU 直接 load/store 邻居 buffer（P2P/SHM 时），不经 proxy。Ring 的 `prev`/`next` 来自 `ncclDevChannel.ring`。

### Tree

`runTreeSplit`（`all_reduce.h:242`）用 `FanAsymmetric<NCCL_MAX_TREE_ARITY=3,1>`（`all_reduce.h:97`），按 `tree.up`/`tree.down[]` 做上下行归约。某些 sm80 + CUDA 11.2/11.3 用 `runTreeUpDown`。

### CollnetDirect / CollnetChain

`all_reduce.h:253`/`630`。Direct 模式 GPU 直接和 SHARP 交互；Chain 模式经 CPU proxy 把数据送进交换机归约，再取回。

### NVLS / NVLS_TREE

`all_reduce.h:389`/`522`。用 NVSwitch 的 `CUmulticast` 做硬件多播和归约，`NCCL_MAX_NVLS_ARITY=32`。NVLS_TREE 在 NVLS 之上再叠一棵树处理跨 NVSwitch 域的部分。

## GPU 与 proxy 的协同

网络段（Net transport）时，GPU kernel 和 CPU proxy 通过 `ncclConnInfo`（`device.h:131`）协同：

```text
ncclConnInfo
├── buffs[NCCL_NUM_PROTOCOLS]   每 protocol 一个 ring buffer
│     (recv 是本地显存；send 侧是远端 recv buffer 的本地镜像)
├── tail / head                 计数器
├── stepSize
├── connFifo (ncclConnFifo*)    每步 {mode, offset, size, ptr}
└── netDeviceHandle
```

- GPU 写数据进 `buffs[proto]`，推进 `head`。
- proxy 轮询 `head`，`isend` 已写部分，推进 `posted/transmitted`。
- proxy `irecv` 收数据进 recv buffer，推进 `tail`。
- GPU 轮询 `tail`，读已收部分做归约。

`NCCL_STEPS`（通常 8）是流水线深度，让 GPU 写 step N+1 与 proxy 发 step N 重叠。`connFifo` 提供更细的每步元数据（offset/size/ptr），proxy 据此知道每步搬哪段。

对 P2P/SHM（同机），`Direct=1`，GPU 直接 load/store 邻居的 `buffs[proto]`，**不经过 proxy**，延迟最低。

## Work FIFO 与回收

`comm->workFifoBuf`（host）/`workFifoBufDev`（device）是 per-comm 的环形 work fifo。`workFifoProduced`/`Consumed` 跟踪字节。每个 `ncclDevWorkBatch` 的 `offsetBase+offsetBitset` 指向 fifo 里的位置。device 处理完更新 `ncclDevChannel.workFifoDone`（`device.h:424`）。host 侧 `reclaimPlan`（`enqueue.cc:1418`）回收 fifo 槽位和 plan 内存。

## 终止

`comm->abortFlag` 传到 `ncclKernelComm.abortFlag`（`device.h:441`），kernel 在 work 循环里检查 `ncclShmem.aborted`（`common.h:379`）退出。

## 小结

- 入口是特化 kernel `ncclDevKernel_<coll>_<ty>_<redop>_<algo>_<proto>`，一个 block 服务一个 channel。
- `ncclKernelMain` 三步：映射 channel → 加载 comm/channel/work 到 shmem → 循环跑 `RunWorkBatch`。
- 协议（Simple/LL/LL128）决定数据格式和同步粒度；算法（Ring/Tree/Collnet/Nvls）决定拓扑形状，二者正交组合覆盖从小消息低延迟到大消息高带宽。
- Ring 用 k-step directSend/RecvReduceDirectSend；网络段靠 `head`/`tail` 计数器与 CPU proxy 流水线协同，同机段则 GPU 直接 load/store 邻居 buffer。
- 理解这一层就能解释 `nsys` 时间线里 `ncclDevKernel_*` 的 grid/block 维度和为什么通信 kernel 会和反向计算重叠（或没重叠）。

进一步：proxy 如何推进网络段见 [Proxy 线程](proxy)；底层硬件能力（GDR/NVLS/CE 等）见 [高级特性](features)；自己读源码的方法见 [源码阅读路线](reading-guide)。
