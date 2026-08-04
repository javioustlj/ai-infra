---
title: 源码分析：AllReduce 调用链
description: 从 ncclAllReduce 到 GPU kernel 的逐函数源码追踪。
weight: 2
---

本文按 NCCL `v2.30.4-1` 源码逐函数追踪一次 `ncclAllReduce` 的完整执行路径。是后续所有子系统分析的入口——把这条链记牢，再读其他模块就有了锚点。文件路径以 `nccl/` 仓库根为基准。

## 总览：五个阶段

```text
① API 入队     ncclAllReduce → ncclEnqueueCheck → taskAppend → collTaskAppend
② Group 提交   (隐式)ncclGroupEndInternal → doLaunches
③ 调度建 Plan  ncclLaunchPrepare → ncclTasksRegAndEnqueue → scheduleCollTasksToPlan
④ Launch       ncclLaunchKernelBefore → ncclLaunchKernel → ncclLaunchKernelAfter
⑤ 设备执行     ncclDevKernel_* → ncclKernelMain → RunWorkBatch
```

前四个阶段在 host 线程，第五个在 GPU。下面逐段展开。

## ① API 入队

```text
ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
  src/collectives.cc:113
```

它只做三件事：填一个 `ncclInfo`（`src/include/info.h:17`，字段 `coll=ncclFuncAllReduce`、`opName="AllReduce"`、`chunkSteps=ALLREDUCE_CHUNKSTEPS`、`sliceSteps=ALLREDUCE_SLICESTEPS`、sendbuff/recvbuff/count/datatype/op/comm/stream），调 `ncclEnqueueCheck(&info)`，返回。**所有 collective 都是这一套**（AllGather/Reduce/Broadcast/SendRecv 在 `collectives.cc:86-260`）。

```text
ncclEnqueueCheck(info)              src/enqueue.cc:3016
  ├─ CommCheck(info->comm)
  ├─ revoked 检查
  ├─ profiler group depth 记账
  ├─ ncclGroupStartInternal()       隐式开 group（即使你没显式调 GroupStart）
  ├─ ncclCommEnsureReady(comm)      enqueue.cc:3033
  ├─ ArgsCheck(info)                enqueue.cc:3041
  ├─ taskAppend(comm, info)         enqueue.cc:3048  ← 真正入队
  ├─ ncclGroupEndInternal()         隐式关 group → 触发 launch
  └─ (非阻塞 comm) 读 async error   enqueue.cc:3056
```

注意：**隐式 group 意味着单独一次 AllReduce 也会走完整的 group launch 路径**，只是这个 group 里只有一个任务。

### taskAppend → collTaskAppend

```text
taskAppend(comm, info)              src/enqueue.cc:2920
```

`taskAppend` 是个分发器：

- Send/Recv、Put/Signal/Wait → `p2pTaskAppend` / `rmaTaskAppend`（enqueue.cc:2495/...）
- 空 collective 直接丢弃；单 rank → `ncclLaunchOneRank`
- AlltoAll/Gather/Scatter 被分解成 p2p send/recv 对
- Blackwell 上大 AllGather（>8MiB、`NCCL_SYM_CE_THRESHOLD`、`ncclCeAvailable`）走 CE
- 其余 collective → `collTaskAppend`

AllReduce 落到 `collTaskAppend`：

```text
collTaskAppend(comm, info)          src/enqueue.cc:2580
  ├─ 从 comm->memPool_ncclTaskColl 分配 ncclTaskColl
  ├─ 填充: func/sendbuff/recvbuff/count/root/datatype/opHost/opDev
  │        chunkSteps/sliceSteps
  ├─ trafficBytes = count*elementSize*ncclFuncTrafficPerByte(func, nRanks)
  ├─ ncclTaskCollSorterInsert(planner.collSorter, task)   enqueue.cc:2641
  └─ ncclGroupCommJoin(comm, ncclGroupTaskTypeCollective)  把 comm 挂进 group
```

关键点：**这一步只入队，不调度**。`ncclTaskColl`（`include/comm.h:190`）被插进 `planner.collSorter`——一个按 `trafficBytes` 分桶的大顶结构（`comm.h:359`），目的是让大任务优先被分配到 channel，便于相邻任务聚合。同时 `ncclGroupCommJoin`（`include/group.h:107`）把 comm 加进 thread-local 的 `ncclGroupCommHead[ncclGroupTaskTypeCollective]` 链表，并 push 一个 `memScoped` 内存栈帧（这一批任务/plan 的分配都从这个栈帧出，group 结束时整帧回收）。

## ② Group 提交

```text
ncclGroupEndInternal()              src/group.cc:753
  (depth 归零才执行；ncclGroupError != success 则提前返回)
  ├─ 构建 ncclGroupJob，拷贝 ncclGroupCommHead[] 和 async job 列表
  ├─ 阻塞 group: groupLaunch(&groupJob->base)        group.cc:849
  └─ 非阻塞 group: std::thread(groupLaunchNonBlocking) group.cc:842
```

```text
groupLaunch → doLaunches            src/group.cc:307
  对每个 clique（同一 intraComm0 的 comms 一组）:
    ├─ ncclLaunchPrepare(comm)                      group.cc:321
    ├─ (group-launch 模式) intraBarrierIn
    ├─ 多轮，每轮:
    │    ncclLaunchKernelBefore_NoUncapturedCuda
    │    ncclLaunchKernel / ncclLaunchCeColl / ncclLaunchRma
    │    ncclLaunchKernelAfter_NoCuda
    └─ ncclLaunchFinish                             group.cc:369
```

clique 内多 comm 的 launch 是**按轮次交错**的（`group.cc:353-365`）：所有 comm 的第 N 个 plan 一起 launch，这样 clique 内各 rank 的同序号 kernel 在时间上重叠，对齐 collective 的进度。非 group-launch 模式则用 `unlaunchedPlansHead != nullptr` 判断是否还要轮（`group.cc:345`）。

## ③ 调度与建 Plan

这是 NCCL 的调度核心，几乎都在 `src/enqueue.cc`。

```text
ncclLaunchPrepare(comm)             src/enqueue.cc:1513
  ├─ ncclTasksRegAndEnqueue(comm)   排空 collSorter、选算法、注册缓冲
  ├─ scheduleCollTasksToPlan / scheduleP2pTasksToPlan
  │    / ncclScheduleBcastTasksToPlan / ncclSymmetricTaskScheduler
  ├─ finishPlan 收尾每个 ncclKernelPlan
  ├─ plan 推入 planner.planQueue    enqueue.cc:1597
  ├─ 设 unlaunchedPlansHead         enqueue.cc:1607
  └─ 流水线串接：user stream 等 deviceStream  enqueue.cc:1611-1664
```

### ncclTasksRegAndEnqueue

```text
ncclTasksRegAndEnqueue(comm)        src/enqueue.cc:299
  ├─ ncclTaskCollSorterDequeueAll  按 trafficBytes 降序排空 collSorter
  ├─ 按 (func, opDev.op, datatype) 分 bin            enqueue.cc:402
  ├─ 相邻 4× 大小内的任务聚合                        enqueue.cc:431
  ├─ 每个聚合调 ncclGetAlgoInfo                      enqueue.cc:437
  ├─ 算 devFuncId                                    enqueue.cc:438
  ├─ 分到 collBins[isCollnet][isNvls]                enqueue.cc:463
  ├─ 拼进 planner.collTaskQueue
  └─ 注册用户缓冲（NCCL_NVLS_REG_BUFFER 等）
```

"4× 聚合"是个有用的设计：同一 group 里多个大小相近的 collective，会被合成一个 kernel plan，减少 launch 数。这也是为什么把多个 AllReduce 包在一个 `ncclGroupStart/End` 里会更快。

### ncclGetAlgoInfo：选算法和协议

```text
ncclGetAlgoInfo(info)               src/enqueue.cc:2036
  ├─ initCollCostTable: collCostTable[ALGO][PROTO]
  ├─ updateCollCostTable: 按 NCCL_ALGO/NCCL_PROTO 过滤
  ├─ 有 tuner 插件: comm->tuner->getCollInfo(...)    enqueue.cc:2058
  │   (插件可覆盖 algorithm/protocol/nMaxChannels)
  └─ 否则 topoGetAlgoInfo                            enqueue.cc:2066
       (基于 comm->bandwidths[func][algo][proto] 的代价模型，
        公式 time = lat*latCount + nBytes/(1000*bw)，见 src/graph/tuning.cc:593)
  → 产出 info->algorithm / info->protocol
         info->nMaxChannels / info->nWarps
```

算法枚举（`include/plugin/nccl_tuner.h:26`）：`TREE(0) / RING(1) / COLLNET_DIRECT(2) / COLLNET_CHAIN(3) / NVLS(4) / NVLS_TREE(5) / PAT(6)`。协议：`LL(0) / LL128(1) / SIMPLE(2)`。

channel 数选择也在 `topoGetAlgoInfo`（enqueue.cc:1973+）：Ring/Tree 从 `nChannels` 起，若 `nBytes < nc*nt*threadThreshold` 就递减 `nc`（小消息少用 channel），CollNet Direct 有专门的减半逻辑，NVLS 取 `min(nvlsChannels, nChannels)`。

### scheduleCollTasksToPlan：分布到 channel

```text
scheduleCollTasksToPlan             src/enqueue.cc:572
  对每个待调度任务、每个可用 channel:
    ├─ 分配 ncclDevWorkColl（连续字节分布 cbd: countLo/Mid/Hi + chunkGrains）
    ├─ calcCollChunking: 算 chunkSize、pattern、proxyOp 模板   enqueue.cc:631
    ├─ ncclAddWorkBatchToPlan: 建 ncclDevWorkBatch 链          enqueue.cc:646
    ├─ ncclAddProxyOpIfNeeded: proxy op 入 plan.channels[c].proxyOpQueue
    └─ plan->channelMask 置位
  plan->kernelFn = ncclDevKernelForFunc[task->devFuncId]        enqueue.cc:787
```

`pattern`（`include/proxy.h:36`）决定 proxy 怎么找 send/recv 邻居：AllReduce+Ring → `ncclPatternRingTwice`；AllReduce+Tree → `ncclPatternTreeUpDown`；还有 `Nvls`/`NvlsTree`/`CollnetDirect`/`CollnetChain`（`enqueue.cc:2123`）。

`ncclDevWorkColl`（`include/device.h:262`）的关键字段：`channelLo/channelHi`、`nWarps`、`recvbuff/sendbuff`、`sendbuffOffset/recvbuffOffset`、`cbd{countLo,countMid,countHi,chunkGrainsLo/Mid/Hi}`、`redOpArg`、一堆位标志（`regUsed/netRegUsed/direct/...`）。这就是 device kernel 每个 channel 拿到的"做什么活"的描述。

### work 的存储位置

`uploadWork` 会根据 plan 大小选三种存储（`enqueue.cc:1223`）：

- `ncclDevWorkStorageTypeArgs`：塞进 kernel 的 4KB args buffer（最快，不经显存）
- `ncclDevWorkStorageTypeFifo`：写进 per-comm 的环形 work fifo（`comm->workFifoBufDev`，`enqueue.cc:1230`），推进 `workFifoProduced`
- `ncclDevWorkStorageTypePersistent`：CUDA-graph 捕获用的 `cudaMallocAsync` 缓冲（`enqueue.cc:1237`）

`finishPlan` 会在能塞下时把 Fifo/Persistent 降级为 Args（`enqueue.cc:210`）。

## ④ Launch

```text
ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan)   src/enqueue.cc:1670
  └─ uploadWork(comm, plan)            enqueue.cc:1674
       把 work 结构写进 device 可见的 fifo/args/persistent，推进 workFifoProduced
```

```text
ncclLaunchKernel(comm, plan)          src/enqueue.cc:1683
  ├─ grid  = { countOneBits(plan->channelMask), 1, 1 }
  ├─ block = { plan->threadPerBlock, 1, 1 }
  ├─ smem  = ncclShmemDynamicSize 或 plan->kernelDynSmem
  ├─ kernel = plan->kernelFn  (某个 ncclDevKernel_<coll>_<ty>_<redop>_<algo>_<proto>)
  ├─ extra[] = { ..., CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs, ... }
  └─ sm90+: 额外设 CGA cluster size (config.cgaClusterSize)、
            CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN、launch-completion-event、
            NVLINK-util scheduling attrs
       → cuLaunchKernelEx；否则 cuLaunchKernel
```

grid.x 就是 channel 数——**一个 block 服务一个 channel**。block 内线程数由 `topoGetAlgoInfo` 决定（Ring/Tree 用不同 warp 数）。

```text
ncclLaunchKernelAfter_NoCuda(comm, plan)   src/enqueue.cc:1784
  ├─ 不需要 host cb: hostStreamPlanTask(comm, plan)   enqueue.cc:1788
  └─ 否则作为 host callback 入队
```

```text
hostStreamPlanTask(comm, plan)        src/enqueue.cc:1392
  ├─ uploadProxyOps(comm, plan)       enqueue.cc:1396
  │    把 plan-local opCount 翻译成全局 comm opCount，
  │    每个 op 调 ncclProxySaveOp       enqueue.cc:1378
  └─ ncclProxyStart(comm)             enqueue.cc:1397
       把 proxy op 投递到共享 ncclProxyOpsPool，唤醒 progress 线程
  (非 persistent plan: 排个 reclaim callback 到 comm.callbackQueue)
```

到这一步，GPU kernel 已经被 launch、CPU proxy 也拿到了待办 op，两边开始并行推进。

## ⑤ 设备执行

```text
ncclDevKernel_<suffix>(args4K)       src/device/common.cu:23 + common.h:420
  (DEFINE_ncclDevKernel 宏生成，例如 ncclDevKernel_AllReduce_Sum_*_Ring_LL128)
  → ncclKernelMain<specializedFnId, RunWorkBatch<coll,ty,redop,algo,proto>>(&args4K.args)
```

```text
ncclKernelMain(args)                 src/device/common.h:346
  ├─ blockIdx.x ↔ channel
  │    n = popcll(channelMask & ((1<<tid)-1));  blockIdx.x==n → channelId=tid
  ├─ warp0: ncclKernelComm → shmem
  ├─ warp1: ncclDevChannel → shmem
  ├─ 其余 warp: loadWorkBatchToShmem(blockIdx.x)   common.h:135
  │    (按 workStorageType 决定 ld.param 还是 ld.global 读 ncclDevWorkBatch)
  └─ 循环直到 nextBatchIx == -1:
       profiler(START)
       SpecializedRunWorkBatch().run()   (或 ncclDevFuncTable[funcId]())
       profiler(STOP)
       loadWorkBatchToShmem(nextBatchIx)
```

```text
RunWorkBatch<Fn,T,RedOp,Algo,Proto>::run   src/device/common.h:282
  对 shmem 里每个 work slot:
    RunWorkColl<...>().run(tid, subtn, work)   common.h:308
      subtn = work->nWarps * WARP_SIZE
      (相邻 work 的 nWarps 变化时插 __syncthreads)
```

### Ring AllReduce 的设备实现

AllReduce 的各特化在 `src/device/all_reduce.h`：

| Algo | Proto | 入口 |
|---|---|---|
| Ring | Simple | `all_reduce.h:234` → `runRing<T,RedOp,ProtoSimple>` |
| Tree | Simple | `all_reduce.h:242` → `runTreeSplit` |
| CollnetDirect | Simple | `all_reduce.h:253` |
| NVLS | Simple | `all_reduce.h:389` |
| NVLS_TREE | Simple | `all_reduce.h:522` |
| CollnetChain | Simple | `all_reduce.h:630` |
| Ring | LL | `all_reduce.h:756` |
| Ring | LL128 | `all_reduce.h:770` |

经典 k-step ring（`all_reduce.h:14` 的 `runRing`），构造 `Primitives<T,RedOp,FanSymmetric<1>,Direct=1,Proto,0>(tid,nthreads,&ring->prev,&ring->next,...)`，然后：

```text
directSend
→ directRecvReduceDirectSend ×(k-2)
→ directRecvReduceCopyDirectSend(postOp)
→ directRecvCopyDirectSend ×(k-2)
→ directRecv
```

Tree 变体用 `FanAsymmetric<NCCL_MAX_TREE_ARITY=3,1>`（`all_reduce.h:97`）。协议层（`src/device/primitives.h:26/46/61`）：`ProtoSimple` 用大 buffer + head/tail 计数；`ProtoLL` 把 buffer 一半当 flag，64-bit line `{data,flag}`；`ProtoLL128` 用 128B line（15/16 个 uint64 是数据）。

每个 channel 处理的数据切片由 `ncclCollCbdPart`（`include/device.h:333`）按 `work->cbd` 和 `channelId` 切出。详见 [kernel](kernel)。

## 回收

非 persistent plan 跑完后，`reclaimPlan`（`enqueue.cc:1418`，作为 `ncclCommCallback` 挂在 `comm->callbackQueue`）回收 plan 的内存和 fifo 槽位。整个 group 的 `memScoped` 栈帧在 `groupEnd` 时整帧释放。

## 小结

| 阶段 | 关键函数 | 文件:行 | 产物 |
|---|---|---|---|
| 入队 | `collTaskAppend` | enqueue.cc:2580 | `ncclTaskColl` 入 `collSorter` |
| Group | `doLaunches` | group.cc:307 | 按 clique 分轮 launch |
| 调度 | `ncclTasksRegAndEnqueue` | enqueue.cc:299 | 选 algo/proto，聚合任务 |
| 调度 | `scheduleCollTasksToPlan` | enqueue.cc:572 | `ncclKernelPlan` + `ncclProxyOp` |
| Launch | `ncclLaunchKernel` | enqueue.cc:1683 | `cuLaunchKernelEx` |
| Proxy | `hostStreamPlanTask` | enqueue.cc:1392 | op 投递到 `OpsPool` |
| 设备 | `ncclKernelMain` | device/common.h:346 | ring/tree 数据搬运+归约 |

记住一条便条即可：

```text
AllReduce → ncclEnqueueCheck → taskAppend → collTaskAppend（入队）
  → (group end) ncclLaunchPrepare → scheduleCollTasksToPlan（建 plan）
  → ncclLaunchKernel（起 GPU）+ uploadProxyOps（喂 proxy）
  → ncclKernelMain → RunWorkBatch（GPU 跑 ring/tree）
```

接下来读 [Communicator 与 Group](communicator) 看 `ncclComm` 的内部结构和 group 的并发模型，或读 [Device Kernel](kernel) 看 GPU 侧的执行细节。
