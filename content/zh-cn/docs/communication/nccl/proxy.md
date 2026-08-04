---
title: Proxy 线程
description: CPU proxy 线程的职责、状态机与数据推进。
weight: 6
---

CPU proxy 线程是 NCCL 跨节点通信的"桥梁"：GPU kernel 不能直接操作网卡，网络段的数据收发必须由 CPU 线程经 RDMA/socket 完成。本文基于 `v2.30.4-1` 的 `src/proxy.cc` 拆解 proxy 的线程模型、状态机、数据推进循环，以及它和 GPU kernel 的协同。

## 为什么需要 proxy

GPU kernel 能做归约、能读写显存、能经 NVLink/PCIe P2P 搬数据，但**不能直接调 IB verbs 或 socket API**。跨节点的数据必须有个 CPU 线程来 `isend`/`irecv`/`test`。NCCL 的做法：

```text
GPU kernel         CPU proxy 线程
   │                    │
   │ 写数据+flag 进      │ 轮询 head/tail
   │ ring buffer         │
   │ (conn->buffs[proto])│
   │◀───────────────────││ 从 ring buffer 取数据 → 网卡 isend
   │ head/tail 同步       │ 从网卡 irecv → 写进 ring buffer
   │                     │
   │ 读 ring buffer       │
   ▼                     ▼
```

GPU 和 proxy 通过 `ncclConnInfo.buffs[protocol]` 这段 ring buffer + `head`/`tail` 计数器同步：GPU 写数据推进 `head`，proxy 看到 `head` 前进就 `isend`；proxy `irecv` 收到数据写进 buffer 推进 `tail`，GPU 看到 `tail` 前进就读。`connFifo`（每步 `{mode,offset,size,ptr}`）则是更细粒度的 GPU↔proxy 命令通道。

对 P2P/SHM 这类同机 transport，数据搬运由 GPU kernel 直接做，**不需要 proxy**。proxy 只在 Net/CollNet transport 上激活。

## 线程模型

每个 `ncclProxyState`（`include/proxy.h:346`）关联一个 GPU 设备，起三类线程：

```text
ncclProxyCreate                src/proxy.cc:1940
  ├─ std::thread(ncclProxyService, ...)     proxy.cc:1640   服务线程
  └─ std::thread(ncclProxyServiceUDS, ...)  proxy.cc:1972   UDS 线程（cuMem fd 传递）

ncclProxyStart                 src/proxy.cc:1024   (首次有 op 时懒起)
  └─ std::thread(ncclProxyProgress, ...)    proxy.cc:943   推进线程
```

- **Service 线程**：处理建连 RPC（`Init/Setup/Connect/Start/Close/Abort/Register/...`）、`poll()` 监听 socket、accept 连接。设 CPU 亲和（`proxyCpuset`）、持有 CUDA context（GDR 注册需要）。每个 local peer 一个 socket + 异步 op 队列。
- **Progress 线程**：数据搬运主循环，懒启动——首次有 proxy op 需要推进时才起，省资源。
- **UDS 线程**：专门处理 cuMem 句柄的 fd 传递（MNNVL/CUmem 场景跨进程注册需要），`ncclProxyServiceUDS`。

`proxyState` 持有：`comm`、`tpRank/tpnRanks/tpLocalnRanks`、`cudaDev`、channel 计数、`ncclNet`/`ncclCollNet`/`ginState` 插件、`abortFlag`、service+progress 线程句柄、`listenSock`、`peerSocks`、`proxyOps`（主线程侧）、`progressState`（progress 线程侧，含 `opsPool`/`active`/`pool`）。

## Op 的投递与取出

主线程（用户 API 线程）在 `hostStreamPlanTask` 里把每个 channel 的网络段工作打包成 `ncclProxyOp`，经 `ncclProxySaveOp` 投递：

```text
ncclProxySaveOp                src/proxy.cc:589
  └─ SaveProxy                 proxy.cc:567
       找到 connector → ncclLocalOpAppend   proxy.cc:483
         ├─ proxyOps = comm->proxyState->proxyOps + proxyConn->tpLocalRank
         ├─ 从 pool 取一个空闲 op slot (freeOp 链表)
         ├─ memcpy(op, proxyOp, sizeof(ncclProxyOp))
         ├─ 接到 nextOps 链表尾部
         └─ op->opCount = pool->ops[nextOpsEnd].opCount + 1
ncclProxyPost                  proxy.cc:471
  写进 ncclProxyOpsPool（共享内存 ring），signal condvar 唤醒 progress 线程
```

`ncclProxyOpsPool`（`include/proxy.h:222`）是 per-local-peer 的共享 ring，主线程写、progress 线程读，无锁交接。`MAX_OPS_PER_PEER` 限制每 peer 待办数，满了会推进后再收。

## Progress 循环

```text
ncclProxyProgress              src/proxy.cc:943
  循环:
    ├─ progressOps(state->active)            proxy.cc:764
    │    对每个 active ncclProxyArgs 调其 progress 回调
    │    (progress 回调 = transportComm->proxyProgress，按 pattern 推进)
    ├─ ncclProxyGetPostedOps                 proxy.cc:790
    │    从 OpsPool 拉新 op
    ├─ ProxyAppend                          proxy.cc:434/838
    │    把新 op 转 ncclProxyArgs，接 active 链表
    │    ncclProxyOpToArgs                   proxy.cc:362
    └─ 无事可做则 yield
```

`ncclProxyArgs`（`proxy.h:178`）是推进的执行单元，含 `subs[NCCL_PROXY_MAX_SUBS]`（`=MAXCHANNELS=64`）：

```text
ncclProxySubArgs   proxy.h:135
  connection / offset / loopSize / channelId / nsteps / nbytes / chunkSize / peer
  posted / received / flushed / transmitted / done / end   每 step 的进度计数
  requests[NCCL_STEPS] / pHandles[NCCL_STEPS]              每步的 IB 请求/内存句柄
ncclProxyArgs      proxy.h:178
  progress (func ptr)   推进函数
  nsubs / done / opCount / sliceSteps / chunkSteps / chunkSize
  totalSendSize / totalRecvSize / sendSizePerRound / recvSizePerRound
  dtype / redOp / pattern / coll / collAPI / protocol / algorithm / state
  sharedBuff[NCCL_STEPS] / sharedSize[NCCL_STEPS]
  next / nextPeer / proxyAppendPtr
```

`progress` 函数指针由 transport 设置，对 NET 插件就是调 `ncclNet->isend`/`irecv`/`iflush`/`test`，把 `subs[]` 里每个 channel 的 `nsteps` 步数据推进，更新 `posted/received/transmitted/done`。pattern（`Ring/RingTwice/TreeUpDown/CollnetChain/...`）决定 send/recv 邻居和方向。

## 连接状态机

proxy 连接的生命周期（`proxy.h:379`）：

```text
connUninitialized(0)
  → connInitialized(1)        ncclProxyMsgInit     建立 connection 对象
  → connSharedInitialized(2)  ncclProxyMsgSharedInit  共享资源
  → connSetupDone(3)          ncclProxyMsgSetup    transport-specific setup
  → connConnected(4)          ncclProxyMsgConnect  可传数据
```

由 service 线程收到对应 RPC 推进。`ncclProxyConnection`（`proxy.h:393`）持有：`send`/`transport`/`shared`、`tcomm`（transport 侧 comm）、`transportResources`、`mhandles[NCCL_NUM_PROTOCOLS]`、`state`、`collNet`、`needsProxyProgress`、`proxyMemHandleQueue`、`netDeviceHandle`。

异步 RPC 通过 `ncclProxyCallAsync`（`proxy.cc:1294`）/`ncclProxyCallBlocking`（`:1379`）发请求，`ncclPollProxyResponse`（`:1320`）轮询结果（存在 `expectedResponses`）。

## 与 GPU kernel 的协同

以 Ring AllReduce 的网络段为例：

1. **GPU 写数据**：kernel 把本轮要发的数据写进 `conn->buffs[SIMPLE]`（远端 recv buffer 的本地镜像 + 本地 send buffer），推进 `head`。
2. **proxy 发送**：progress 线程看到 `head` 增加，对每 step 调 `ncclNet->isend(buff+offset, size, mhandle)`，记录 `requests[step]`，更新 `posted`。
3. **proxy 接收**：对收方向，`irecv` 收到的数据写进本地 recv buffer 的对应位置，更新 `received`。
4. **GPU 读数据**：kernel 看到 `tail` 增加（proxy 收完），读 recv buffer 做归约。
5. **flush**：某些协议（Simple）需要 `iflush` 保证远端可见性；`ncclTopoNeedFlush`（`paths.cc:542`）判断是否需要。

这种 ring buffer + 双计数器的设计让 GPU 和 proxy 能**流水线重叠**：GPU 写 step N+1 的同时 proxy 在发 step N。`NCCL_STEPS`（通常 8）控制流水线深度。

`ncclProxyOp`（`proxy.h:69`）携带 proxy 推进所需的全部参数：`connection`、`nbytes`/`opCount`/`root`、`nsteps`/`chunkSize`/`sliceSize`/`loopSize`/`loopOffset`/`channelSize`、`sliceSteps`/`chunkSteps`/`channelId`、`dtype`/`redOp`/`coll`/`collAPI`/`pattern`/`protocol`/`algorithm`、`reg`、`sendMhandle`/`recvMhandle`、`sendbuff`/`recvbuff`、`isOneRPN`、`ringAlgo`（net offload 的 ring 算法对象）、`nChannels`/`nPeers`。

## GDR 与 fd 传递

GPUDirect RDMA 让网卡直接 DMA GPU 显存，省 CPU 拷贝。proxy 在此承担**内存注册**：首次用到某 GPU buffer 时，经 `ncclProxyMsgRegister` 调 `ncclNet->regMr`/`regMrDmaBuf`，把 mhandle 存进 `connection->mhandles[proto]`。两种模式：`nv_peermem`（`NCCL_PTR_CUDA`）和 DMA-BUF（`NCCL_PTR_DMABUF`，免 peermem，新内核）。

对 MNNVL/CUmem 跨进程注册，需要传递 cuMem 句柄的 fd，走 UDS socket：`ncclProxyClientGetFdBlocking`（`proxy.cc:1249`）/`ncclProxyClientBatchQueryFdBlocking`（`:1268`）在 UDS 线程上完成，`ncclProxyServiceUDS` 处理。

## 终止与容错

`comm->abortFlag`/`abortFlagDev` 触发终止。proxy 通过 `ncclProxyMsgAbort` 取消在途操作，progress 线程检查 `state->abortFlag` 退出。`proxyState->refCount` 管理生命周期，split comms 共享同一 `proxyState`，最后一个 comm 销毁时才真正 join 线程。`directMode` 用于某些 GIN/LSA 场景绕过 proxy 直接 GPU 推进。

## 小结

- proxy 存在是因为 GPU 不能直接操作网卡；网络段由 CPU 线程 `isend`/`irecv`。
- 三类线程：service（建连 RPC）、progress（数据搬运，懒启动）、UDS（cuMem fd 传递）。
- 主线程把 `ncclProxyOp` 经 `ncclProxyOpsPool`（共享 ring）投递给 progress 线程，无锁交接。
- progress 调 transport 的 `proxyProgress` 推进 `ncclProxyArgs.subs[]`，靠 ring buffer 的 `head`/`tail` 与 GPU kernel 流水线协同。
- GDR 把内存注册下放到 proxy（`regMr`/`regMrDmaBuf`），让网卡直接 DMA 显存。

进一步：GPU 侧如何消费这些 buffer 见 [Device Kernel](kernel)；GDR/GIN 等底层能力见 [高级特性](features)。
