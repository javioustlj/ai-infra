---
title: 高级特性
description: GDR / GIN / NVLS / RMA / CE / symmetric memory 的能力与实现。
weight: 9
---

NCCL 在基础 ring/tree 之上，针对新一代硬件有一组高级特性，目标是减少 CPU 参与、利用硬件归约、降低延迟。本文基于 `v2.30.4-1` 源码梳理六类特性的能力边界和实现位置，供排障和调优时定位。

## GDR：GPUDirect RDMA

**能力**：网卡直接 DMA GPU 显存，省掉 CPU 中转拷贝。跨节点带宽和延迟双双改善。

**实现**：`src/transport/net_ib/gdr.cc`。两条路径：

| 模式 | 标志 | 依赖 | 设置点 |
|---|---|---|---|
| nv_peermem | `NCCL_PTR_CUDA` | `nv_peermem.ko` 内核模块 | `net_ib/init.cc:470` |
| DMA-BUF | `NCCL_PTR_DMABUF` | 较新内核 + `ibDev->dmaBufSupported` | `init.cc:474` |

**判断**：`ncclTopoCheckGdr`（`src/graph/paths.cc:441`）查 NIC+GPU 的 GDR 支持；`ncclTopoIsGdrAvail`（`:518`）。proxy 首次用到某 GPU buffer 时经 `ncclProxyMsgRegister` 调 `ncclNet->regMr`/`regMrDmaBuf` 把 mhandle 存进 `connection->mhandles[proto]`。`ncclTopoNeedFlush`（`paths.cc:542`）判断是否需要 `iflush` 保证远端可见性。

**排障**：`NCCL_DEBUG=INFO` 会打印是否启用 GDR、用了 peermem 还是 dma-buf。带宽远低于网卡额定时常是 GDR 没开（退回 CPU bounce）。

## GIN：GPU-Initiated Network

**能力**：GPU 直接发起网络操作，**绕开 CPU proxy**。对延迟敏感的小消息和需要 GPU 直接控制网络流水的场景有用。

**实现**：`src/gin/gin_host.cc` + `src/transport/net_ib/gin.cc` + `src/nccl_device/gin_*.cc` + `src/include/gin/`。`comm->globalGinSupport`（`gin_host.cc:26`）分 `NCCL_GIN_CONNECTION_NONE/FULL`。需 Volta+（compCap≥70，`:61`）。`ncclGinIbGdrSupport`（`net_ib/gin.cc:17`）查 IB GDR。`ncclProxyConnector.proxyGinProgress`（`device.h:156`）是 GIN 的推进函数（替代普通 proxyProgress）。GIN 插件（`src/plugin/gin`）提供 `ncclGin_t` 接口，`getProperties` 返回 `netDeviceType == NCCL_NET_DEVICE_GIN_PROXY`。per-rank 的 GPU 侧窗口 `ncclGinWindow` 见 `src/include/devcomm_v22902.h`。

**与 proxy 的关系**：GIN 启用时，网络段不再走 CPU proxy 线程的 `isend`/`irecv`，而是 GPU kernel 直接经 GIN 接口发起。`directMode`（`proxyState`）标记此路径。

## NVLS：NVLink SHARP

**能力**：用 NVSwitch 的硬件多播 + 归约，在节点内做集合通信时减少 GPU kernel 的工作量，降低延迟、提升吞吐。

**实现**：`src/transport/nvls.cc` + `src/nccl_device/lsa_barrier.cc`。用 CUDA `CUmulticast` 对象（`CUmulticastObjectProp` + `CUmemGenericAllocationHandle`，CUDA 12.1+）。`ncclNvlsBufferSetup`（`nvls.cc:376`）分配：

- `mcBuff`/`mcCredit`：多播地址
- `ucBuff`/`ucCredit`：单播地址

`ncclNvlsTreeConnect`（`:269`）在 NVLS 上接树。`NCCL_MAX_NVLS_ARITY=32`、`NCCL_MAX_NVLS_TREE_ARITY=3`。`ncclNvlsSharedRes`（`include/transport.h:83`）持有这些对象。由 `NCCL_NVLS_ENABLE` 和 `comm->nvlsSupport`/`nvlsRegSupport` 控制。

**注册缓冲**：`nvlsRegSupport` 允许 registered-buffer NVLS，经 `ncclNvlsRegResourcesQuery`（`enqueue.cc:2076`）查询。注册路径让 NVLS 直接用用户 buffer，省一次拷贝。

**排障**：`NCCL_DEBUG=INFO` 打印 NVLS 是否启用、几个 channel。节点内带宽异常时先确认 NVLS 是否开了（老 CUDA 版本或 NVSwitch 缺失会退回普通 ring）。

## RMA：one-sided 通信

**能力**：`ncclPutSignal`/`ncclSignal`/`ncclWaitSignal` 的一侧写入 + 信号语义，类似 NVSHMEM/GASNet，用于细粒度跨节点数据搬运。

**实现**：`src/rma/`（`rma.cc`/`rma_ce.cc`/`rma_proxy.cc`/`rma_proxy_launch.cc`/`rma_proxy_progress.cc`）+ `src/include/rma/`。API 在 `src/collectives.cc:224/238/251`。两条路径：

- proxy：`ncclRmaProxyPutLaunch`/`ncclRmaProxyWaitLaunch`
- CE：`ncclRmaCePutLaunch`/`ncclRmaCeWaitLaunch`（跑在 CE stream）

状态在 `comm->rmaState`（`comm.h:~797`），per-context 任务队列在 `planner.rmaTaskQueues`。需 CUDA 12.5+ driver 和 `comm->hostRmaSupport`。

## CE：Copy Engine collectives

**能力**：用 CUDA copy engine（CE）经 `cuStreamBatchMemOp` 做节点内 NVLink 集合通信，**不 launch GPU kernel**。延迟低、不占 SM。

**实现**：`src/ce_coll.cc` + `src/include/ce_coll.h`。`ncclCeInit`（`ce_coll.cc:24`）设 `comm->ceColl.baseUCSymReadyPtr/baseUCSymComplPtr`。`ncclCeAvailable`（`:105`）要求 CUDA 12.5+、全 NVLink 连通、symmetric 支持、注册窗口。`ncclCeLaunchBatchOps`（`:338`）发批量 memop。

**触发**：

- `NCCL_CTA_POLICY_ZERO`（`enqueue.cc:2957`）——配 0-CTA 策略时走 CE。
- Blackwell 上大 AllGather（>8 MiB、minCompCap≥100，`enqueue.cc:3000`）自动走 CE。

CE 与 symmetric memory 窗口配合工作。

## Symmetric memory

**能力**：各 rank 注册一段对称窗口，rank 间直接 load/store 对方窗口（不经 FIFO），用于 CE 和 `NCCL_CTA_POLICY_EFFICIENCY` 路径。节点内极低延迟。

**实现**：`src/register/` + `src/sym_kernels.cc` + `src/include/sym_kernels.h` + `src/device/symmetric/`。窗口注册在 group end 时经 `ncclDevrWindowRegisterInGroup`（`group.cc:~250`）。`ncclGetSymRegType`（`sym_kernels.cc:671`）分类窗口注册类型。`ncclSymkInitOnce`（`:430`）初始化 symmetric kernel，`ncclSymkPickKernel`（`:595`）/`ncclSymkMakeDevWork`（`:658`）建 device work。由 `comm->symmetricSupport` 和 `comm->isAllDirectNvlink` 控制。

## 其他

### DMA-BUF 注册

`regMrDmaBuf` 在 `ncclNet_v12_t`（`net_v12.h:99`）和 `ncclCollNet_v12_t`（`:177`）都有：`(data, size, type, offset, fd)`，接收 Linux dma-buf fd。用于 GPUDirect 免 `nv_peermem`，支持新内核。比 GDR peermem 路径更通用。

### MNNVL（多节点 NVLink）

`comm->MNNVL`（`comm.h:592`）、`comm->clique`/`cliqueRank`（`:594`）。`ncclTopoCheckMNNVL`（`paths.cc:412`）。用 cuMem IPC（`cuMemSupport`/`dmaBufSupport`）+ UDS fd 传递（`ncclProxyClientGetFdBlocking`，`proxy.cc:1249`）跨节点注册。P2P transport 的 `P2P_CUMEM` 走这条。`src/mnnvl.cc` + `isMultiRankGpu`（`comm.h:593`）处理一 GPU 多 rank。

### NVLink Domain

`comm->nvlDomainInfo`（`comm.h:598`，`ncclNvlDomainInfo_v5_t`）+ `comm->nvlDomainSize`（`:634`），用于限定 NVLS/symmetric 作用范围（`minRanksPerNvlDomain/maxRanksPerNvlDomain`，`clusterUuid`）。

### GDC memory

`NCCL_NET_MAP_GDCMEM`（`net.cc:29`）：Grace/Blackwell Coherent 内存映射，net transport 的一种内存类型。

## 哪个特性在哪起作用

| 场景 | 相关特性 |
|---|---|
| 跨节点 GPU 显存直收发 | GDR (peermem/dma-buf) |
| 跨节点 GPU 直发、省 proxy | GIN |
| 节点内 NVSwitch 归约/多播 | NVLS |
| 节点内免 kernel 集合通信 | CE + symmetric |
| 一侧写入 + 信号 | RMA |
| 跨节点 NVLink fabric | MNNVL |
| Blackwell 大 AllGather | CE (自动触发) |

## 小结

- GDR 让网卡直 DMA 显存；GIN 让 GPU 直发网络绕开 proxy；NVLS 用 NVSwitch 硬件归约；CE 用 copy engine 免 kernel；RMA 给一侧写入语义；symmetric 让节点内直接 load/store 对方窗口。
- 这些特性大多需要新 CUDA/驱动和特定拓扑（全 NVLink、NVSwitch、MNNVL fabric），老环境会自动回退到基础 ring/tree + CPU proxy。
- 排障路径：`NCCL_DEBUG=INFO` 打印每个特性是否启用；带宽/延迟异常时先确认预期特性是否真开了，再看是不是退回了 baseline 路径。

进一步：自己读这些特性的源码建议按 [源码阅读路线](reading-guide) 的第三遍来做。
