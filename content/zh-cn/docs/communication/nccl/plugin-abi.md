---
title: 插件 ABI
description: NCCL 如何接入第三方网络、tuner、profiler 插件。
weight: 8
---

NCCL 的网络栈、调优、打点都做成插件 ABI，让外部库（厂商 RDMA/SHARP 实现、自定义调优、profiler）能挂进来而不改 NCCL 本体。本文基于 `v2.30.4-1` 的 `src/plugin/` 和 `src/include/plugin/`，拆解插件加载、各接口、版本兼容。

## 插件加载

`src/plugin/plugin_open.cc` 负责 dlopen。5 种插件类型（`enum ncclPluginType`）：

| 类型 | 库前缀 | 环境变量 | 符号 |
|---|---|---|---|
| NET | `libnccl-net` | `NCCL_NET_PLUGIN` | `ncclNetPlugin_vN` |
| GIN | `libnccl-gin` | — | — |
| TUNER | `libnccl-tuner` | `NCCL_TUNER_PLUGIN` | `ncclTunerPlugin_v6` |
| PROFILER | `libnccl-profiler` | — | `ncclProfilerPlugin_v6` |
| ENV | `libnccl-env` | — | — |

加载流程（`plugin_open.cc`）：`getLibPath` 用 `dlinfo(RTLD_DI_LINKMAP)` 定位 libnccl 自身路径 → 拼 `.so` 全路径 → `ncclOsDlopen` 打开 → dlsym 取版本符号。特殊名 `"STATIC_PLUGIN"` 表示用静态链接的插件符号（不 dlopen）。

## Net 插件

最有分量的插件。`src/plugin/net/net.cc:28-44` 维护版本数组 `getNcclNet[7] = {v12,v11,...,v6}`，**从新到旧试**，加载首个匹配的 `ncclNetPlugin_vN` 符号。`netPluginLib_t`（`net.cc:60`）持有 `dlHandle`、`ncclNet`、`ncclNetVer`、`ncclCollNet`、状态机（`LoadFailed/LoadReady/InitReady/Enabled/Disabled`）、引用计数、物理/虚拟设备列表。支持多插件（`NCCL_NET_MAX_PLUGINS`，`netPluginLibs[]`）。

### ncclNet_t (v12)

`src/include/plugin/net/net_v12.h:75`，vtable：

```text
init(ctx, commId, config, logFunction, profFunction)
devices / getProperties
listen / connect(handles, ...)      connect 现在可返回 sendDevComm（device offload）
accept(...)                          可返回 recvDevComm
regMr / regMrDmaBuf / deregMr        内存注册（GDR/DMA-BUF）
isend / irecv / iflush / test
closeSend / closeRecv / closeListen
getDeviceMr                          把 mhandle 拷成 device 可见 ptr
irecvConsumed                        通知 device 已完成的 recv
makeVDevice                          虚拟网卡
finalize / setNetAttr
```

`ncclNetAttr_v12_t`（`net_v12.h:67`）暴露 `sendCommAttrs/recvCommAttrs`（`maxConcurrentPeers/maxFlowsPerPeer/...`）、`op/algo/proto`，通过 `setNetAttr` 让 NCCL 告知插件当前的 op/algo/proto，插件据此调优（比如调verbs 的 QP 数）。

### ncclCollNet_t (v12)

`net_v12.h:156`，SHARP/交换机归约接口：

```text
init / devices / getProperties
listen / connect(handles[], nranks, rank)
reduceSupport(dataType, redOp)
regMr / regMrDmaBuf / deregMr
iallreduce / iallgather(ncclNetSGE_v12_t 分段列表, window offset/bytes)
ireducescatter / iflush / test / closeColl / closeListen
makeVDevice / finalize
```

v6 只有 `iallreduce`，后续版本加了 `iallgather`/`ireducescatter`。`ncclNetSGE_v12_t` 支持 scatter-gather，让一次集合操作处理多段 buffer。

### 版本兼容

每个版本一个递减的 vtable：v6 最简（`iallreduce` only），v12 最全（含 `irecvConsumed`/`makeVDevice`/`setNetAttr`/device offload）。NCCL 用 `ncclNet_vN_t` 结构体适配，老插件自动兼容。这是为什么外部插件（如 NCCL RDMA SHARP plugin）能跨多个 NCCL 版本工作。

### 内置插件

`ncclNetIb`（`src/transport/net_ib/`）和 `ncclNetSocket`（`net_socket.cc`）是内置的，实现同一 v12 vtable，作为 `NCCL_NET_NUM_INTERNAL_PLUGINS` 注册在 `net.cc`。**外部插件优先**——只要 `NCCL_NET_PLUGIN` 或默认路径找到 `.so`，就覆盖内置 IB/socket。

## Tuner 插件

`src/plugin/tuner/tuner.cc` + `src/include/plugin/tuner/tuner_v5.h`。符号 `ncclTunerPlugin_v6`（`nccl_tuner.h:23`）。`ncclTuner_v6_t` 提供：

```text
init / finalize
getCollInfo(ctx, func, nBytes, numPipeOps, costTable, nAlgos, nProtos, regBuff, *nMaxChannels)
```

钩在 `ncclGetAlgoInfo`（`enqueue.cc:2058`），**在内置 topo 代价模型之前**调用。插件可以填 `costTable` 或直接设 `algorithm`/`protocol`/`nMaxChannels`，覆盖内置选路。这是做拓扑特化调优（针对某集群的实测带宽/延迟表）的官方入口。

内置代价模型见 [topology-graph](topology-graph) 的 `tuning.cc` 一节，常量来自 `ncclTunerConstants_v5_t`（按 Volta/Ampere/Hopper/Blackwell 分档）。

## Profiler 插件

`src/plugin/profiler/profiler.cc` + `src/include/profiler.h`。符号 `ncclProfilerPlugin_v6`。通过 `ncclProfilerStart*Event`/`ncclProfilerRecord*EventState` 散布在 NCCL 各处（host API、launch、kernel `profiler(START/STOP/FINI)`、proxy op）。`comm->ncclProfiler`（`comm.h:~795`）持有状态。还有个内置 `profilerTransport`（`src/transport/profiler.cc:52`）作为 transport 注册表的一项，用于打点。

Profiler plugin 是 NCCL Inspector、NCCL Profiler Plugin 这类工具的集成点——它们通过这个 ABI 采集集合通信的时间线和统计，而不需改 NCCL 源码。

## GIN 插件

`src/plugin/gin/gin.cc` + `src/gin/gin_host.cc`。经 `comm->sharedRes->ginState.ncclGin` 挂载（`gin_host.cc:49`）。`getGlobalGinType`/`getGlobalRailedGinType`（`:21/35`）决定 GIN 连接类型。GIN 插件 `getProperties` 返回 `netDeviceType == NCCL_NET_DEVICE_GIN_PROXY`。详见 [features](features) 的 GIN 一节。

## ENV 插件

`src/plugin/env/env.cc`，允许通过插件覆盖 NCCL 环境变量。用于不便设 env 的部署场景（如容器化里集中配置）。

## 拓扑集成

网络插件不只是一组函数，还要进拓扑图。`ncclTopoProcessNet`（`src/graph/topo.h:175`）把插件设备处理进 XML 拓扑。`ncclTopoNetInfo`（`topo.h:173`）携带 `coll`/`gin`/`net`/`netPluginIndex`/`maxDevsPerNic`/`dmaBufSupport`/`mergeLevel`/`forceMerge`（NIC 融合）和函数指针 `getDevCount/setVirtDevCount/getProperties/makeVDevice/devices`。这让图搜索能感知插件能力（如是否支持 SHARP、是否支持 DMA-BUF），选路时纳入考量。

## 调试

- `NCCL_DEBUG=INFO` 会打印加载了哪个插件、版本号。
- `NCCL_NET_PLUGIN` 指定外部网络插件路径。
- 插件加载失败会回退到内置 IB/socket，日志里能看到 `LoadFailed` → fallback。
- 多插件时 `netPluginLibs[]` 顺序决定优先级。

## 小结

- 5 类插件（NET/GIN/TUNER/PROFILER/ENV）经 dlopen + 版本符号加载，外部优先于内置。
- Net 插件 v6→v12 vtable 递增，老插件自动兼容；`ncclNet_t` 管 send/recv，`ncclCollNet_t` 管 SHARP 归约。
- Tuner 插件在内置代价模型之前介入，可覆盖 algo/proto/channel 数选择——拓扑特化调优的入口。
- Profiler 插件经散布的 hook 采集时间线，是外部观测工具的集成点。
- 插件能力通过 `ncclTopoNetInfo` 注入拓扑图，影响图搜索选路。

进一步：基于这些 ABI 的高级能力（GDR/GIN/NVLS/CE/RMA）见 [高级特性](features)。
