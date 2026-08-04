---
title: Transport 与建连
description: P2P/SHM/Net/NVLS/CollNet 选型与连接状态机。
weight: 5
---

Transport 层回答"数据怎么搬"。NCCL 把不同物理通道抽象成统一的 vtable，初始化时按拓扑选型、建连，运行时由 proxy 线程（网络）或 GPU kernel（NVLink/SHM）推进数据。本文基于 `v2.30.4-1` 的 `src/transport.cc` + `src/transport/`。

## Transport 注册表

```text
src/include/transport.h:24-34
NTRANSPORTS = 4  (外加 PROFILER=4 用于打点)
TRANSPORT_P2P = 0    同机 GPU 直连 (NVLink/PCIe/C2C)
TRANSPORT_SHM = 1    同机跨进程共享内存
TRANSPORT_NET = 2    跨节点 (IB/RoCE/socket，经网络插件)
TRANSPORT_COLLNET = 3  SHARP/交换机内归约

ncclTransports[] = { &p2pTransport, &shmTransport, &netTransport,
                     &collNetTransport, &profilerTransport }
```

每个 transport 是一个 vtable（`transport.h:128`）：

```text
struct ncclTransport {
  name[8]
  canConnect(...)              判断能否用于某对 peer
  send / recv : ncclTransportComm*
}

struct ncclTransportComm {     transport.h:116
  setup / connect / free
  proxySharedInit / proxySetup / proxyConnect / proxyFree / proxyProgress
  proxyRegister / proxyDeregister
  ...
}
```

`canConnect` 接 `ncclPeerInfo`（hostHash/pidHash/cudaDev/busId）和 `graph->typeIntra/typeInter`，决定一对 rank 该用哪个 transport。

## 五种 Transport

### P2P (`src/transport/p2p.cc`)

同机 GPU 间直传，绕开 CPU。`p2pType`（`p2p.cc:20`）：`P2P_DIRECT`（`cudaDeviceEnablePeerAccess`，NVLink/PCIe 直达）、`P2P_INTERMEDIATE`（经中间 GPU）、`P2P_IPC`（`cudaIpcOpenMemHandle`）、`P2P_CUMEM`（MNNVL 用 cuMem IPC）。`canConnect` 检查同 hostHash 且路径类型允许 P2P。这是单机多卡最快的路径。

### SHM (`src/transport/shm.cc`)

同机跨进程，用共享内存（`shm_open` + CUDA IPC）。P2P 不可用时的同机回退。环境变量 `NCCL_SHM_DISABLE` / `NCCL_SHM_USE_CUDA_MEMCPY` / `NCCL_SHM_MEMCPY_MODE` / `NCCL_SHM_LOCALITY` 控制行为。

### NET (`src/transport/net.cc`)

跨节点，经网络插件 `ncclNet_t`。内存映射类型 `NCCL_NET_MAP_*`（`net.cc:25`）：`HOSTMEM / DEVMEM / SHARED_HOSTMEM / SHARED_DEVMEM / GDCMEM`。`connect`/`accept` 走插件，CPU proxy 调 `isend`/`irecv`/`test`。建连句柄打包在 `ncclConnect.data[CONNECT_SIZE=256]` 里交换。

内置两个内部插件：`ncclNetIb`（`net_ib/`，IB/RoCE）和 `ncclNetSocket`（TCP 回退）。外部插件优先（见 [plugin-abi](plugin-abi)）。

### COLLNET (`src/transport/coll_net.cc`)

SHARP/交换机内归约。`canConnect`（`:143`）检查 `comm->ncclCollNet` 且 `graph->collNet`。`setup` 调 `ncclProxyConnect(comm, TRANSPORT_COLLNET, ...)`。分 chain（经 CPU proxy，`COLLNET_GROUP_NSUBS=8`）和 direct（GPU 直接）两种模式。把 AllReduce 的归约操作下推到 IB 交换机，减轻端点 GPU/CPU 压力。

### NVLS (`src/transport/nvls.cc`)

NVLink SHARP / 多播。用 CUDA `CUmulticast` 对象在 NVSwitch 上做多播和归约（CUDA 12.1+）。`ncclNvlsBufferSetup`（`:376`）分配 `mcBuff`/`mcCredit`（多播）和 `ucBuff`/`ucCredit`（单播）。`ncclNvlsTreeConnect`（`:269`）在 NVLS 上接树形拓扑。`NCCL_MAX_NVLS_ARITY=32`、`NCCL_MAX_NVLS_TREE_ARITY=3`。由 `NCCL_NVLS_ENABLE` 和 `comm->nvlsSupport/nvlsRegSupport` 控制。

## 选型与建连流程

建连分两步：先设位掩码，再统一握手。这把"哪条 channel 该连谁"和"怎么连"解耦。

### 第一步：设位掩码

```text
ncclTransportP2pConnect(comm, c, nRecv, recvRanks, nSend, sendRanks, connIndex)
  src/transport.cc:45
  只做: connectRecv[peer] |= 1ULL<<c;  connectSend[peer] |= 1ULL<<c;
  (不真正建连)
```

按拓扑形状批量调用：

```text
ncclTransportRingConnect  src/transport/generic.cc:14
  每 channel: ncclTransportP2pConnect(comm, c, 1, &ring.prev, 1, &ring.next, 0)
ncclTransportTreeConnect  generic.cc:49
  连 NCCL_MAX_TREE_ARITY 个 down + 1 个 up（及反向）
ncclTransportPatConnect  generic.cc:67
  PAT 算法的 binomial tree peers
```

### 第二步：握手建连

```text
ncclTransportP2pSetup(comm, graphs, ...connInfo...)   src/transport.cc:119
  对 i = 1..nRanks-1:
    recvPeer = (rank-i)%nR  sendPeer = (rank+i)%nR   (环形遍历邻居)
    每 peer 最多 ncclParamConnectRoundMaxPeers()=128 个一批
    对每个 mask 置位的 channel:
      ① selectTransport<IsSend>(comm, graph, connect, c, peer, connIndex, &type)
           transport.cc:24
         看 ncclPeerInfo + graph->typeIntra/typeInter → 选 p2p/shm/net
         调 transport->send/recv.setup，填一个 ncclConnect blob（listen 句柄/shm 名/net 句柄）
      ② 双向交换 ncclConnect:
         bootstrapSend/recv(peer, ncclConnect blob)    transport.cc:180
         (双方都拿到对方的 setup 数据)
      ③ while(!allChannelsConnected):
           connector->transportComm->connect(comm, data, 1, rank, conn)
           返回 ncclSuccess → connected=1
                    cudaMemcpyAsync 把 ncclConnInfo 拷进 device 镜像 devPeersHostPtr[peer]
                    ncclInProgress → 继续轮询
```

`selectTransport`（`transport.cc:24`）是选型核心：模板 `<int IsSend>`，综合 `ncclPeerInfo`（是否同机）和 `graph->typeIntra/typeInter`（路径类型）在 p2p/shm/net 间挑。选 net 时还记 `transportType` 供后续 `ncclProxyConnect`。

## 连接器与连接信息

建连产物是每 peer 每方向的 `ncclConnector`（`include/device.h:158`）：

```text
ncclConnector
├── connected / hasSeen / p2pOnly
├── proxyConn (ncclProxyConnector)   proxy 连接句柄
├── transportComm (ncclTransportComm*)  vtable 指针
├── transportResources               transport 私有资源
└── conn (ncclConnInfo)
     ├── buffs[NCCL_NUM_PROTOCOLS]   每 protocol 一个 ring buffer（recv 是本地，send 是远端）
     ├── mhandles[]                  内存注册句柄
     ├── tail / head                 fifo 计数器
     ├── flags / shared
     ├── stepSize
     ├── connFifo (ncclConnFifo*)    GPU↔proxy 的每步 {mode,offset,size,ptr}
     ├── step / llLastCleaning
     └── netDeviceHandle             网络设备句柄（GDR/GIN 等）
```

`ncclChannelPeer`（device.h:229）持有 `send[NCCL_MAX_CONNS]` / `recv[NCCL_MAX_CONNS]` 两个连接器（`NCCL_MAX_CONNS=2`：0 给 collective，1 给 p2p）。device 侧镜像 `ncclDevChannelPeer`（device.h:410）只保留 `ncclConnInfo`，砍掉 host-only 字段。

## 运行时连接（runtimeConn）

上面是初始化时全建连的路径。NCCL 还支持**懒建连**（`runtimeConn`）：当 `ncclParamRuntimeConnect()` 且 `cuMemSupport` 都开时，`initTransportsRank` 只跑 `setupChannel` + NVLS/CollNet setup，transport 连接推迟到首次使用时才建。这对大集群能显著缩短初始化时间，代价是首次 collective 有建连延迟。

## Proxy 连接（网络段）

网络 transport 的 `setup` 会调 `ncclProxyConnect`（`src/proxy.cc:1130`）建立 proxy 侧连接：

```text
ncclProxyConnect(comm, TRANSPORT_NET, 1, rank, &proxyConn)
  ├─ 连 peerAddresses[tpRank] 的 socket
  ├─ 发 ncclProxyMsgInit 请求（带 transport/send/tpLocalRank/tpRank/sameProcess）
  ├─ 收 ncclProxyInitResp：返回 connection 指针 + devShmPath
  │    (devShmPath 指向共享的 ncclProxyOpsPool 区域)
  └─ 填好 proxyConn
```

`ncclProxyOpsPool`（`include/proxy.h:222`）是 per-local-peer 的共享内存 ring，主线程把 `ncclProxyOp` 写进去，progress 线程从这里取——这是 host 和 proxy 线程交接工作的通道。连接状态机（`proxy.h:379`）：

```text
connUninitialized → connInitialized → connSharedInitialized
  → connSetupDone → connConnected
```

由 `ncclProxyMsgInit/Setup/Connect` 推进。服务侧 `proxyServiceInitOp`（`proxy.cc:1585`）读 socket 上的 (connection, reqSize, respSize, reqBuff, opId)，分派到 `tcomm->proxyConnect/proxySetup/proxyRegister`。详见 [proxy](proxy)。

## 内存注册

跨节点传输需要注册内存（pin 住，让网卡 DMA）。每个 transport 的 `proxyRegister` vtable 项在 proxy 服务线程里、首次有 proxy op 引用远端 buffer 时被调（经 `ncclProxyMsgRegister`）。对 IB 插件，这走 `regMr`/`regMrDmaBuf`（`include/plugin/net/net_v12.h`），支持 `NCCL_PTR_CUDA`（nv_peermem）和 `NCCL_PTR_DMABUF`（DMA-BUF，免 peermem）。详见 [features](features) 的 GDR 一节。

## 选择速查

| 场景 | 优先 transport | 触发条件 |
|---|---|---|
| 同机 GPU-GPU，NVLink/PCIe 直达 | P2P_DIRECT | 同 hostHash + PATH_NVL/PIX/PXB |
| 同机但 P2P 不可用 | SHM | 同 hostHash + PHB/SYS |
| 跨节点 | NET (IB/RoCE) | PATH_NET |
| 跨节点 + SHARP | COLLNET | graph->collNet && 插件支持 |
| 同机 NVSwitch 多播/归约 | NVLS | nvlsSupport + CUDA 12.1+ |

理解选型日志的关键：`ncclTransportP2pSetup` 里的 `selectTransport` 会按 `graph->typeIntra/typeInter` 选 transport，`NCCL_DEBUG=INFO` 里打印的 `via` 字段就是它选的结果。

## 小结

- 五种 transport 通过统一 vtable 抽象，`canConnect` + 路径类型决定选型。
- 建连两步走：`ncclTransportP2pConnect` 设位掩码 → `ncclTransportP2pSetup` 批量选型、交换 `ncclConnect` blob、循环 `connect` 直到 `connected=1`。
- 连接产物是 `ncclConnector` + `ncclConnInfo`（每 protocol 一个 ring buffer + fifo 计数器），host 状态镜像到 device。
- 网络/CollNet 段额外走 `ncclProxyConnect` 建 proxy 连接，经 `ncclProxyOpsPool` 交接工作。
- 懒建连（runtimeConn）对大集群缩初始化时间。

进一步：proxy 线程如何推进网络数据见 [Proxy 线程](proxy)；NVLS/CollNet 的底层能力见 [高级特性](features)。
