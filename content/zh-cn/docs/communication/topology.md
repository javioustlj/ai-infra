---
title: 拓扑与传输
description: GPU 互连与网络拓扑如何决定通信路径和带宽。
weight: 2
---

通信性能的上限由物理互连决定。理解拓扑才能解释为什么同样的 AllReduce 在不同节点上带宽差几倍。

## 互连层级（由快到慢）

- NVLink / NVSwitch：GPU 间直连，带宽最高（NVLink 4 单向可达 50 GB/s 量级），是单机多卡首选路径。
- PCIe：GPU 经 PCIe 总线互联，受 Root Complex 和 NUMA 拓扑影响，跨 NUMA 带宽明显下降。
- InfiniBand / RoCE：节点间高速网络，常配合 RDMA 和 GPUDirect 降低 CPU 参与。
- 以太网：成本最低，性能和延迟最弱，适合控制面或非关键通信。
- SHARP：在 IB 交换机中做集合通信归约，减轻端点 CPU/GPU 压力，适合大规模 AllReduce。

## 拓扑感知的关键点

- NUMA 亲和：GPU 和网卡归属同一个 NUMA 节点时带宽最高，跨 NUMA 会损失性能。
- P2P 可用性：NVLink/PCIe P2P 是否启用直接决定 GPU 间能否零拷贝传输。
- 网络拓扑：机架内、Leaf-Spine、多跳路径的带宽和拥塞差异，影响 ring/tree 算法的选择。
- GPUDirect RDMA：让 GPU 显存直接经网卡收发，避开 CPU 中转和内存拷贝。

## 排查工具

- `nvidia-smi topo -m`：查看单机 GPU 互连关系（NVLink / SYS / PHB 等）。
- `nccl-topo` / NCCL 日志：查看 NCCL 选定的传输路径和 channel。
- IB 工具：`ibstat`、`ibv_devinfo`、`perfquery` 查看链路速率和误码。
