---
title: NCCL 调优
description: 常用环境变量、参数和调优思路。
weight: 3
---

NCCL 在多数情况下"开箱即用"，但大规模或特殊拓扑下需要调参才能逼近物理带宽上限。

## 关键环境变量

- `NCCL_DEBUG`：设为 `INFO` 或 `WARN` 查看选路、channel 和初始化信息，调优第一步。
- `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA`：指定通信网卡，避免选到管理网或回环口。
- `NCCL_NET`：选择网络插件（如 `IB`、`SOCKET`、`NCCL-RDMA-SHARP` 插件）。
- `NCCL_P2P_LEVEL`：控制 P2P 使用的层级（NVLink / PIX / PXB / PHB / SYS），用于绕开不可用路径。
- `NCCL_BUFFSIZE` / `NCCL_NCHANNELS`：缓冲区大小与 channel 数，影响带宽利用和负载均衡。
- `NCCL_TIMEOUT` / `NCCL_IB_TIMEOUT`：超时设置，长距离或拥塞网络需要放宽。
- `NCCL_ALGO` / `NCCL_PROTO`：强制选择算法（Ring / Tree / CollNet）和协议（Simple / LL / LL128）。

## 调优思路

1. 先确认物理层正常：链路速率、P2P 可用、网卡归属正确的 NUMA。
2. 用 `NCCL_DEBUG=INFO` 看 NCCL 实际选的算法、channel 数和传输路径。
3. 单机先用 `nccl-tests` 测 AllReduce/AllGather 带宽，对照理论上限。
4. 多机分别测机内和跨机带宽，定位是拓扑、网络还是配置问题。
5. 只在基线确认后再调环境变量，一次只改一个。

## 常见坑

- 容器里 P2P 被禁用：需开启 `--privileged` 或挂载 `/dev/nvidia*` 并配置 capabilities。
- NCCL 选到管理网卡：带宽骤降，需显式指定高速网卡。
- IB 链路误码高：表现为带宽抖动和 timeout，先查硬件和光模块，不要先调参。
