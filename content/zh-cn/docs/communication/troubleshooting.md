---
title: 故障排查
description: NCCL 常见报错与排查路径。
weight: 5
---

通信库的故障往往表现为训练 hang 住、初始化超时或带宽异常，需要一套固定的排查路径。

## NCCL hang / 初始化卡住

- 多 rank 启动不同步：某个 rank 启动慢或未起来，其他 rank 在集合操作处无限等待。
- 网络不通：防火墙、IB 子网、PKey、RoCE 配置阻止 P2P 或 RDMA。
- 网卡选错：NCCL 选到管理网，用 `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` 显式指定。
- 容器权限不足：P2P、RDMA、`/dev/infiniband` 未正确挂载或 capabilities 缺失。

## 超时类报错

- `NCCL_IB_TIMEOUT` / `watchdog` 超时：常见于长距离、高负载或误码链路。
- 排查链路：`ibstat`、`perfquery` 看速率和误码，必要时放宽超时作为临时缓解。

## 版本与兼容

- CUDA / 驱动 / NCCL / 框架版本不一致：高频且隐蔽，先核对四个版本。
- NCCL 升级后行为变化：算法或默认参数改变，用 `NCCL_DEBUG=INFO` 对比。

## 性能异常

- 带宽远低于理论上限：先查拓扑（`nvidia-smi topo -m`、NUMA 亲和），再查链路和 P2P。
- 个别节点慢导致整体退化：用 `nccl-tests` 的多节点模式定位慢 rank。
- 间歇性抖动：多为网络丢包或拥塞，结合 IB 计数和交换机监控排查。

## 排查顺序建议

1. 物理/链路层：链路速率、误码、是否可用。
2. 拓扑/权限：P2P、NUMA、容器挂载。
3. 配置：网卡、超时、算法选择。
4. 版本：驱动/CUDA/NCCL/框架一致性。
5. 算法/参数：确信底层正常后再调环境变量。
