---
title: 通信与集合通信库
linkTitle: 通信库
description: NCCL、Gloo、UCX/UCC 等集合通信库的原理、拓扑、调优与排障。
weight: 4
---

分布式训练和分布式推理的扩展效率，很大程度取决于多 GPU、多节点之间的通信是否高效。通信库负责 AllReduce、AllGather、Broadcast 等集合通信原语，是 GPU 利用率、扩展比和任务稳定性的常见瓶颈来源。

## 关注什么

- 选型：NCCL、Gloo、MPI、UCX、UCC 各自适合什么场景，能否混用。
- 拓扑感知：NVLink / NVSwitch、PCIe、InfiniBand / RoCE、SHARP 如何影响通信路径和带宽。
- 调优：环境变量、channel 数、传输协议选择、缓冲区与 timeout 调参。
- 性能分析：如何判断瓶颈在通信、计算还是数据，常用 profiling 工具与指标。
- 故障排查：NCCL hang、超时、版本不匹配、P2P 不可用、IB 丢包等典型问题。

## 内容组织

- [选型与对比](libraries)：主流集合通信库的定位、差异和选型建议。
- [拓扑与传输](topology)：理解 GPU 间互连、网络拓扑如何决定通信路径。
- [NCCL 调优](nccl-tuning)：常用环境变量、参数和调优思路。
- [性能分析](profiling)：通信瓶颈定位方法与工具链。
- [故障排查](troubleshooting)：常见报错和排查路径。

## 与其他章节的关系

通信库是 [训练基础设施](../training/) 扩展效率的核心，也影响 [推理服务](../serving/) 中多副本和专家并行场景的延迟。GPU 利用率异常时，往往要先排除通信等待再查计算和数据。
