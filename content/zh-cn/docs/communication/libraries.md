---
title: 选型与对比
description: NCCL、Gloo、MPI、UCX、UCC 的定位与选型。
weight: 1
---

集合通信库的选择取决于硬件栈、框架集成和性能要求。多数 AI 工作负载以 NCCL 为主，其他库在特定场景下补充或替代。

## 主流通信库

- NCCL：NVIDIA 为 GPU 集合通信优化的库，是 PyTorch、TensorFlow 在 NVIDIA 平台的默认后端，支持 NVLink、InfiniBand、RoCE 和 SHARP。
- Gloo：Facebook 开发的集合通信库，CPU 后端成熟、部署简单，常用于无 GPU 或调试场景；GPU 通信能力弱于 NCCL。
- MPI（OpenMPI / MPICH）：HPC 领域通用，CPU 间通信成熟，支持 GPU CUDA-aware MPI；集成复杂，生态偏传统 HPC。
- UCX：统一通信框架，抽象 IB/RoCE/Shared Memory/NCCL 等底层传输，常作为 MPI 和框架的传输层。
- UCC：基于 UCX 的集合通信库，目标是把 NCCL 类 GPU 通信和 CPU 集合通信统一到一套接口。

## 选型建议

- NVIDIA GPU + 主流深度学习框架：默认 NCCL，优先发挥 NVLink 和 SHARP。
- 仅 CPU 或调试/小规模：Gloo 部署最轻。
- 与 HPC 生态深度集成、已有 MPI 工具链：CUDA-aware MPI，必要时用 UCX 加速。
- 混合后端：框架允许不同操作用不同后端（例如参数同步用 NCCL，控制面用 Gloo）。

## 常见误区

- 把 NCCL 当成"通用最快"：在非 NVIDIA 硬件或不支持的网络上未必最优。
- 忽视版本匹配：驱动、CUDA、NCCL、框架四者版本不一致是高频故障源。
- 只看带宽不看延迟：AllReduce 的树形/环形算法在小消息时延迟敏感，大消息时带宽敏感。
