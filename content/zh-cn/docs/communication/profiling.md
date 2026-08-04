---
title: 性能分析
description: 通信瓶颈定位方法与工具链。
weight: 4
---

分布式任务跑得慢，需要先把"通信等待""计算慢""数据供给慢"三类原因分开。

## 分层定位

- 计算侧：用 Nsight Systems / `torch.profiler` 看_KERNEL 执行时间，判断 GPU 是否真的在算。
- 通信侧：看 communication kernel（如 `ncclAllReduce`）占总时间的比例，以及是否与计算重叠。
- 数据侧：看 DataLoader、CPU 预处理和存储 IO 是否在等待。

## 常用工具

- `nccl-tests`：基准测试 AllReduce/AllGather/Broadcast 的带宽和延迟，建立基线。
- Nsight Systems：生成时间线，直观看到通信 kernel 与计算 kernel 是否重叠。
- NVTX：在代码里打标，把训练 step、前向、反向、优化器分段，方便定位通信调用点。
- NCCL Profiler Plugin / NCCL Inspector：采集集合通信操作的时间线与统计。
- 框架 profiler：PyTorch `torch.profiler`、TensorFlow profiler，关联通信与计算。

## 典型指标

- 通信占比：所有集合通信 kernel 时间 / step 总时间。占比高说明扩展受限。
- 带宽利用率：实测带宽 / 拓扑理论上限，低则排查拓扑和配置。
- 重叠率：通信与计算重叠的时间 / 通信总时间，反映梯度同步是否被掩盖。
- 等待时间：rank 之间的同步等待，可能揭示不均衡或慢节点。

## 参考

博客 [NCCL 集合通信与 GPU 性能分析资源汇总](../../blog/nccl-profiling-resources/) 汇总了 NCCL 和 Nsight/NVTX 的官方文档与工具链接。
