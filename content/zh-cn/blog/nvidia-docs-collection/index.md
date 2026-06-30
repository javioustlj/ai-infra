---
title: NVIDIA 官方文档资源汇总
linkTitle: NVIDIA 文档导航
description: >
  搜集整理 NVIDIA 各产品线的官方文档链接，涵盖 CUDA、TensorRT、Triton、NIM、RAPIDS 等 AI 基础设施核心组件，方便快速查阅。
date: 2026-06-30
author: AI Infra 编辑组
weight: 1
---

在构建和维护 AI 基础设施的过程中，NVIDIA 的软硬件生态是绕不开的核心。然而，NVIDIA 产品线众多，文档分散在不同站点，查找起来往往费时费力。本文将 NVIDIA 各主要产品线的官方文档链接汇总整理，方便大家快速定位所需资料。

---

## 🏠 总入口

| 名称 | 链接 | 说明 |
|------|------|------|
| **NVIDIA 文档中心** | [docs.nvidia.com](https://docs.nvidia.com/) | 所有 NVIDIA 产品文档的统一入口，可按产品类别浏览 |
| **NVIDIA 开发者门户** | [developer.nvidia.com](https://developer.nvidia.com/) | 开发者资源、SDK 下载、论坛与培训 |
| **NVIDIA API 目录** | [docs.api.nvidia.com](https://docs.api.nvidia.com/) | NVIDIA API Catalog 文档，涵盖 NIM 等 API 服务 |

---

## 🔧 CUDA 与基础计算库

| 名称 | 链接 | 说明 |
|------|------|------|
| **CUDA Toolkit** | [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/) | CUDA 编程指南、Runtime API、Driver API 及全部数学库（cuBLAS、cuFFT、cuRAND 等） |
| **CUDA Inline PTX Assembly** | [docs.nvidia.com/cuda/inline-ptx-assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html) | CUDA 内联 PTX 汇编参考，直接在 CUDA 代码中嵌入 GPU 汇编指令 |
| **CCCL（CUDA C++ Core Libraries）** | [nvidia.github.io/cccl](https://nvidia.github.io/cccl/unstable/cpp.html) | CUDA C++ 核心库（Thrust、CUB、libcudacxx），GPU 并行算法与原语基础库 |
| **cuDNN** | [docs.nvidia.com/cudnn](https://docs.nvidia.com/cudnn/) | CUDA 深度神经网络库，提供卷积、池化等 GPU 加速原语 |
| **NCCL** | [developer.nvidia.com/nccl](https://developer.nvidia.com/nccl#section-how-nccl-works) | NVIDIA 集体通信库，多 GPU / 多节点 AllReduce、Broadcast 等通信原语 |
| **NVSHMEM** | [developer.nvidia.com/nvshmem](https://developer.nvidia.com/nvshmem) | NVIDIA OpenSHMEM 实现，基于 PGAS 的 GPU 间细粒度共享内存通信 |
| **NVIDIA HPC SDK** | [docs.nvidia.com/hpc-sdk](https://docs.nvidia.com/hpc-sdk/) | HPC 编译器（C/C++/Fortran）、OpenACC/OpenMP 及 Nsight 性能分析工具 |
| **Compute Sanitizer** | [docs.nvidia.com/compute-sanitizer](https://docs.nvidia.com/compute-sanitizer/) | CUDA 内存错误检测与线程同步检查工具 |

---

## 🚀 深度学习推理与部署

| 名称 | 链接 | 说明 |
|------|------|------|
| **TensorRT** | [docs.nvidia.com/deeplearning/tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/) | 高性能深度学习推理优化器与运行时，含 Quick Start、ONNX 部署、算子参考等 |
| **Triton Inference Server** | [docs.nvidia.com/deeplearning/triton-inference-server](https://docs.nvidia.com/deeplearning/triton-inference-server/) | 生产级推理服务器，模型仓库配置、推理协议、后端集成与性能调优 |
| **NVIDIA NIM** | [docs.nvidia.com/nim](https://docs.nvidia.com/nim/) | NVIDIA 推理微服务，提供容器化的 AI 模型部署方案，含 API 参考与部署指南 |

---

## 📊 数据处理与数据科学

| 名称 | 链接 | 说明 |
|------|------|------|
| **RAPIDS** | [docs.nvidia.com/rapids](https://docs.nvidia.com/rapids/) | GPU 加速数据科学框架，涵盖 cuDF（数据帧）、cuML（机器学习）、cuGraph（图分析）等 |
| **DALI** | [docs.nvidia.com/dali](https://docs.nvidia.com/dali/) | GPU 加速数据加载与预处理库，支持多种图像/视频/音频格式与框架集成 |

---

## 🖥️ 数据中心 GPU 与管理

| 名称 | 链接 | 说明 |
|------|------|------|
| **Data Center GPU 驱动** | [docs.nvidia.com/datacenter/tesla](https://docs.nvidia.com/datacenter/tesla/) | Tesla/A100/H100 等 DC GPU 驱动安装、配置与发行说明 |
| **DCGM** | [docs.nvidia.com/datacenter/dcgm](https://docs.nvidia.com/datacenter/dcgm/) | 数据中心 GPU 管理器，GPU 集群监控与管理工具 |
| **DGX 平台** | [docs.nvidia.com/dgx](https://docs.nvidia.com/dgx/) | NVIDIA DGX 系统文档，硬件配置与软件栈部署 |
| **vGPU** | [docs.nvidia.com/vgpu](https://docs.nvidia.com/vgpu/) | 虚拟 GPU 软件，GPU 虚拟化与多实例 GPU（MIG）配置 |

---

## 🌐 网络与互连

| 名称 | 链接 | 说明 |
|------|------|------|
| **NVIDIA Networking 文档** | [networking-docs.nvidia.com](https://networking-docs.nvidia.com/) | 网络产品文档总入口：Cumulus Linux 交换机、Mellanox 网卡、DPU/DOCA 等 |
| **NVLink / NVSwitch** | [networking-docs.nvidia.com/interconnect](https://networking-docs.nvidia.com/interconnect) | NVLink 拓扑配置与 NVSwitch 高速互连规格 |

---

## ☁️ 云原生与企业部署

| 名称 | 链接 | 说明 |
|------|------|------|
| **NGC（GPU Cloud）** | [docs.nvidia.com/ngc](https://docs.nvidia.com/ngc/) | GPU 优化容器、预训练模型与 Helm Chart 仓库 |
| **Cloud Native Technologies** | [docs.nvidia.com/datacenter/cloud-native](https://docs.nvidia.com/datacenter/cloud-native/) | Kubernetes GPU 调度、容器工具包与云原生部署指南 |
| **NVIDIA AI Enterprise** | [docs.nvidia.com/ai-enterprise](https://docs.nvidia.com/ai-enterprise/) | 端到端企业 AI 平台，部署指南与生命周期管理 |

---

## 🤖 垂直行业平台

| 名称 | 链接 | 说明 |
|------|------|------|
| **Clara（医疗健康）** | [docs.nvidia.com/clara](https://docs.nvidia.com/clara/) | 医疗 AI 框架，医学影像、DICOM 适配与 AI 应用开发 |
| **Jetson（边缘 AI）** | [docs.nvidia.com/jetson](https://docs.nvidia.com/jetson/) | 嵌入式 AI 计算平台，JetPack SDK、多媒体 API 与传感器驱动 |
| **DRIVE（自动驾驶）** | [developer.nvidia.com/drive/documentation](https://developer.nvidia.com/drive/documentation) | 自动驾驶平台，DRIVE OS、SDK 与传感器集成 |
| **Omniverse** | [docs.nvidia.com/omniverse](https://docs.nvidia.com/omniverse/) | 3D 协作与仿真平台，Kit 开发、USD 合成与 RTX 渲染 |
| **Cosmos** | [docs.nvidia.com/cosmos](https://docs.nvidia.com/cosmos/) | 物理感知世界基础模型平台，用于自动驾驶与机器人的仿真训练 |

---

## 📚 学习资源

| 名称 | 链接 | 说明 |
|------|------|------|
| **Deep Learning Examples** | [github.com/NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) | 官方 SOTA 深度学习训练示例，覆盖 CV、NLP、推荐系统、语音等 |
| **Magnum IO SDK** | [developer.nvidia.com/magnum-io](https://developer.nvidia.com/magnum-io) | 加速 I/O 与数据处理 SDK，HPC 场景下的存储与通信优化 |
| **Hopper 架构深度解析** | [developer.nvidia.com/blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) | NVIDIA Hopper GPU 架构白皮书式深度解读，含 TMA、DPX、Transformer Engine 等新特性详解 |

---

## 💡 使用建议

1. **收藏总入口**：[docs.nvidia.com](https://docs.nvidia.com/) 几乎所有文档都能从这里导航到。
2. **关注版本号**：NVIDIA 文档按版本发布（如 TensorRT 10.x、CUDA 12.x），请确认文档版本与你使用的软件版本一致。
3. **善用搜索**：docs.nvidia.com 全站搜索功能强大，可直接搜索 API 名称或错误信息。
4. **Release Notes 必读**：每个产品的 Release Notes 是了解新特性、已知问题与兼容性变更的第一手资料。
5. **NGC 容器优先**：部署时优先使用 NGC 优化容器，已预装驱动、CUDA 及框架，省去环境配置烦恼。

---

> 📌 **本文持续更新中**，如发现链接失效或有遗漏的产品线，欢迎提交 PR 补充。
