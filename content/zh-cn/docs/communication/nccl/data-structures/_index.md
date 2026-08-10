---
title: 数据结构剖析
linkTitle: 数据结构剖析
description: NCCL 关键内部数据结构的源码级拆解：设计动机、字段布局、操作语义与使用场景。
weight: 20
---

本节聚焦 NCCL 内部那些"不显眼但支撑了一切"的关键数据结构。这类结构通常定义在 `src/include/` 下、被整个子系统共用，理解了它们，读热路径代码会顺畅很多。每个结构按"设计动机 → 字段布局 → 各操作的语义 → 在 NCCL 里被谁怎么用 → 容易卡住的细节"为主线，所有描述以 `v2.30.4-1` 源码为准。

## 本节内容

- [ncclMemoryStack：LIFO 帧级内存池](memory-stack)：NCCL 热路径上的临时内存来自哪里、为什么它的释放几乎不花代价。
- [ncclMemoryPool：同尺寸对象的空闲链表池](memory-pool)：建在 `ncclMemoryStack` 之上的 per-type 回收缓存，让单个对象能跨次复用；为什么后端必须是 `memPermanent`。
- [ncclIntruQueue：侵入式单链表队列](intru-queue)：用成员指针模板把 next 字段交给调用方的侵入式 FIFO；O(1) 入队/出队/批量拼接，以及 MPSC 变体的无锁多生产者入队。
- [ncclIntruAddressMap：侵入式指针哈希表](address-map)：类型擦除的侵入式哈希表，key/next 留在调用方对象里；Knuth 乘法哈希、懒初始化、2:1 扩容再哈希、删空自动回收，进程级窗口索引就用它。

前三件容器构成 NCCL 热路径的"分配→排队→回收"链：stack 是地基（持有大块内存、bump 分配），pool 是回收层（per-type 空闲链表、跨次复用），queue 是组织层（把任务节点串成 FIFO 交给调度器）。第四件 `ncclIntruAddressMap` 是旁挂的索引层——不在任务生命周期链上，而是服务"按 key 查对象"的场景，和 queue 同属成员指针模板驱动的侵入式家族，一个排队一个查找，各管一类关系组织。
