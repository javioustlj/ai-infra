---
title: 参考资料
description: 术语、检查清单和常用命令。
weight: 7
---

这里用于放置可以反复查阅的短文档。

## 术语

- GPU 利用率：GPU 正在执行计算的时间占比，不能单独代表训练或推理效率。
- 显存利用率：模型权重、激活、KV cache 或 batch 数据占用显存的比例。
- Dynamic batching：在很短时间窗口内合并多个请求，以提升加速器吞吐。
- Checkpoint：训练过程中的状态快照，用于恢复、评估或后续微调。
- SLO：面向用户承诺的服务目标，例如可用性或 P99 延迟。

## 常用命令

```bash
hugo server
hugo --minify
hugo mod graph
hugo mod tidy
```
