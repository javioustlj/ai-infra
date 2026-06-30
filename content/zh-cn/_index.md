---
title: AI Infra 技术文档
description: 面向训练、推理、数据和平台工程的 AI 基础设施知识库。
params:
  body_class: td-navbar-links-all-active
---

{{% blocks/cover
  title="AI Infra 技术文档"
  height="full td-below-navbar"
  image_anchor="center"
%}}

从 GPU 集群、分布式训练、推理服务到可观测性，把 AI 基础设施的工程知识整理成可以持续演进的文档。
{.display-6}

<div class="td-cta-buttons my-5">
  <a {{% _param btn-lg primary %}} href="docs/">
    开始阅读
  </a>
  <a {{% _param btn-lg secondary %}}
    href="{{% param github_repo %}}"
    target="_blank" rel="noopener noreferrer">
    GitHub
    {{% _param FA brands github "" %}}
  </a>
</div>

{{% blocks/link-down color="info" %}}

{{% /blocks/cover %}}

{{% blocks/lead color="white" %}}

这个站点用于沉淀 AI Infra 的架构、实践和排障经验。内容会优先围绕工程师每天真正会遇到的问题展开：容量怎么估、训练为什么慢、推理链路如何降延迟、GPU 利用率怎么解释、平台接口如何设计。

{{% /blocks/lead %}}

{{% blocks/section color="primary" type="row" %}}

{{% blocks/feature title="架构地图" icon="fa-diagram-project" url="docs/architecture/" %}}

梳理数据、训练、模型仓库、推理网关、调度和可观测性的关系，先建立全局视角。

{{% /blocks/feature %}}

{{% blocks/feature title="工程实践" icon="fa-screwdriver-wrench" url="docs/serving/" %}}

记录可复用的配置、容量估算、上线检查和故障处理方法，让经验可以被复盘和复用。

{{% /blocks/feature %}}

{{% blocks/feature title="持续更新" icon="fab fa-github" url="https://github.com/javioustlj/ai-infra" %}}

站点源码托管在 GitHub，通过 GitHub Pages 自动发布。每一次提交都可以成为一次文档迭代。

{{% /blocks/feature %}}

{{% /blocks/section %}}

{{% blocks/section color="white" type="row" %}}

{{% blocks/feature title="推荐阅读顺序" icon="fa-route" %}}

从 [学习路径](docs/getting-started/) 开始，再进入 [系统架构](docs/architecture/)、[训练基础设施](docs/training/) 和 [推理服务](docs/serving/)。

{{% /blocks/feature %}}

{{% blocks/feature title="写作原则" icon="fa-pen-nib" %}}

每篇文档尽量回答三个问题：问题背景是什么、关键权衡在哪里、落地时应该检查什么。

{{% /blocks/feature %}}

{{% blocks/feature title="适合人群" icon="fa-users-gear" %}}

适合关注 AI 平台、GPU 集群、MLOps、模型服务和云原生基础设施的工程师。

{{% /blocks/feature %}}

{{% /blocks/section %}}
