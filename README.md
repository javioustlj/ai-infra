# AI Infra 技术文档

这是一个基于 [Hugo](https://gohugo.io/) 和 [Docsy](https://www.docsy.dev/) 的 AI Infra 技术文档站，模板来自 [docsy-example](https://github.com/google/docsy-example)，发布目标是 GitHub Pages。

Docsy 通过 Hugo Module 引入，版本由 `go.mod` / `go.sum` 管理。

## 本地预览

需要 Hugo extended 版本和 Go。仓库根目录执行：

```bash
hugo server
```

默认地址是 <http://localhost:1313/>。

## 生产构建

```bash
hugo --minify
```

构建结果会生成到 `public/`，该目录不提交到仓库。

## 更新 Docsy

```bash
hugo mod get -u github.com/google/docsy/theme
hugo mod tidy
```

## GitHub Pages

仓库包含 `.github/workflows/pages.yaml`。推送到 `main` 分支后，GitHub Actions 会构建 Hugo 站点，上传 GitHub Pages artifact，并由 Pages 部署这个 artifact。

首次使用时，在 GitHub 仓库里打开：

```text
Settings -> Pages -> Build and deployment -> Source -> GitHub Actions
```

站点地址：

```text
https://javioustlj.github.io/ai-infra/
```

## 内容结构

```text
content/zh-cn/
  _index.md                 # 首页
  docs/                     # 技术文档
  search.md                 # Docsy 搜索页
  site.md                   # 站点信息
```

`content/en`、`content/no` 和 `content/fa` 是从 docsy-example 带来的参考内容，当前 Hugo 配置只构建 `content/zh-cn`。
