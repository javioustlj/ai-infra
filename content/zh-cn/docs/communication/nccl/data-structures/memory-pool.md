---
title: ncclMemoryPool：同尺寸对象的空闲链表池
description: 建在 ncclMemoryStack 之上的 per-type 空闲链表回收池；Cell 类型擦除、同尺寸不变量、为何后端必须是 memPermanent。
weight: 2
---

上一篇讲的 [`ncclMemoryStack`](memory-stack) 是个 LIFO 帧级分配器：成批对象靠 Pop 一次性回收，释放几乎不花代价。但它有一个短板——**回收是"成片"的，没法把单个对象还回去再单独复用**。一旦 Pop，整帧的小对象全没了；而很多场景里，对象的生命周期并不是整批同生共死，而是"用一个、放回一个、下次再取一个"。

`ncclMemoryPool` 就是补这个短板的。它是一个**同尺寸对象的空闲链表（free-list）**：Alloc 时先看链表头有没有空闲的，有就直接取（零 malloc），没有才向底层 `ncclMemoryStack` 要一块新的；Free 时把对象挂回链表头，留给下次 Alloc 复用。它本身不持有大块内存，真正的存储仍来自后端 stack——所以可以把它理解成"stack 之上的一层 per-type 回收缓存"。

源码全部内联在 `src/include/utils.h:318-365`，声明和注释在 `:142-158`。

> 相关阅读：这是 [ncclMemoryStack](memory-stack) 的姊妹结构。通信器上的六个 `memPool_*` 字段都以 `memPermanent` 为后端，见 [Communicator 结构与字段地图](../communicator#内存与任务池)。

## 整体模型：Cell、head、tail

struct 极简，就三个东西（`utils.h:320-326`）：

```cpp
struct ncclMemoryPool {
  struct Cell {
    Cell *next;
  };
  struct Cell* head;
  struct Cell* tail; // meaningful only when head != nullptr
};
```

`Cell` 是个"代理头"，只有一个 `next` 指针。`head` 是空闲链表的头，`tail` 是尾部（注释特意说 `tail` 只在 `head != nullptr` 时有意义，空池时 `tail` 是悬空值）。单看这个结构，`Cell` 和 `head/tail` 描绘的就是一条单向链表：

```
ncclMemoryPool
 │
 └─ 空闲链表:  head ──► [Cell|next] ──► [Cell|next] ──► ... ──► [Cell|next=null]  ◄── tail
                          ▲
                          └─ Alloc 从这里摘; Free 往这里插（头插）
```

但 `Cell` 并不是一个独立分配的对象——它是**复用已分配对象内存的第一个字**做出来的。这是整个池最关键的一个技巧，下面展开。

## Cell 的类型擦除：对象内存的第一个字当 next 指针

`Cell` 只有一个 `next` 指针，大小正好是一个机器字（8 字节 on x86-64）。当一个 `T` 类型的对象被 Free 时，代码并不另外分配一个 `Cell` 来登记它，而是直接把那个对象的指针 reinterpret 成 `Cell*`，把对象内存的**第一个字**改写成 `next`：

```cpp
template<typename T>
inline void ncclMemoryPoolFree(struct ncclMemoryPool* me, T* obj) {
  using Cell = ncclMemoryPool::Cell;
  Cell* cell = reinterpret_cast<Cell*>(obj);   // 同一块内存,换个视角
  cell->next = me->head;                        // 把第一字写成 next
  if (me->head == nullptr) me->tail = cell;
  me->head = cell;                              // 头插
}
```

这能成立的前提是：**每个被池管理的对象，其大小至少能容纳一个 `Cell`**。Alloc 里用 `cellSize = max(sizeof(Cell), sizeof(T))` 保证了这一点——对象比 Cell 大就用对象自己的大小，比 Cell 小（不太可能，因为 Cell 只有一个指针）就至少分配 Cell 大小。于是每个 cell 的头一个字在"空闲态"时当 `next` 用，在"占用态"时是对象 `T` 的第一个字段。两种状态共用同一块内存，靠对象"死/活"来切换语义：

```
占用态(对象T活着)            空闲态(对象已Free,挂回链表)
┌───────────────┐            ┌───────────────┐
│ T 的第一字段   │            │  next 指针    │ ──► 指向原 head
│ T 的第二字段   │            │  (其余字节是  │
│ ...           │            │   残留,不再被 │
│ T 的末字段     │            │   当对象读)   │
└───────────────┘            └───────────────┘
   Cell* 视角同这块            Cell* 视角同这块
```

Free 时对象已经"逻辑死亡"，调用方不再读它，所以把第一字覆写成 `next` 是安全的。Alloc 时从链表摘下这块，紧接着 `memset(cell, 0, sizeof(T))` 清零，第一字的 `next` 残留也被抹掉，调用方拿到一个干净的 `T*`。这就是类型擦除的来回：同一块内存，活着时是 `T`，空闲时是链表节点。

## Alloc：快路径摘头，慢路径向后端要

```cpp
template<typename T>
inline T* ncclMemoryPoolAlloc(struct ncclMemoryPool* me, struct ncclMemoryStack* backing) {
  using Cell = ncclMemoryPool::Cell;
  Cell* cell;
  if (COMPILER_EXPECT(me->head != nullptr, true)) {   // 快路径:链表非空
    cell = me->head;
    me->head = cell->next;                            // 摘头
  } else {                                            // 慢路径:链表空,向后端 stack 要
    size_t cellSize = std::max(sizeof(Cell), sizeof(T));
    size_t cellAlign = std::max(alignof(Cell), alignof(T));
    cell = (Cell*)ncclMemoryStack::allocate(backing, cellSize, cellAlign);
  }
  memset(cell, 0, sizeof(T));                          // 清零 sizeof(T)
  return reinterpret_cast<T*>(cell);
}
```

快路径和 `ncclMemoryStack::allocate` 一样干脆：链表非空就摘头，一次指针读加一次赋值，零 malloc。`COMPILER_EXPECT(..., true)` 提示这条几乎总成立——只要池"热"起来（有 Free 过的可用 cell），后续 Alloc 都走这条。慢路径只在池冷启动、或当前回收的 cell 不够时才走，这时向 `backing` 这个 `ncclMemoryStack` 调它的内部 `allocate` 要一块新内存。

两个细节值得注意。一是慢路径用的是 `ncclMemoryStack::allocate`（内部函数），不是公开的 `ncclMemoryStackAlloc`——因为后者会 `memset` 整块，而这里只要 `memset(sizeof(T))`，省掉 cellSize 大于 sizeof(T) 时尾部那几个字节的清零。二是 `cellSize`/`cellAlign` 取 `Cell` 和 `T` 的较大者，既保证 cell 装得下 `Cell`（类型擦除前提），又保证装得下 `T` 且对齐满足两者。

## 同尺寸不变量：一个池只能管一种 T

头部注释明确要求（`utils.h:143-148`）：

> A free-list of same-sized allocations. It is an invalid for a pool instance to ever hold objects whose type have differing (sizeof(T), alignof(T)) pairs.

也就是说，**一个 `ncclMemoryPool` 实例从构造到销毁，只能服务于同一种 `(sizeof, alignof)` 的对象**。原因看 Free/Alloc 就明白：Free 把对象按 `Cell*` 挂链，Alloc 取下来当 `T*` 用。如果同一个池先管 `T1`（比如 40 字节）再管 `T2`（比如 200 字节），那从链表摘下的某个旧 `T1` cell 只有 40 字节，却要被当成 200 字节的 `T2` 用——越界写坏后续内存。所以混用是"invalid"，不是"可能出错"，是定义上就不允许。

NCCL 靠"每个类型一个独立池"来满足这条不变量。通信器上挂了六个池（`comm.h:699-705`），各自专管一种任务结构：

```cpp
// pools backed by comm->memPermanent
struct ncclMemoryPool memPool_ncclTaskBcast;   // ncclTaskBcast
struct ncclMemoryPool memPool_ncclTaskColl;    // ncclTaskColl  (最常用)
struct ncclMemoryPool memPool_ncclTaskP2p;     // ncclTaskP2p
struct ncclMemoryPool memPool_ncclTaskRma;     // ncclTaskRma
struct ncclMemoryPool memPool_ncclProxyOp;     // ncclProxyOp
struct ncclMemoryPool memPool_ncclKernelPlan;  // ncclKernelPlan
```

字段名里就带上了类型名，是一种"用命名固化不变量"的约定——每个池天然只被对应类型的 Alloc/Free 访问。

## Free 不把内存还给 stack：所以后端必须是 memPermanent

这是理解 `ncclMemoryPool` 最重要的一点，也是个容易和 `ncclMemoryStack` 混淆的地方。

`ncclMemoryPoolFree` 只是把 cell 挂回自己的空闲链表，**完全没有通知后端 stack**。那块内存从后端 stack 的 bumper 里 carve 出来之后，就永远"脱离"了 stack 的 LIFO 流——stack 的 bumper 不会因为 pool Free 而回拨，stack Pop 也不会把这些 cell 收回去。换句话说，**一旦某块内存被 pool 拿去当 cell，它在 stack 视角里就是"已分配、占着不走"的，直到 stack 整体 Destruct 才随 hunk 一起释放。**

这就引出一个硬约束：**后端 stack 里承载这些 cell 的区段，不能被 Pop 弹掉**。否则 Pop 一回拨 bumper，pool 空闲链表里那些 cell 指向的内存就成了"逻辑作废区"，下次 Alloc 摘下来用的是被上层覆盖过的脏数据，甚至指针悬空。头部注释说的正是这件事：

> If memory backing any currently held object is deallocated then it is an error to do anything other than reconstruct it, after which it is a valid empty pool.

"deallocated"在这里就是指后端 stack 被 Pop/Destruct 掉了对应区段。一旦发生，这个 pool 只能重建（Construct 清空链表），不能再用。

所以 NCCL 的六个池**全部以 `memPermanent` 为后端**（每个 Alloc 调用都传 `&comm->memPermanent`）。`memPermanent` 从不 Push/Pop，只有 nil frame 一层，区段永不回拨——正好满足"cell 所在区段不被 deallocate"的要求。用 `memScoped` 当后端会很危险：每个 group 结束 Pop 一次，cell 瞬间全悬空。

这里有个有意思的张力：**任务对象（task）在语义上是 group 作用域的——一个 group 里分配、同一个 group 里用完放回**；但它们的**存储**却放在 `memPermanent` 这个永久栈上、靠 pool 跨 group 回收。也就是说，逻辑生命周期（per-group）和存储生命周期（per-comm）被故意拆开了：group 内 Alloc/Free 来回倒腾的是 pool 里的 cell，而这些 cell 的物理内存在通信器存活期一直被持有、跨 group 反复复用。这样既享受了 per-group 的"用完就还"语义，又避免了每个 group 都向 stack 真的要新内存（只有池冷启动那一下要）。

## TakeAll：批量转移（当前未被使用）

```cpp
inline void ncclMemoryPoolTakeAll(struct ncclMemoryPool* me, struct ncclMemoryPool* from) {
  if (from->head != nullptr) {
    from->tail->next = me->head;              // from 的尾接上 me 的头
    if (me->head == nullptr) me->tail = from->tail;
    me->head = from->head;                     // me 的新头是 from 的头
    from->head = nullptr;                      // from 清空
  }
}
```

把 `from` 池里所有空闲 cell 整条搬到 `me` 池头。O(1) 操作，只动头尾指针。这是为"线程局部池向全局池归还"之类场景准备的——比如某线程自己攒了一池 cell，结束时一把倒回公共池。目前在 NCCL 里没有调用点（定义了但未使用），属于预留设施。`tail` 字段存在的全部意义就是让这个 O(1) 批量转移成为可能：没有 `tail`，要把 `from` 整条接上 `me` 就得遍历 `from` 找尾，退化成 O(n)。所以 `head` 够日常 Alloc/Free 用，`tail` 专门为 TakeAll 服务。

## 在 NCCL 里的实际用法

六个池的 Alloc/Free 分布在 `enqueue.cc`、`group.cc`、`rma/`、`scheduler/` 里，模式高度一致：**group 内按需 Alloc 一个任务结构、填好字段挂进调度队列；用完（或 group 结束清理时）Free 回池**。几个典型调用点：

分配一个 collective 任务（`enqueue.cc:2673`），用完在 group 清理时放回（`enqueue.cc:1440`）：

```cpp
// group 内入队时分配
struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
t->func = info->coll;
...
// group 提交完、清理时放回池
ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, ct);
```

RMA 子系统大量用 `memPool_ncclTaskRma`，包括把一个 WaitSignal 任务拆成 CE/Proxy 两个任务时新 Alloc（`rma.cc:206/221`）、拆完把原任务 Free（`rma.cc:241`）、proxy/ce 处理完各自 Free（`rma_proxy_launch.cc:237`、`rma_ce.cc:231`）。

Kernel plan 的 Alloc 在 `enqueue.cc:1530`，Free 在提交完成后 `enqueue.cc:1478` 和 `group.cc:419`。

注意一个构造上的小不对称：只有 `memPool_ncclKernelPlan` 和 `memPool_ncclProxyOp` 在 `init.cc:517-518` 显式调了 `ncclMemoryPoolConstruct`；其余四个任务池没显式构造，靠通信器整体分配时的零初始化（`head=nullptr` 等价于 Construct）——注释 `:152` 说 "Equivalent to zero-initialization"，所以这个省略是安全的。

## 和 ncclMemoryStack 的关系：回收缓存 vs 内存来源

把两个结构放一起看，分工很清楚：

- **`ncclMemoryStack` 是内存来源**：真正持有大块 hunk、做 bump 分配、负责 Pop 的成片回收和 Destruct 的彻底释放。
- **`ncclMemoryPool` 是回收缓存**：不持有大块内存，只在 stack 之上维护一条 per-type 空闲链表，让单个对象能"还回去—再取出"地跨次复用。

两者配合出一个分层的热路径：一次 group 里第一次需要 `ncclTaskColl` 时，pool 链表空，向 `memPermanent` stack 要一块新 cell（走 stack 的 bump 快路径）；用完 Free 回 pool 链表。下一次 group 再要 `ncclTaskColl`，pool 链表非空，直接摘头（连 stack 都不碰）。于是**稳态下 group 之间的任务对象几乎零实际分配**——全在 pool 链表里来回倒，只有总量超过历史峰值时才偶尔向 stack 要新块。

至于"pool Free 不还 stack、物理内存常驻到通信器销毁"这一点，前面解释过为什么这反而正是想要的行为：它换来了稳态零分配。代价是 `memPermanent` 的 hunk 会随历史峰值留存——但通信器本就是长生命周期对象，这点常驻是可接受的。

## 设计要点回顾

`ncclMemoryPool` 用最小的结构（一个 `Cell`、一个 `head`、一个 `tail`）实现了同尺寸对象的高效回收。Cell 的类型擦除让空闲对象的第一字兼做链表节点，零额外元数据开销。同尺寸不变量靠"每类型一池"的命名约定保证，混用类型是定义上的非法。Alloc 快路径摘头、慢路径才向后端 stack 要，稳态全走快路径。

最关键的约束是 Free 不把内存还给 stack——cell 一旦从 stack carve 出来就脱离了 LIFO 流，所以后端必须是 `memPermanent` 这种永不 Pop 的栈，否则 Pop 会让整条空闲链表悬空。NCCL 把六个池都挂在 `memPermanent` 上，让任务对象在逻辑上 per-group 用完即还、在物理上跨 group 常驻复用，用存储的常驻换来了稳态的零分配。`tail` 字段和 `TakeAll` 为批量转移预留，当前未启用。

## 源码索引

- 头部注释与前置声明：`src/include/utils.h:142-158`
- 结构体定义：`src/include/utils.h:320-326`
- Construct：`src/include/utils.h:328-330`
- Alloc（快/慢路径）：`src/include/utils.h:332-347`
- Free（头插）：`src/include/utils.h:349-356`
- TakeAll（批量转移）：`src/include/utils.h:358-365`
- 通信器六个池字段：`src/include/comm.h:699-705`
- 池构造调用点：`src/init.cc:517-518`
- 典型 Alloc/Free 调用：`src/enqueue.cc`（多处）、`src/group.cc:417-419`、`src/rma/rma.cc:206-241`
