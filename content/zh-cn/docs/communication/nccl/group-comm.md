---
title: Group 提交模型与 Communicator 线程模型
description: group 批量提交、thread 如何用链表维护它拥有的 comm、join/leave 哨兵语义，以及两条不可妥协的硬约束。
weight: 4
---

读 NCCL 源码时，最容易在上手阶段卡住、但一旦想通就豁然开朗的，是 **thread / comm / group 三者的关系**。这篇文档专门把这三者的关系讲透：group 是什么、它怎么绑定到线程上、线程又怎么用一条链表"记住"本轮 group 里涉及了哪些 comm，以及由此生出的两条使用上的硬约束。

理解这些，是排查"任务挂起""跨 stream 并发报错""非阻塞模式下复用 comm 被拒绝""跨线程共享 comm 出现静默错误"等问题的前提。本文基于 NCCL `v2.30.4-1` 源码。

## 先建立一个直觉：group 是一次"批量提交"

NCCL 的大部分用户 API（`ncclSend`、`ncclAllReduce`…）有一个反直觉的特性：**调用它们并不会立刻启动任何 GPU kernel，也不会发出任何网络包。** 它们只是把这次操作"记一笔账"，塞进 comm 内部的任务队列里。真正的连接建立、数据搬运、kernel 发射，全部推迟到 group 结束的那一刻才发生——group 结束是 NCCL 的"提交点"。

```cpp
ncclGroupStart();
ncclAllReduce(comm1, ...);   // 只记账
ncclSend(comm2, ...);        // 只记账
ncclAllReduce(comm3, ...);   // 只记账
ncclGroupEnd();              // 这里才一次性发射上面三笔操作
```

分两步走（先记账、后提交）的好处是：一个 group 里多个操作可以统一调度——相邻任务能合并成一个 kernel、多个 comm 的 kernel 能跨 rank 对齐时机、出错时能整体回滚。这也是为什么把多个集合操作包在 `ncclGroupStart/End` 里会比逐个调用更快。

### 即使不写 GroupStart/End，每次 API 仍隐式包了一层 group

这里有个容易忽略的细节：哪怕你只调一次 `ncclAllReduce`、根本没碰 `ncclGroupStart/End`，这次调用也**完整走了 group 的提交流程**。区别只在于——这层 group 是 NCCL 偷偷替你开的。

机制藏在所有 API 的公共入口 `ncclEnqueueCheck` 里（`src/enqueue.cc:3016`）。每一个 `ncclAllReduce`/`ncclSend`/… 最后都会调到它，而它进函数第一件事就是开 group、出函数前关 group：

```cpp
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  // ① 开 group：把嵌套深度 +1
  NCCLCHECK(ncclGroupStartInternal());          // enqueue.cc:3029

  // ② 通信器就绪检查、参数校验
  NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);   // :3033
  NCCLCHECKGOTO(ArgsCheck(info), ret, fail);                   // :3041

  // ③ 记账：把这次操作挂进 comm 的 planner 队列（不发射）
  NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);      // :3048

exit:
  ncclGroupErrCheck(ret);
  // ④ 关 group：把深度 -1，归零时才真正触发发射
  NCCLCHECK(ncclGroupEndInternal());            // enqueue.cc:3053
  return ret;
fail:
  goto exit;
}
```

而 `ncclGroupStartInternal` / `ncclGroupEndInternal` 本身非常薄（`src/include/group.h:90`）：

```cpp
inline ncclResult_t ncclGroupStartInternal() {
  ncclGroupDepth++;        // 只是深度 +1
  return ncclSuccess;
}
```

关键是 `ncclGroupEndInternal`（`src/group.cc:753`）里这一句——**只有当深度减到 0 时才真正干活**：

```cpp
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo) {
  ...
  if ((--ncclGroupDepth) > 0) goto exit;   // 深度没归零，直接返回什么都不做
  ...
  // 深度归零了：才走 doLaunches，把攒着的任务一次性发射
}
```

这就是隐式 group 和显式 group 能共存、且互不干扰的关键。`ncclGroupDepth` 是个**嵌套计数器**：

- **单次 API 调用（你不写 GroupStart/End）**：进 `ncclEnqueueCheck` 时深度从 0→1，出时从 1→0 归零，立即触发发射。所以这次调用看起来是"同步"的——调完任务就被发射了，但底层其实走完了一整个 group 流程，只是这个 group 里只有一个任务。
- **显式 `ncclGroupStart()` 包住多个 API**：你手动 `Start` 把深度从 0→1；这之后每调一个 API，`ncclEnqueueCheck` 内部的 Start/End 把深度在 1↔2 之间来回拨，**始终没归零**，所以每个 API 结束都不触发发射，只记账；直到你手动 `ncclGroupEnd()` 把深度从 1→0，才一次性发射这一整批。

换句话说，`ncclEnqueueCheck` 里的隐式 Start/End 对显式 group 是**透明**的：它只是把深度加一又减一，净效果为零，真正的提交时机仍然由你最外层的显式 `GroupEnd` 控制。这样设计的好处是——无论用户关不关 group，API 内部的代码路径是同一条，不用为"单次调用"和"批量调用"维护两套逻辑。

## 关键一步：group 是绑定在线程上的

group 机制最核心、也最容易被忽视的一点是：**"当前 group"是每个线程各自一份的状态。** NCCL 用 `thread_local` 变量来记录它（`src/include/group.h:84-88`）：

```cpp
extern thread_local int ncclGroupDepth;                       // group 嵌套深度
extern thread_local struct ncclComm* ncclGroupCommHead[...];  // 本 group 涉及哪些 comm
extern thread_local int ncclGroupBlocking;                    // 本 group 是否阻塞
```

`ncclGroupStart()` 只是把 `ncclGroupDepth` 加一，并不做任何实质工作。真正的登记是**惰性**的：只有当某个 comm 的 API 第一次被调到时，这个 comm 才被"拉进"当前线程的 group。这个拉的动作就是后面要讲的 `ncclGroupCommJoin`。

一句话：**group 不属于某个 comm，也不属于某个进程，它属于"某个线程当前正在进行的这批操作"。** 每个线程有自己的 group 上下文，互不可见。

## 一个 group 可以挂多个 comm，每个 comm 自带一个 planner

一个 group 里可以涉及多个 comm——比如对 comm1、comm2 分别发操作。这些 comm 被串成一条链表（`ncclGroupCommHead`），由当前线程持有。后面会讲这条链表怎么维护，这里先看每个 comm 自己长什么样。

**每个 comm 有一个独立的 planner**（`comm->planner`），planner 里挂着各类任务队列，API 按类型分流进去：

```text
planner->peers[peer].sendQueue / recvQueue   ← 点对点 P2P（ncclSend / ncclRecv）
planner->peers[peer].bcastQueue              ← Broadcast
planner->collSorter                          ← 普通集合（AllReduce 等，按数据量分桶）
planner->rmaTaskQueues                       ← 单边 RMA（PutSignal / WaitSignal）
```

以 `ncclSend` 为例，它的入队路径是：

```text
ncclSend(comm, sendbuff, count, dt, peer, stream)
  └ ncclEnqueueCheck(info)                      src/enqueue.cc:3016
     ├ ncclCommEnsureReady                       确认 comm 可用
     └ taskAppend → p2pTaskAppend               src/enqueue.cc:2495
        ├ ncclGroupCommJoin(comm, collective)    把 comm 登记进当前 group  ★
        ├ ncclMemoryPoolAlloc(ncclTaskP2p)       从 comm->memScoped 切一块内存
        └ 挂进 planner->peers[peer].sendQueue
     └ ncclGroupEndInternal → depth 归零时触发发射
```

注意打 ★ 的那行 `ncclGroupCommJoin`——它是线程与 comm 建立所属关系的关键动作，也是本文的重头戏。在此之前先记住一句话：**入队阶段全程是纯内存记账，没有 CUDA 调用、没有网络通信。** 所有 comm 的任务都安静地躺在各自的 planner 队列里，等 group 结束时统一发射。

## join：线程把一个 comm 纳入自己的 group

`ncclGroupCommJoin` 是"comm 进入 group"的入口，`ncclGroupCommLeave` 是出口。join 做的事可以用一句话概括：**把这个 comm 挂进本线程的 group 链表，并为它准备好本轮要用的内存和 planner。**

```cpp
inline void ncclGroupCommJoin(struct ncclComm* comm, int type) {
  if (comm->groupNext[type] == reinterpret_cast<struct ncclComm*>(0x1)) {
    // 仅当 comm 当前是空闲的（哨兵 0x1）才需要做实质初始化：
    //   1. 把 comm 插进 thread_local 链表 ncclGroupCommHead[type]
    //   2. 给 comm->memScoped 推一层栈帧，本轮任务从这里分配
    //   3. 清零 comm->planner（保留 peers 数组本体），从干净状态开始规划
  }
  ncclGroupBlocking = comm->config.blocking;   // 这两行每次都执行
}
```

下面把三件事拆开讲。

### 第一件：把 comm 插进本线程的 group 链表

这是最核心的动作。本线程的 group 用一条单链表记录涉及的所有 comm，`groupNext[type]` 是链表的 next 指针：

```text
ncclGroupCommHead[type]  →  commA  →  commB  →  ...  →  nullptr
```

插入不是简单的 push_back，而有顺序要求（`src/include/group.h:112-122`）：

```cpp
struct ncclComm** pp = &ncclGroupCommHead[type];
while (*pp != nullptr && comm->intraComm0 != (*pp)->intraComm0)   // 先找同一个 clique
  pp = &(*pp)->groupNext[type];
if (*pp == nullptr) {                                              // 没找到同 clique，按 commHash 升序
  pp = &ncclGroupCommHead[type];
  while (*pp != nullptr && (*pp)->commHash < comm->commHash) pp = &(*pp)->groupNext[type];
}
comm->groupNext[type] = *pp;   // 链入
*pp = comm;
```

两条规则：

1. **同一个 clique 的 comm（`intraComm0` 相同，即同一进程实体下的兄弟 comm）必须排在一起。** 先沿链表找有没有同 clique 的，有就插在它旁边。
2. **不同 clique 之间按 `commHash` 升序排列。**

为什么要这么讲究顺序？因为 group 结束发射时，`doLaunches` 要按 clique 分批处理——它靠"相邻的 comm 是否同 clique"来切分批次。如果同 clique 的 comm 不挨着，批次就切不对。这个顺序约束是给后面的发射阶段铺路。

### 第二件：给这个 comm 推一层内存栈帧

`ncclMemoryStackPush(&comm->memScoped)` 给 comm 的内存栈推一层新的"栈帧"。这个 group 里给该 comm 分配的所有临时对象——任务节点（`ncclTaskP2p`/`ncclTaskColl`）、kernel plan 等——都从这层栈帧里切。

这就是为什么 join 必须发生在 `taskAppend` 分配任务**之前**：没有这层栈帧，`ncclMemoryPoolAlloc` 无处可切。而用栈帧的好处是，group 结束时一次 `Pop` 就能把整批任务内存全部回收，不必逐个 free。

### 第三件：清零 planner，从干净状态开始

对于集合类型的 group，join 会 `memset(&comm->planner, 0, ...)` 把本轮的规划状态（任务计数、各种队列头）清零。注意 `planner.peers` 数组本体是长期分配的，代码先把它暂存、清零后再还原，避免每个 group 都重新 malloc。每个 group 是一轮独立的规划，必须从干净状态开始。

## 哨兵 `0x1`：comm 怎么知道自己"在不在 group 里"

join 开头那个 `if (comm->groupNext[type] == 0x1)` 用了一个哨兵值 `0x1` 来判断 comm 当前是否空闲。它的状态语义是：

| 状态 | `groupNext[type]` 的值 | 含义 |
|---|---|---|
| 不在任何 group（空闲） | `0x1` | 下次 join 要做初始化 |
| 已加入某个 group | 真实链表指针或 `nullptr`（链表尾） | 已登记，下次 join 跳过初始化 |

容易记反的一点：**`0x1` 表示"不在 group 里"，不是"在 group 里"。** comm 空闲时是 `0x1`，被 join 进 group 后变成链表指针，被 leave 摘出后又变回 `0x1`。

### 它是"幂等守卫"，不是互斥锁

这个哨兵最常见的误读，是以为它能防止 comm 同时进入两个 group。**它不能。** 看那个 `if`：它只有"做初始化 / 跳过初始化"两个分支，**没有"拒绝进入"的分支**。如果 `groupNext[type]` 不是 `0x1`（说明已被登记过），join 不会报错退出，而是直接跳过初始化、继续往下追加任务。

哨兵 `0x1` 真正的作用是 **幂等**：保证同一个 comm 在**同一个 group 内**被多次用到时（比如 group 里对同一个 comm 连发两个 `ncclSend`）只初始化一次——第二次 join 看到 `groupNext` 不是 `0x1`，跳过插链表和清零，只更新一下 `ncclGroupBlocking`。它是个去重开关，不是个锁。

这个设计之所以安全，是**因为它假定"读到的非 `0x1` 一定是本线程自己上一轮留下的"**。只要每个 comm 自始至终只被同一个线程碰，这个假定就成立，哨兵就够用。一旦跨线程共享 comm，这个假定就被击穿——这正是下一节要讲的失败场景的根源。

## group 结束：遍历链表发射，然后清空

group 结束时，`doLaunches`（`src/group.cc:307`）消费这条线程维护的 comm 链表，把各 comm planner 上攒的任务一次性发射。它按 clique 分批，每个 clique 内逐个 comm 处理：

```text
doLaunches(head):
  cliqueHead = head
  do:                                      // 遍历每个 clique
    comm = cliqueHead
    do:                                    // 遍历 clique 内各 comm
      cudaSetDevice(comm->cudaDev)
      ncclLaunchPrepare(comm)              // 把 planner 任务排进 kernel plan
      (group-launch 模式) clique 内对齐
      comm = comm->groupNext[Collective]
    while (comm != nullptr && 同 clique)
    // 多轮 launch：各 comm 同序号 plan 交错发射
    // 最终轮：ncclLaunchFinish(comm)
    cliqueHead = 下一个 clique
  while (cliqueHead != nullptr)
```

这里就能看出 join 时"同 clique 相邻"的用意了：`while (comm->intraComm0 == cliqueHead->intraComm0)` 之所以能正确圈出一个 clique，正是因为 join 把同 clique 的 comm 插到了一起。

发射完成后，清理阶段会调 `ncclGroupCommLeave` 遍历整条链表，对每个 comm 做两件事：把 `groupNext[type]` 改回 `0x1`（标记"已退出，下次 join 要重新初始化"），`Pop` 掉本轮栈帧（整批任务内存一次性回收）。最后链表头被置空，线程回到"无 group"状态。一进一出，干干净净。

把整个模型画出来，是这样一张内存视图：

```text
线程 T 的 group 上下文（thread_local，别的线程看不到）
┌──────────────────────────────────────────────────┐
│  ncclGroupDepth / ncclGroupBlocking              │
│  ncclGroupCommHead[Collective] ─┐                │
└─────────────────────────────────┼────────────────┘
                                  ▼
   commA ──groupNext──▶ commB ──groupNext──▶ nullptr   （按 clique + commHash 排好序）
   (intraComm0=X)         (intraComm0=X)               ← 同 clique 相邻
        │                      │
        ▼                      ▼
   commA.planner          commB.planner      ← 每个 comm 独立，任务挂在各自队列里
   commA.memScoped[top]   commB.memScoped[top] ← join 时 push 的栈帧，leave 时 pop 掉
```

一句话收束：**线程通过 `ncclGroupCommHead` 这条 thread_local 链表"拥有"本轮 group 涉及的所有 comm。** join 时按顺序把它们链入、为每个备好内存与 planner；leave 时整链清空、各 comm 复位。comm 上的 `groupNext` 哨兵只是这条所属关系的本地标记——它的安全性，完全依赖"这个 comm 自始至终只被这一个线程碰"。

## 两条不可妥协的硬约束

讲完了正常流程，现在可以理解 NCCL 定下的两条使用契约了。它们不是某段代码强制执行的运行时检查，而是**设计前提**——NCCL 的内部状态（group 链表、planner、内存栈、哨兵）既没有加锁也没有并发检测，全靠使用者主动遵守。

### 约束一：一个 communicator 任意时刻只能被一个线程驱动

NCCL 的 communicator **不是线程安全的**。需要多线程并行通信时，正确做法是**为每个线程创建各自的 communicator**（按 rank 拆分或建多个 comm），让它们各自独立走 group 流程，互不触碰对方的 comm。NCCL 的多 stream / 多 comm 并行就建立在这种"每线程一个 comm"的模式上。

> **一个 communicator 在任意时刻只能被一个线程驱动。** 不能跨线程共享同一个 comm——尤其不能在一个线程还有未关闭的 group 时，让另一个线程交叉使用同一个 comm。

NCCL 不会为违反这条契约的行为给出清晰报错。如果你跨线程共享了 comm，得到的是**静默的错误**。推演一下这个失败过程就能理解为什么：

假设线程 A 在显式 group 里用了 `comm`（此时 `groupNext` 已不是 `0x1`，planner 上有 A 的任务，`memScoped` 顶着 A 推的栈帧），同时线程 B 用同一个 `comm` 发 `ncclSend`：

```text
B: ncclGroupCommJoin(comm, collective)
     groupNext != 0x1  → 跳过初始化（不插 B 的链表、不 push 栈帧、不清 planner）
     无任何报错，直接返回
   ncclMemoryPoolAlloc(ncclTaskP2p)        ← 照常执行，切的是 A 正在用的栈顶
   挂进 comm->planner.peers[peer].sendQueue ← 和 A 的任务串进了同一条队列
B: ncclGroupEndInternal → 触发 B 的 doLaunches
     遍历 B 自己的 ncclGroupCommHead —— 里面根本没有 comm（B 从没插进去过）
     → B 的任务既不被 B 发射，也不被 B 清理，成了孤儿
```

三个后果层层叠加：

1. **全程没有拒绝**：B 一路畅通地把任务挂进共享的 planner 和内存栈，没有任何检查拦它。
2. **B 的任务成了孤儿**：B 的 group 结束时不会遍历到这个 comm，B 提交的任务根本不会被发射。
3. **与 A 发生数据竞争**：那个孤儿任务还挂在 `comm->planner.sendQueue` 上。等 A 的 group 结束、A 的 `doLaunches` 遍历这个 comm 时，会**把 B 的任务也一起发射**——用 A 的上下文去发 B 提交的任务。与此同时 A、B 还在并发清零 planner、并发切内存栈顶，这是教科书级的数据竞争。

表现到外面，就是通信莫名挂起、数据莫名其妙错乱、或者难以复现的崩溃。这就是为什么这条契约不可妥协——NCCL 不为这种用法做防御性检查，违反它得到的是静默错误，不是清晰报错。

### 约束二：非阻塞模式下，同一个 comm 的下一个 group 要等上一个做完

这条是约束一的自然延伸，针对的是非阻塞（non-blocking）模式。

开启非阻塞模式（`config.blocking == 0`）时，`ncclGroupEnd` 不会原地等通信完成，而是：给每个参与 comm 标记 `ncclInProgress`、起一个后台线程去真正跑连接和发射、然后立刻返回 `ncclInProgress`。用户之后靠 `ncclCommGetAsyncError` 轮询，等它变回 `ncclSuccess`。

这里就出现了约束一的变体：**前一个 group 的后台线程还在跑，相当于另一个"执行流"占着这个 comm。** 此时当前线程不能再往这个 comm 提交新 group，否则两边又会争抢同一个 planner 和内存栈。

> **在非阻塞模式下，对同一个 communicator，提交下一个 group 之前，必须确保上一个非阻塞 group 已经完成**（`ncclCommGetAsyncError` 返回 `ncclSuccess`）。

这个约束由 `ncclCommEnsureReady`（`src/init.cc:419`）在每个新 API 进入时强制检查：

```cpp
NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
if (ret == ncclInProgress) {
  WARN("Attempt to use communicator before the previous operation returned ncclSuccess");
  ret = ncclInvalidArgument;   // 占线 → 翻译成用法错误
  goto exit;
}
/* 其他错误值原样透传 */
```

为什么把 `ncclInProgress` 翻译成 `ncclInvalidArgument` 而不是原样返回？因为对一个还在跑的 comm 发新操作是**用法错误**；如果原样返回 `ncclInProgress`，用户会误以为"这次新操作也被异步提交了"，反而是更大的误导。

三个容易混的点，顺手澄清：

- **同一个 group 里塞多个操作是允许的**，这正是 group 的用途。约束说的是"一个非阻塞 group 的**后台执行期**"不能和"下一个 group 的**提交期**"重叠。
- **多个不同的 communicator 可以各自同时跑非阻塞 group**，各自的 `asyncResult` 和后台线程互不冲突——这是多 stream / 多 comm 并行的基础。
- **阻塞 group 不存在这个问题**：`ncclGroupEnd` 原地等通信完成才返回，返回时 comm 已经空闲，下一次直接放行。

两条约束其实是同一原则：**一个 communicator 在任意时刻只能被一个执行流驱动。** 阻塞模式下，执行流就是当前线程串行调用；非阻塞模式下，前一个 group 的后台线程也算一个执行流，它没退场前，当前线程不能再插队。

## 小结

1. group 是一次"批量提交"：API 调用只记账，真正的发射在 group 结束时。
2. group 是**绑定在线程上**的——`thread_local` 状态，每个线程有自己的 group 上下文，互不可见。
3. 一个 group 可以挂多个 comm；每个 comm 自带一个 planner，任务按类型进各自的队列。
4. 线程靠 `ncclGroupCommHead` 这条链表"拥有"本轮的 comm：join 时按 clique 相邻 + commHash 升序把 comm 链入，并为每个 comm 推一层内存栈帧、清零 planner；group 结束时 `doLaunches` 遍历链表发射，leave 清空整链、各 comm 复位。
5. 哨兵 `0x1` 是"comm 不在 group 里"的标记，作用是让同 group 内重复 join 只初始化一次（幂等），**它不是互斥锁**——它假定非 `0x1` 一定是本线程自己留下的。
6. 两条硬约束：① 一个 comm 任意时刻只能被一个线程驱动；② 非阻塞模式下同一个 comm 的下一个 group 要等上一个完成。本质都是"一个 comm 同时只能被一个执行流驱动"，违反会静默出错。

## 源码位置速查

| 主题 | 位置 |
|---|---|
| group 的 thread_local 状态、join/leave、哨兵 `0x1` | `src/include/group.h:84-157` |
| `ncclGroupEndInternal`、非阻塞 group 起 `ncclInProgress` | `src/group.cc` |
| `doLaunches` 按 clique 分轮发射 | `src/group.cc:307` |
| `groupCleanup` / `groupLocalResetJobState`（leave 与链表清空） | `src/group.cc:381, 390` |
| `ncclEnqueueCheck` 入队总控 | `src/enqueue.cc:3016` |
| `p2pTaskAppend` 点对点入队 | `src/enqueue.cc:2495` |
| `ncclCommEnsureReady`（非阻塞串行检查） | `src/init.cc:419` |
| `ncclCommGetAsyncError` | `src/init.cc:3247` |
| comm 结构、`groupNext`、`planner`、`memScoped` | `src/include/comm.h` |

进一步可读 [Communicator 结构与字段地图](communicator) 看 `ncclComm` 的完整字段布局，或 [源码分析：AllReduce 调用链](allreduce-flow) 看从 API 到 GPU kernel 的逐函数追踪。
