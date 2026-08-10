---
title: ncclIntruQueue：侵入式单链表队列
description: 用成员指针模板把 next 字段交给调用方的侵入式 FIFO；O(1) 入队/出队/批量拼接，以及 MPSC 变体的无锁生产。
weight: 3
---

前两篇讲的 [`ncclMemoryStack`](memory-stack) 管分配、[`ncclMemoryPool`](memory-pool) 管回收，但要真的把任务"排起队来"交给调度器处理，还需要一个容器把它们串起来。`ncclIntruQueue` 就是这个容器——一个**侵入式（intrusive）的单链表 FIFO 队列**。

所谓"侵入式"，是指队列本身不拥有节点内存，节点也不内嵌队列元数据；队列只存 `head`/`tail` 两个指针，而把"指向下一个节点"的 `next` 指针**留在节点自己的结构体里**，由调用方通过模板参数告诉队列"用哪个字段当 next"。这跟 `std::queue` 那种"容器包节点"的模式正好相反，好处是零额外分配、零拷贝、节点可以同时挂在多条队列上。源码内联在 `src/include/utils.h:368-459`，声明和注释在 `:160-191`。

> 相关阅读：任务节点从 `ncclMemoryPool` 来，挂进 `ncclIntruQueue` 排队，group 结束再 Free 回池——三个结构串成一条"分配→排队→回收"的链。

## 整体模型：head、tail，和一个外置的 next

struct 极简（`utils.h:369-372`）：

```cpp
template<typename T, T *T::*next>
struct ncclIntruQueue {
  T *head, *tail;
};
```

就两个指针。`head` 指向队首（下次 Dequeue 取出的那个），`tail` 指向队尾（最近一次 Enqueue 挂上去的那个）。注意模板第二个参数 `T *T::*next`——这是个**成员指针（pointer-to-member）**，指向"T 里的某个 `T*` 字段"。队列所有操作都用 `x->*next` 这种语法访问节点的 next 字段，而 next 字段叫什么名字、在 T 里什么位置，完全由调用方在实例化时决定。

头部注释给了一个一图到位的例子（`utils.h:161-169`）：

```cpp
struct Foo {
  struct Foo *next1, *next2; // 可以同时是两条链的成员
};
ncclIntruQueue<Foo, &Foo::next1> list1;
ncclIntruQueue<Foo, &Foo::next2> list2;
```

同一个 `Foo` 对象，用 `next1` 挂在 `list1` 上、用 `next2` 挂在 `list2` 上——两条队列独立遍历，互不干扰。这就是侵入式队列相比容器式队列最大的灵活性：**一个节点能同时身兼多职**，只要它备了多个 next 字段。NCCL 真用上了这一点：`ncclProxyOp` 有个专门的 `enqNext` 字段（区别于通用 `next`），这样同一个 proxy op 既能挂在某条通用链上、又能挂进 `proxyOpQueue`，互不干扰。

链表形状是经典的"单链表 + tail 指针"：

```
ncclIntruQueue
 │
 ├─ head ──► [T|next] ──► [T|next] ──► ... ──► [T|next=null]  ◄── tail
 │             ▲                                           ▲
 │             │ Enqueue 从 tail 进                        │ EnqueueFront 也插这里(变新head)
 │             └─ Dequeue 从 head 出
 │
 └─ (next 字段在每个 T 节点自己体内,队列不持有它)
```

`tail` 的唯一作用是让 Enqueue 不必遍历找尾——O(1) 入队。没有 tail 的话，单链表入队要走到链表末尾，退化成 O(n)。`head` 同时服务于 Dequeue（摘头）和 Empty 判断（`head == nullptr`）。

## 各操作的语义

### Enqueue：尾进

```cpp
template<typename T, T *T::*next>
inline void ncclIntruQueueEnqueue(ncclIntruQueue<T,next> *me, T *x) {
  x->*next = nullptr;                                   // 新节点 next 清空
  (me->head ? me->tail->*next : me->head) = x;          // 空队:设 head;非空:接在 tail 后
  me->tail = x;                                         // tail 指向新节点
}
```

第二行是个紧凑写法，用三元表达式同时覆盖两种情况：队列空时新节点成为 `head`（`tail` 此时是悬空的，不能写 `tail->*next`），非空时把新节点接到 `tail->*next`。然后 `tail` 总是更新成新节点。节点必须已经分配好（典型来自 `ncclMemoryPoolAlloc`），Enqueue 只动指针。

### EnqueueFront：头插

```cpp
template<typename T, T *T::*next>
inline void ncclIntruQueueEnqueueFront(ncclIntruQueue<T,next> *me, T *x) {
  if (me->head == nullptr) me->tail = x;                // 空队时也要设 tail
  x->*next = me->head;                                  // 新节点指向原 head
  me->head = x;                                         // 新节点变 head
}
```

插到队首而不是队尾，相当于把队列当栈用。空队时要额外维护 `tail`（否则 tail 还是悬空）。这在需要"插队"或"倒序处理"时有用。

### Dequeue：头出

```cpp
template<typename T, T *T::*next>
inline T* ncclIntruQueueDequeue(ncclIntruQueue<T,next> *me) {
  T *ans = me->head;
  me->head = ans->*next;                                // head 后移
  if (me->head == nullptr) me->tail = nullptr;          // 队空:tail 也清空
  return ans;
}
```

摘头、head 后移。关键是最后一行：如果摘完队空了，`tail` 必须清空成 nullptr。否则 `tail` 会指向一个已经出队的节点，下次 Enqueue 时 `tail->*next = x` 就写进了已出队节点的 next 字段——虽然那块内存可能还活着（在 pool 里），但队列逻辑就断了。**Dequeue 假设队列非空**——它不检查 `head == nullptr`，对空队 Dequeue 会解空指针。所以调用前一般先 `ncclIntruQueueEmpty` 判断，或者用下面的 `TryDequeue`。

### TryDequeue：安全的头出

```cpp
template<typename T, T *T::*next>
inline T* ncclIntruQueueTryDequeue(ncclIntruQueue<T,next> *me) {
  T *ans = me->head;
  if (ans != nullptr) {                                 // 先判空
    me->head = ans->*next;
    if (me->head == nullptr) me->tail = nullptr;
  }
  return ans;                                           // 空则返回 nullptr
}
```

和 Dequeue 逻辑一样，只是多了空检查，空队时返回 nullptr 而非崩溃。NCCL 里大量 `while (!ncclIntruQueueEmpty(...)) { x = ncclIntruQueueDequeue(...); }` 的循环其实可以用 TryDequeue 写得更紧凑，但代码里两种风格都有。

### Delete：按比较函数删中间节点

```cpp
template<typename T, T *T::*next>
inline T* ncclIntruQueueDelete(ncclIntruQueue<T,next> *me, T *x, bool (*cmp)(T*, T*)) {
  T *prev = nullptr;
  T *cur = me->head;
  bool found = false;
  while (cur) {
    if (cmp(cur, x)) { found = true; break; }           // 用 cmp 找目标
    prev = cur;
    cur = cur->*next;
  }
  if (found) {
    if (prev == nullptr) me->head = cur->*next;         // 删的是 head
    else prev->*next = cur->*next;                      // 删中间:前驱跳过它
    if (cur == me->tail) me->tail = prev;               // 删的是 tail:tail 前移
  }
  return cur;                                           // 没找到返回 nullptr
}
```

这是唯一一个 O(n) 操作。FIFO 本身不擅长中间删除，但偶尔需要（比如从连接的 `proxyMemHandleQueue` 里摘掉某个已失效的内存句柄）。`cmp` 是调用方提供的比较函数，找到第一个 `cmp(cur, x)` 为真的节点摘掉。三种位置（head/中间/tail）都要正确维护指针，尤其删 tail 时 `tail` 要回退到前驱——这也是 `tail` 指针维护里最容易漏的一种情况。

### Transfer：整条拼接（O(1)）

```cpp
template<typename T, T *T::*next>
void ncclIntruQueueTransfer(ncclIntruQueue<T,next> *dst, ncclIntruQueue<T,next> *src) {
  (dst->tail ? dst->tail->next : dst->head) = src->head;  // src 整条接到 dst 尾
  if (src->tail) dst->tail = src->tail;                   // dst 的新 tail 是 src 的 tail
  src->head = nullptr;                                    // src 清空
  src->tail = nullptr;
}
```

把 `src` 整条链表搬到 `dst` 尾部，**O(1)**，只动头尾指针，不遍历。这是个很实用的批量操作：调度时常常先把任务按某种属性分进几个临时队列（`collBins[isCollnet][isNvls]`），分完再把这几条队列依次 Transfer 拼成最终的总队列（`enqueue.cc:474`）。比逐个 Enqueue 高效得多——前者 O(总节点数)，后者 O(队列数)。

注意这里写的是 `dst->tail->next` 而不是 `dst->tail->*next`——源码里这行用的是裸 `.next` 字段名而不是成员指针语法。这是个实现细节：因为 `Transfer` 是模板、`T` 的具体类型已知，直接写 `next` 字段能编译通过（成员指针 `next` 指向的就是这个字段）。不过严格说这种写法依赖 `next` 这个成员指针恰好指向名为 `next` 的字段，对 `&Foo::next1` 这类实例就不通用了——算是个实现上偏实用的妥协。

## 为什么用侵入式：三个结构性的好处

把 next 字段放在节点自己体内、而不是队列容器里，带来三个好处，正好对应 NCCL 的三种用法。

**节点零额外分配。** `std::queue` 每入队一个元素要 new 一个容器节点把元素包起来，出队再 delete——这在热路径上不可接受。侵入式队列的节点就是任务对象本身，入队/出队只动指针，零分配。任务对象从 `ncclMemoryPool` 来，本来就已经分配好了，直接挂链即可。

**一个节点可挂多链。** 用不同的 next 字段，同一个对象能同时身在多条队列。`ncclProxyOp` 用 `enqNext` 挂进 `proxyOpQueue`、用通用 `next` 挂别处；`ncclTaskRma` 可以按上下文挂进 `rmaTaskQueueCe` 或 `rmaTaskQueueProxy`。容器式队列做不到这点——一个对象被一个容器拥有后没法同时进另一个。

**和内存池解耦。** 队列不关心节点从哪来、到哪去，只管串指针。节点从 pool 取、用完还 pool，队列只是一个临时视图。这跟 memoryPool 的"回收缓存"定位天然契合：pool 管生死，queue 管顺序。

## 在 NCCL 里的实际用法

`ncclIntruQueue` 是 NCCL 里用得最广的容器结构，通信器和 planner 上挂了几十条。典型几类：

**planner 的任务分桶队列**（`comm.h:341-346`，挂在 `ncclKernelPlan` 上）。每条 plan 有六条任务队列，按类型分开：

```cpp
struct ncclIntruQueue<struct ncclTaskP2p,  &ncclTaskP2p::next>  p2pTaskQueue;
struct ncclIntruQueue<struct ncclTaskBcast, &ncclTaskBcast::next> bcastTaskQueue;
struct ncclIntruQueue<struct ncclTaskColl,  &ncclTaskColl::next>  collTaskQueue;
struct ncclIntruQueue<struct ncclTaskRma,   &ncclTaskRma::next>   rmaTaskQueueProxy;
struct ncclIntruQueue<struct ncclTaskRma,   &ncclTaskRma::next>   rmaTaskQueueCe;
struct ncclIntruQueue<struct ncclProxyOp,   &ncclProxyOp::enqNext> proxyOpQueue;
```

注意 `proxyOpQueue` 用的是 `&ncclProxyOp::enqNext`——专用的入队 next 字段，因为 proxy op 可能还要挂在别处。入队例如 `enqueue.cc:273` 的 `ncclIntruQueueEnqueue(&plan->proxyOpQueue, op)`，group 清理时 `group.cc:416` 的 `ncclIntruQueueDequeue(&plan->proxyOpQueue)` 逐个取出 Free 回池。

**per-peer 的收发队列**（`comm.h:434-438`，planner 的 `Peer` 结构）。每个 rank 一个 Peer，各有 send/recv/bcast 三条队列，攒该 peer 的 p2p 任务：

```cpp
struct Peer {
  bool sendSeen, recvSeen;
  struct ncclIntruQueue<struct ncclTaskP2p,  &ncclTaskP2p::next>  sendQueue;
  struct ncclIntruQueue<struct ncclTaskP2p,  &ncclTaskP2p::next>  recvQueue;
  struct ncclIntruQueue<struct ncclTaskBcast, &ncclTaskBcast::next> bcastQueue;
};
```

**分桶 + Transfer 拼接**（`enqueue.cc:463-476`）。调度 collective 任务时，先按 `(isCollnet, isNvls)` 四种组合分进四个临时 `collBins` 队列，分完再依次 Transfer 进最终的 `collTaskQueue`：

```cpp
ncclIntruQueueEnqueue(&collBins[isCollnet][isNvls], aggBeg);   // 分桶
...
for (int isCollnet=0; isCollnet<=1; isCollnet++)
  for (int isNvls=0; isNvls<=1; isNvls++)
    ncclIntruQueueTransfer(&planner->collTaskQueue, &collBins[isCollnet][isNvls]);  // 整条拼回
```

这是 Transfer 的典型场景：先分散归类、再 O(1) 批量合并。

**跨线程的 callback 队列用的是 MPSC 变体**（见下一节）。

## MPSC 变体：无锁多生产者单消费者

`ncclIntruQueue` 是单线程的——没有同步。但 NCCL 有个跨线程场景：proxy 线程、device 回收线程等需要往通信器主线程的队列里塞回调（`ncclCommCallback`），多个线程同时入队、主线程单线程消费。这就是 **MPSC（Multi-Producer Single-Consumer）** 场景，`ncclIntruQueueMpsc`（`utils.h:463-571`）为此而生。

struct 比 single-thread 版复杂（`utils.h:463-468`）：

```cpp
template<typename T, T *T::*next>
struct ncclIntruQueueMpsc {
  T* head;
  uintptr_t tail;                 // 注意是 uintptr_t,不是 T*
  struct ncclThreadSignal* waiting;
};
```

`tail` 被故意存成 `uintptr_t` 而不是 `T*`，是为了能塞进几个特殊的"哨兵"值来表达队列状态。这套实现用了三个哨兵：

- `0x0`：正常的"活跃、非空/空但有人在管"状态基线。
- `0x1`：消费者正在 `waiting`（睡在条件变量上）。
- `0x2`：队列被 `abandon`（消费者主动放弃，生产者入队会失败/需自行处理）。
- 普通 `>0x2` 的值：`tail` 指向最后一个入队节点的真实指针。

入队（`utils.h:483-500`）的核心是无锁的：先把新节点 `x->next` 置空，再用原子 `exchange` 把 `tail` 换成 `x`——这个 exchange 是多生产者安全的，多个线程同时入队时 exchange 保证只有一个拿到旧 tail、其余拿到彼此刚塞进去的节点，天然串成链。拿到旧 `utail` 后，根据它是否 ≤0x2（哨兵）决定把 `x` 挂到 `me->head` 还是 `prev->next`。如果旧值是 `0x1`（消费者在睡），还要通过 `ncclThreadSignal` 唤醒它——这里那段看似多余的 `unique_lock lock(waiting->mutex); }`（立即析构解锁）是刻意的内存屏障，确保生产者不会抢在消费者进入 wait 之前发 notify，避免丢失唤醒。

消费（`DequeueAll`，`utils.h:503-540`）是一次性取出整条链：先 load `head`，空的话按 `waitSome` 决定自旋 10µs 再睡条件变量。拿到非空 head 后，把 head 清空、用 exchange 把 tail 归零取回尾节点，然后沿 `next` 链从 head 走到 tail——注意因为无锁，生产者的链可能还没完全接好，中间某节 `next` 可能暂时是 null，所以消费侧有自旋等待 `x1 != nullptr` 的循环（`spins==1024` 时 `yield` 让出 CPU）。

这套 MPSC 的精妙之处在于：**入队完全无锁（靠原子 exchange 串链）、出队批量（一次取整条，摊薄同步代价）、用 tail 哨兵值编码状态避免额外锁**。代价是出队时要 spin 等待生产者把链接好，以及实现复杂度远高于 single-thread 版。NCCL 只在最热的跨线程回调场景用它——`ncclComm::callbackQueue`（`comm.h:731`），proxy/回收线程 `ncclIntruQueueMpscEnqueue`（`enqueue.cc:1403/1487`），主线程 `ncclCommPollCallbacks` 里 `ncclIntruQueueMpscDequeueAll`（`comm.h:808`）批量取出执行回调。

## 三件容器放一起：stack / pool / queue

把这三篇连起来看，NCCL 热路径的内存与任务管理是一条清晰的链：

`ncclMemoryStack` 是**地基**——真正持有大块 hunk、做 bump 分配，`memScoped` 给 per-group 临时对象、`memPermanent` 给常驻对象。`ncclMemoryPool` 是**回收层**——在 `memPermanent` 之上为每种任务类型维护空闲链表，让单个对象能跨 group 复用，稳态零分配。`ncclIntruQueue` 是**组织层**——把从 pool 取出的任务节点串成 FIFO，交给调度器分桶、Transfer、消费。

三者的接合点是个干净的三步式生命周期：`ncclMemoryPoolAlloc<T>` 从池（必要时从后端 stack）要一个 `T*` → 填好字段 → `ncclIntruQueueEnqueue` 挂进某条队列 → 调度器 `Dequeue` 取出处理 → `ncclMemoryPoolFree` 还回池。内存的"生死"归 pool/stack 管，任务的"先后"归 queue 管，职责不交叉。这套分工让 NCCL 在每个 group 几十万次任务对象流转里几乎不碰系统 malloc/free，热路径只剩指针操作。

## 设计要点回顾

`ncclIntruQueue` 用成员指针模板把 next 字段外置给调用方，自己只存 head/tail 两个指针，是零分配的侵入式单链表 FIFO。head 出 tail 进，O(1) 入队出队；tail 的存在让入队不必遍历，也让 Transfer 能 O(1) 整条拼接。侵入式带来三个好处：节点零额外分配、一个节点可用多个 next 字段同时挂多链、和内存池/栈彻底解耦。Dequeue 假设非空、Delete 是唯一 O(n) 操作，其余都是 O(1)。MPSC 变体用 `tail` 存 uintptr_t 哨兵值编码队列状态，靠原子 exchange 无锁串链、批量出队、条件变量唤醒，服务跨线程回调这一种场景。

## 源码索引

- 头部注释与前置声明：`src/include/utils.h:160-191`
- 结构体定义：`src/include/utils.h:369-372`
- Construct / Empty / Head / Tail：`src/include/utils.h:375-393`
- Enqueue / EnqueueFront：`src/include/utils.h:396-407`
- Dequeue / Delete / TryDequeue：`src/include/utils.h:410-451`
- Transfer（O(1) 拼接）：`src/include/utils.h:454-459`
- MPSC 变体结构体：`src/include/utils.h:463-468`
- MPSC Enqueue / DequeueAll / Abandon：`src/include/utils.h:483-571`
- planner 上的六条任务队列：`src/include/comm.h:341-346`
- per-peer 收发队列：`src/include/comm.h:434-438`
- 跨线程 callback 队列（MPSC 用法）：`src/include/comm.h:731, 808`、`src/enqueue.cc:1403`
