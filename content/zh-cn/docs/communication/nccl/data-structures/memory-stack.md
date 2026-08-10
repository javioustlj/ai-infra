---
title: ncclMemoryStack：LIFO 帧级内存池
description: NCCL 热路径临时内存池的设计动机、frame/hunk/unhunk 三层结构与 Push/Pop 成片回收机制，附容易卡住的关键点。
weight: 1
---

NCCL 在热路径上要频繁地分配大量小块临时内存——通信任务的工作节点、kernel 参数、调度用的数组等等。如果每次都走系统的 `malloc`/`free`，既慢又会产生碎片。`ncclMemoryStack` 就是为此自实现的一个内存池。

它的核心想法是：**回收不按对象来，按"框架（frame）"来**。一整批对象在同一个 frame 里分配，释放时不需要逐个 `free`，只要把 frame 弹掉，成百上千个对象就一起回收了。这使得分配几乎只是 bump pointer 往前推一下，释放几乎只是一次结构体赋值。

源码分布在两处：结构体定义和内联实现在 `src/include/utils.h:230-315`，少数两个函数的 out-of-line 实现在 `src/misc/utils.cc:260-355`。下面顺着设计思路来讲。

> 相关阅读：`ncclComm` 上的 `memPermanent`/`memScoped` 两个字段就是两个 `ncclMemoryStack`，见 [Communicator 结构与字段地图](../communicator#内存与任务池)；group 里 Push/Pop 的调用点见 [Group 提交模型](../group-comm)。

## 整体模型：栈、frame 和 hunk

把 `ncclMemoryStack` 想象成一个可以压栈弹栈的栈，但栈的每一层不是一个对象，而是一个"工作台"，叫 **frame**。一次 `Push` 推出一个新的空工作台，之后所有的分配都在这个最顶层的工作台上进行；一次 `Pop` 把整个工作台连同它上面所有东西一起撤掉，回到上一个工作台。这就是头部注释里说的 "frames containing many objects are pushed and popped"——LIFO 的粒度是 frame，不是对象。

每个 frame 下面真正放内存的地方叫 **hunk**：一块通过 `malloc` 拿到的大块连续内存。frame 内部维护两个指针 `bumper` 和 `end`，都指向当前 hunk 内部——`bumper` 是已用到哪儿，`end` 是这块 hunk 的尽头。分配就是把 `bumper` 对齐一下、往前推 `size`，只要没超过 `end` 就成功了。这种 bump allocator 是最快的分配方式之一：没有空闲链表、没有锁、只有一次对齐加法和一次比较。

struct 定义在 `utils.h:230-251`。三个嵌套结构的层次关系大致是这样：

```
ncclMemoryStack
│
├─ stub (Hunk)            ← 占位假 hunk，nil frame 挂它，无真实内存
│
└─ topFrame (Frame)       ← 栈顶工作台，所有 Alloc 都在它里面推进
     │
     │  bumper ──┐  都指向 ↓ 当前 hunk 内部
     │  end ─────┘        bumper 是"下一处可分配"，end 是"这块到头"
     │
     ├─ hunk ───────────► [ Hunk header | obj | obj | obj |  (空)  ... ]
     │                      ▲                  ▲   ▲
     │                      │                  │   └─ bumper ── 前面是已分配，后面到 end 是空闲
     │                      │                  └─ 已分配的小对象（连续紧凑摆放）
     │                      └─ malloc 拿到的大块开头
     │   above ──► (空 hunk 缓存，Pop 时没 free，留着复用) ──► above ──► ...
     │
     ├─ unhunks ──► [ Unhunk:{next,obj} ] ──► [ Unhunk ] ──► nullptr
     │                   │                  (本 frame 的外逃大对象代理链)
     │                   └─ obj ──► malloc 出来的大对象本体
     │
     └─ below ──► [ 上一层 Frame 的快照 ] ──► below ──► ... ──► nil frame (below==nullptr)
```

读这张图的关键：`ncclMemoryStack` 本体几乎不存东西，真正干活的是 `topFrame`。一个 frame 的状态由四样东西完全确定——当前在哪个 `hunk`、`bumper`/`end` 走到哪、有哪些外逃的 `unhunks`、以及上一个 frame（`below`）是谁。`Push` 就是把这四样快照存起来 再开新的，`Pop` 就是把这四样恢复回去。hunk 里 `bumper` 前面是已经分配出去的紧凑对象，从 `bumper` 到 `end` 是还没用的空闲区——分配就是让 `bumper` 往右挪。记住这个画面，后面所有函数都只是在摆弄这张图里的某几根线。

这里有个第三种结构 `Unhunk`，先记住它的存在，到讲慢路径时再说它是干什么用的。struct 定义如下：

```cpp
struct ncclMemoryStack {
  struct Hunk {
    struct Hunk* above;   // 反向栈指针，串起各 hunk
    size_t size;          // 本块总大小（含 header）
  };
  struct Unhunk {         // 大对象外逃时用的代理 header
    struct Unhunk* next;
    void* obj;
  };
  struct Frame {
    struct Hunk* hunk;    // 当前满载进行中的 hunk
    uintptr_t bumper, end;// 指向 top hunk 内部
    struct Unhunk* unhunks; // 本 frame 产生的外逃对象代理链
    struct Frame* below;  // 上一层 frame
  };
  struct Hunk stub;       // nil frame 用的占位假 hunk
  struct Frame topFrame;  // 栈顶 frame
};
```

## 初始状态：nil frame

构造函数在 `utils.h:253-261`，把栈初始化成只有一个 frame 的状态：

```cpp
inline void ncclMemoryStackConstruct(struct ncclMemoryStack* me) {
  me->stub.above = nullptr;
  me->stub.size = 0;
  me->topFrame.hunk = &me->stub;  // 指向没有真实内存的占位 stub
  me->topFrame.bumper = 0;
  me->topFrame.end = 0;           // end==0 ⇒ 任何分配都会溢出
  me->topFrame.unhunks = nullptr;
  me->topFrame.below = nullptr;   // below==nullptr ⇒ 这是 nil frame
}
```

这个 frame 被称为 **nil frame**。它的 `below` 是 `nullptr`，挂的 hunk 是个没有实际内存的 `stub`，`bumper` 和 `end` 都是 0。注释特别交代：**nil frame 不能被弹出**。原因一看 `Pop` 的实现就明白——`Pop` 会做 `topFrame = *topFrame.below`，如果 nil frame 被弹，那就是解引用空指针，直接崩。这是用崩溃来强制保证"至少留一层"。

这层设计还有个语义上的好处：在 nil frame 期间分配的对象，因为 nil frame 永不弹出，它们的生命周期就等于整个栈的生命周期，一直活到 `Destruct`。NCCL 正是用这个特性来放"通信器级常驻"的内存（后面会看到）。

## 分配的快路径

绝大多数分配走的是这条快路径，它是整个设计里最该读明白的一段，在 `utils.h:263-273`：

```cpp
inline void* ncclMemoryStack::allocate(struct ncclMemoryStack* me, size_t size, size_t align) {
  uintptr_t o = (me->topFrame.bumper + align-1) & ~(uintptr_t(align) - 1); // 对齐
  void* obj;
  if (COMPILER_EXPECT(o + size <= me->topFrame.end, true)) {
    me->topFrame.bumper = o + size;     // 推进 bump 指针
    obj = reinterpret_cast<void*>(o);
  } else {
    obj = allocateSpilled(me, size, align); // 慢路径
  }
  return obj;
}
```

逻辑很直白：先把 `bumper` 按 `align` 向上取整得到候选地址 `o`，如果 `o + size` 没超过 `end`，就把 `bumper` 更新成 `o + size`、返回 `o`。`COMPILER_EXPECT(..., true)` 是在告诉编译器：这条分支几乎总是成立，按"成立"来优化分支预测。只有当当前 hunk 真的不够用的时候，才退到 `allocateSpilled` 慢路径。

那行对齐位运算 `(bumper + align-1) & ~(align-1)` 看着唬人，其实是个经典写法：`align` 是 2 的幂，`align-1` 就是"低位全 1"的掩码，取反后低位全 0。先加 `align-1` 保证够"进位"，再按 `align` 向下抹零，等价于"向上取整到 align 的倍数"，但只用加法和位与、没有除法。比如 `bumper=13, align=8`：`(13+7)&~7 = 20&~7 = 16`，正是 ≥13 的最小 8 的倍数。

对外暴露的 `ncclMemoryStackAlloc` 在快路径基础上加了一件事：清零。它总是把返回的内存 `memset` 成 0（`utils.h:275-279`）。所以调用方拿到的内存默认就是干净的，不用自己再 memset。

另外两个模板重载也是直接调内部 `allocate` 再清零。类型化的 `Alloc<T>` 用 `sizeof(T)` 和 `alignof(T)` 自动推导：

```cpp
template<typename T>
inline T* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t n) {
  void *obj = ncclMemoryStack::allocate(me, n*sizeof(T), alignof(T));
  memset(obj, 0, n*sizeof(T));
  return (T*)obj;
}
```

还有个更专门的重载 `AllocInlineArray<Header, Element>`，用来在**一块连续内存里同时放一个 Header 和紧随其后的若干个 Element**。它的做法是把 Header 的大小向上对齐到 `alignof(Element)`，再追加 `nElt` 个 Element 的大小，整体对齐取两者里更严格的：

```cpp
template<typename Header, typename Element>
inline Header* ncclMemoryStackAllocInlineArray(struct ncclMemoryStack* me, size_t nElt) {
  size_t size = sizeof(Header);
  size = (size + alignof(Element)-1) & -alignof(Element);  // Header 按 Element 对齐取整
  size += nElt*sizeof(Element);                            // 追加数组
  size_t align = alignof(Header) < alignof(Element) ? alignof(Element) : alignof(Header);
  void *obj = ncclMemoryStack::allocate(me, size, align);
  memset(obj, 0, size);
  return (Header*)obj;
}
```

布局上长这样，一块连续内存里头部在前、数组紧贴其后：

```
分配返回的地址
  │
  ▼
┌──────────┬─────────┬───────────┬───────────┬─────────────┐
│  Header  │ padding │ Element[0]│ Element[1]│ ... Element │
│          │ (对齐)  │           │           │   [nElt-1]  │
└──────────┴─────────┴───────────┴───────────┴─────────────┘
▲                     ▲
│                     │
返回 Header*            Element 数组按 alignof(Element) 对齐开始
调用方写 Header         调用方用 (Element*)(header+1) 或 memcpy 写入
```

padding 是为了让 Element 数组的起点满足 `alignof(Element)`——把 Header 的大小向上取整到这个对齐值。这种"头部 + 变长负载"的紧凑布局省去了为负载单独分配、单独跟踪、单独释放的麻烦——frame 一弹，整块连带回收。NCCL 在构造工作节点时大量用到它（见后文实例）。

## 分配的慢路径：当前 hunk 不够用时怎么办

`allocateSpilled` 是 `ncclMemoryStack` 复杂性集中所在，在 `utils.cc:260-335`。它要处理"hunk 用满了"的各种情况，思路是按优先级依次尝试几种出路。

**第一种情况：当前 hunk 还剩很多(≥8KB)，却还是放不下这个对象。** 这说明对象本身就比较大。与其为它新开一个大 hunk（可能浪费），不如直接给它单独 `malloc` 一块，再在当前 hunk 里放个小代理登记一下。代码里直接 `goto unhunked` 跳到最后那段处理：

```cpp
if (me->topFrame.end - me->topFrame.bumper >= 8<<10)  // 剩余 ≥8KB 却放不下
  goto unhunked;
```

**第二种情况：复用上方已经空了的 hunk。** 这里要解释一个关键点——`Pop` 的时候 hunk 本身并没有被 `free` 掉，只是 frame 状态回退了。那些用过又空出来的 hunk 通过 `above` 指针串在当前 hunk 之上，作为缓存留着下次复用。`above` 注释里叫 "reverse stack pointer"，就是因为它指向的是逻辑上"更高、更早分配过、现已空"的块。慢路径会先看看 `top->above` 有没有一个放得下的：

```cpp
if (top && top->above) {
  struct Hunk* top1 = top->above;
  uintptr_t uobj = (reinterpret_cast<uintptr_t>(top1) + sizeof(struct Hunk) + align-1) & -uintptr_t(align);
  if (uobj + size <= reinterpret_cast<uintptr_t>(top1) + top1->size) {
    me->topFrame.hunk = top1;        // 切到这个空 hunk
    me->topFrame.bumper = uobj + size;
    me->topFrame.end   = reinterpret_cast<uintptr_t>(top1) + top1->size;
    return reinterpret_cast<void*>(uobj);
  }
}
```

这个分支什么时候才进得去，是个容易卡住的问题，后文"几个容易卡住的关键点"会专门讲。

**第三种情况：真的需要开新 hunk，或者外逃。** 先估算新 hunk 的大小——上一个 hunk 的大小再加 64KB，容量随溢出次数线性增长，这样反复遇到同样规模的分配时不会反复开一样大的块：

```cpp
size_t nextSize = (top ? top->size : 0) + (64<<10);
```

但开之前再判断一下：如果预计这个新 hunk 也容不下当前对象（说明对象太大），而且当前 hunk 里还能放下一个 `Unhunk` 代理，那就别开 hunk 了，直接走外逃更划算。否则就真的 `malloc` 一块新 hunk，挂到 `above` 链上，把它设成当前 hunk，再尝试把对象放进去。

**第四种情况：外逃分配（`unhunked` 标号那段）。** 到这一步，决定让对象脱离 hunk 流单独 `malloc`，但要在 hunk 里留一个 `Unhunk` 代理 `{next, obj}` 登记它，这样以后 `Pop` 的时候能顺着代理链找到并 `free` 掉对象本体。这个机制用图最能说明白：

```
── hunk 内部（bump 流）─────────────────────────────────
  [ ...其他对象 | Unhunk 代理 | (bumper 继续往后挪) ... ]
                   │
                   │  { next, obj }
                   │
                   └──obj──► ┌─────────────────┐   ← 独立 malloc 的大块
                             │   大对象本体     │     不在 hunk 线性流里
                             └─────────────────┘

  本 frame 的外逃代理链（头插）:
    topFrame.unhunks ──► [这个代理] ──► [更早的外逃代理] ──► null
```

代理本身只占十几字节（一个 `next` 加一个 `obj` 指针），老老实实待在 hunk 的 bump 流里；它 `obj` 指向的那块大内存是单独 `malloc` 来的。`Pop` 时顺着 `unhunks` 链逐个 `free` 掉 `obj` 本体即可。机制的本质就在这里：把大对象从 LIFO 的 bump 流里抽出来单独管，但用一个位于 hunk 内的小代理追踪它——既不破坏 hunk 线性分配的简洁，又能在回收时正确处理掉大对象。最后，如果连 `malloc` 都失败，函数会 `WARN` 之后 `abort()`—— NCCL 把这视为不可恢复错误。

## Push 和 Pop：frame 级的压栈弹栈

理解了 frame 和 hunk，Push/Pop 就很自然了。`Push` 在 `utils.h:299-306`：

```cpp
inline void ncclMemoryStackPush(struct ncclMemoryStack* me) {
  using Frame = ncclMemoryStack::Frame;
  Frame tmp = me->topFrame;                                          // 先快照当前 frame
  Frame* snapshot = (Frame*)ncclMemoryStack::allocate(me, sizeof(Frame), alignof(Frame)); // 在栈内分配空间存快照
  *snapshot = tmp;                                                   // 拷贝过去
  me->topFrame.unhunks = nullptr;                                    // 新 frame 起始：没有外逃对象
  me->topFrame.below = snapshot;                                     // 新 frame 的 below 指向旧 frame 快照
}
```

这里有个很巧的地方：**保存旧 frame 用的内存，就是从"当前这个旧 frame"自己里面 allocate 出来的**。也就是说旧 frame 的状态被原样写进了它自己内部的一块 `sizeof(Frame)` 空间，然后 `topFrame` 被改写成一个"延续当前 hunk 和 bumper、但 `below` 指向那份快照"的新 frame。Push 几乎零额外开销——一次小 bump 加一次结构体拷贝，连额外堆分配都没有。

`Pop` 是这一切的 payoff，在 `utils.h:308-315`：

```cpp
inline void ncclMemoryStackPop(struct ncclMemoryStack* me) {
  ncclMemoryStack::Unhunk* un = me->topFrame.unhunks;
  while (un != nullptr) {   // 先 free 本 frame 内所有外逃对象
    free(un->obj);
    un = un->next;
  }
  me->topFrame = *me->topFrame.below;  // 整体恢复成上一层 frame
}
```

它做两件事：先遍历当前 frame 的外逃对象链，逐个 `free` 掉对象本体；然后把 `topFrame` 整体覆盖回 `below` 指的那份快照——bumper、end、hunk 全部回到 Push 之前的状态。

关键就在第二步：**当前 frame 在 hunk 里分配的成百上千个小对象，根本不需要逐个 free**，只要把 `bumper` 回拨，它们就全"逻辑回收"了，hunk 内存留作下一个 frame 复用。真正的代价只有外逃对象链的 `free`（通常很少）加上一次结构体赋值。这就是头部注释里 "deallocation is extremely cheap since it's done at the frame granularity" 的字面含义——回收代价从 O(对象数) 降到了 O(外逃对象数 + 1)。

那这一步的"成片释放"到底发生了什么，看图最直观。`Pop` 之后 hunk 里的对象一字节都没动，bird's-eye 视角是 `bumper` 往左一跳：

```
Pop 之前（当前 frame 堆了一堆对象）          Pop 之后（回到上一层 frame）

hunk: [ Hdr | o | o | o | o | o | (空) ... ]   hunk: [ Hdr | o | o |  (空..............) ]
              ▲            ▲                              ▲   ▲
              │            └─ bumper                      │   └─ bumper 回拨到这里
              └─ 这一批小对象                              └─ 上一 frame 留下的对象
                  (随本 frame 分配)                           保留不动

  └─ unhunks 链: U1 ─► U2 ─► null         └─ U1.obj、U2.obj 已 free，整条链废弃
                  │    │
                 obj  obj  ← 这些 malloc 的大对象被逐个 free 掉

  below ──► [上一层 frame 快照]            topFrame 直接被这份快照覆盖
```

左边是某个 frame 在它生命周期里攒下的状态：hunk 里 bumper 之前挤满了它分配的小对象，unhunks 链上挂着几个外逃的大对象。`Pop` 一执行，右边就是结果——bumper 回到上一层 frame 当时的位置（再往左的对象是更早 frame 留下的，保留不动），hunk 内存本身没被 free、留作下次复用；外逃对象逐个 free 掉，代理链废弃；整个 `topFrame` 被上一层的快照覆盖。注意：**这一大批小对象的"回收"没有产生任何 per-object 操作**，只是 bumper 一个指针回拨——这正是 frame 级 LIFO 的全部魔力所在。

## 销毁：Destruct

栈销毁时要真正把所有 hunk 和外逃对象还给系统，在 `utils.cc:337-355`：

```cpp
void ncclMemoryStackDestruct(struct ncclMemoryStack* me) {
  // 先 free unhunks：因为 frame 快照和 unhunk 代理都躺在 hunk 内部，
  // 必须在 free hunk 之前访问它们，否则就是 use-after-free
  struct ncclMemoryStack::Frame* f = &me->topFrame;
  while (f != nullptr) {
    struct ncclMemoryStack::Unhunk* u = f->unhunks;
    while (u != nullptr) { free(u->obj); u = u->next; }
    f = f->below;
  }
  // 再沿 above 链 free 所有真实 hunk
  // stub 不是 malloc 来的，所以从 stub.above 开始
  struct ncclMemoryStack::Hunk* h = me->stub.above;
  while (h != nullptr) {
    struct ncclMemoryStack::Hunk *h1 = h->above;
    free(h);
    h = h1;
  }
}
```

注释点明了顺序的重要性：frame 快照本身和 unhunk 代理都存在 hunk 内部，所以必须先顺着 frame 链 free 完所有外逃对象，再去 free hunk，否则会访问已释放的内存。`stub` 是占位用的、不是 `malloc` 来的，所以 hunk 遍历从 `stub.above` 开始，跳过它。

## 几个容易卡住的关键点

读完前几节机制，有几个问题读者常会追问，而且答案常常跟直觉相反。把它们单独讲清楚，整个 `ncclMemoryStack` 的行为就不再有藏着的细节。

### hunk 才是真正的整块 buffer，frame 和 memoryStack 都只是簿记

严格说，frame 和 memoryStack 本身都不是一块连续 buffer。真正充当整块连续内存的是 **hunk**——每一块 hunk 是一次 `malloc` 拿到的、地址连续的大块，对象物理上就紧凑摆在它的可用区里。memoryStack 和 frame 都不含数据载荷，只用来"记账"，记录"我现在在用哪块 hunk、用到哪儿了"。

```
memoryStack（簿记容器）
 │
 ├─ 挂着若干 hunk ──► [整块连续 malloc 内存] ◄── 真正的 buffer 在这儿
 │                    stub.above → h1 → h2 → ...
 │
 └─ topFrame（簿记，非 buffer）
      ├─ hunk: 指针,指向"当前在用哪块"
      ├─ bumper/end: 指针,指向那块 hunk 内部用到哪、到头在
      ├─ unhunks: 代理链头
      └─ below: 上一层 frame 快照
```

所以一个 memoryStack 跟内存的关系是"管辖一串 malloc 块"，不是"本身是一整块"。frame 快照本身都被 Push 时塞进了某块 hunk 的内存里——frame 连自己都是"寄居"在 hunk 里的。

### 一个 memoryStack 有几个 frame？一个 frame 有几个 hunk？

**frame 的数量 = nil frame（1 个）+ 当前还没弹掉的 Push 次数。** 栈 `Construct` 后就有且仅有 nil frame 这一个底，永远存在、永不可 Pop。此后每次 `Push` 摞一层，每次 `Pop` 摞掉最上层（但不可能是 nil）。所以任意时刻 frame 数量由 Push/Pop 的嵌套深度决定。对应到 NCCL 用法：`memPermanent` 从不 Push，自始至终只有 nil frame 一层——通信器常驻内存全分配在 nil frame 里，靠"nil 不可弹"保证它们活到通信器销毁。`memScoped` 每次进 group 调一次 Push（多一层）、出 group 调一次 Pop（少一层），group 进行期间通常是 nil + 1 层作用域 frame = 2 层。

**一个 frame 有一个"活跃 hunk"（bumper/end 所指那块），外加通过 `above` 可达的若干空 hunk 缓存；多帧可共用同一块活跃 hunk 的不同区段。** `topFrame.hunk` 是单指针，只指向当前正在往上分配的那块。但 Pop 时不 free 的旧 hunk 经 `above` 串起来留作复用，从当前 hunk 顺着 `hunk->above` 能到一串"已空、待复用"的块。另外 Push 时新 frame 延续旧 frame 的 hunk 和 bumper 继续往后分配——两层 frame 可以共用同一块 hunk 的不同段。

### 什么时候会新建一个 hunk（malloc 新 hunk）

新 hunk 的 `malloc` 只在快路径 `allocate` 发现当前 hunk 放不下对象、进入慢路径 `allocateSpilled` 之后才可能发生；而且慢路径会先试几条更省的出路，都不行才真的申请。条件梳理清楚是这样：

- **触发慢路径**：当前 hunk 剩余空间装不下这次的对象（含对齐 padding），即 `o + size > end`。
- **慢路径按优先级试四条出路**，只有前三条都没走掉才 malloc：
  1. 当前 hunk 还剩 ≥8KB 却仍放不下 → 对象偏大，走外逃（给对象单独 malloc，不开 hunk）。
  2. `top->above` 有可复用的空 hunk 且对象放得下 → 复用，不 malloc（见下一小节）。
  3. 预估新 hunk（上一块大小 + 64KB）也容不下对象、当前 hunk 还能塞下 Unhunk 代理 → 外逃。
  4. 前三条都不成立 → 真 `malloc` 新 hunk，大小为 `(top->size + 64KB)`，挂到 `above` 链上设为活跃。

几个要点。每块新 hunk 比上一块大 **64KB**，容量线性增长（第一块 64KB，第二块 128KB，第三块 192KB…），随分配规模变大而长，避免反复开同样大小的块。第一块真实 hunk 诞生于栈的第一次分配——`Construct` 后 `topFrame` 挂的是没有真实内存的 `stub`，`bumper=end=0`，第一次 `allocate` 必走慢路径。注意**超大对象不会催生等大的 hunk**：如果对象大到 `nextSize < sizeof(Hunk)+64+size`，会走第三条外逃路径单独 malloc 放置，hunk 只服务常规大小的分配。`malloc` 失败则 `WARN` 后 `abort()`，NCCL 视为不可恢复错误。

### hunk 什么时候 free：Pop 完全不 free hunk，只有 Destruct 才释放

这是最容易记错的一点。**Pop 期间 hunk 永远不 free**，被 Pop 掉的 hunk 只是被"晾"起来留作复用。hunk 真正被 `free` 只发生在 `ncclMemoryStackDestruct`（`utils.cc:337-355`）。前面 Destruct 那节已经把代码列过，这里只把归宿对照清楚：

- **unhunks 链上的 obj**：Pop 逐个 free（本 frame 的）；Destruct 也逐个 free（沿整条 frame 链）。
- **hunk 里的小对象**：Pop 不 free，靠 bumper 回拨逻辑回收；Destruct 不单独 free，随 hunk 整体被 free。
- **hunk 本体**：Pop **完全不 free**，留作复用；Destruct 沿 `above` 链逐个 free。

也就是说，hunk 一旦 `malloc` 出来，就一直在栈里被持有、被复用，直到整个 `ncclMemoryStack` 被销毁才一次性归还系统。这个"延迟到销毁才还"的策略换来的是 Push/Pop 期间零系统释放、热路径极快，代价是 hunk 内存会在栈存活期一直被持有。对应到 NCCL：`memScoped`、`memPermanent` 攒下的 hunk 都留到 `ncclCommDestroy` 才各调一次 `Destruct` 彻底归还——只要通信器还活着，它名下 memoryStack 的 hunk 就一直被持有复用。

### 慢路径"复用 above 空块"分支：只有 Pop 过的栈才进得去

`allocateSpilled` 里这段复用分支常常让人困惑——`above` 链方向是老→新，而 Push/Pop 似乎只 free unhunk 的 obj、和 hunk 无关，那这块"空 hunk"到底从哪冒出来？

```cpp
if (top && top->above) {                 // top 是当前活跃 hunk
  struct Hunk* top1 = top->above;        // top1 是 above 链上的下一块
  ... if (对象放得下 top1) { 切到 top1 用; return; }
}
```

困惑的根源在一句误解：**Pop 其实会动 `topFrame.hunk` 这个指针**。`Pop` 那行 `me->topFrame = *me->topFrame.below` 是整块结构体覆盖，快照里存着旧 frame 的 `hunk` 字段，所以 Pop 会把"当前活跃 hunk"指针**回拨到 Push 时所在的那块（更老的）hunk**。正是这个回拨，让一块"更新但已空"的 hunk 被晾在 `above` 链上、等下次复用——这就是本分支能进去的根因。

用一个具体时间线坐实它（记两块 hunk：`h1` 64KB 先建、`h2` 128KB 后建，链方向 `stub.above→h1→h2→null`，老→新）：

```
初始    : topFrame.hunk = stub（无内存）
第1次分配: stub 放不下 → malloc h1, stub.above=h1, 活跃=h1
Push    : 快照当前 {hunk=h1,...} 存进 h1; 新 frame 仍 hunk=h1, bumper 继续推进
分配一堆: h1 满了 → malloc h2, h1.above=h2, h2.above=null, 活跃=h2
          注意此刻 top=h2, 而 h2.above=null——同一 frame 一路往前分配时,
          活跃 hunk 永远是最新的,它的 above 必空,本分支根本进不去
Pop     : free 本 frame unhunks; topFrame 被快照覆盖回 {hunk=h1,...}
          ★ topFrame.hunk 从 h2 回拨成 h1 ★
          h2 不再被 topFrame 指着,但 h1.above=h2 链还在 → h2 变成"空 hunk 缓存"
          (h2 里的对象全随 bumper 回拨逻辑作废了,内存字节还在但账本上没人用)
下次分配: topFrame.hunk=h1, 从旧 bumper 推进; h1 又满 → 进 allocateSpilled
          top = h1,  top->above = h2  ← 非空!  对象放得下 → 命中本分支,切回 h2 复用,跳过 malloc
```

所以本分支能进去的完整条件是：**此前发生过 Pop，把 `topFrame.hunk` 回拨到了一块较老的 hunk，而该老 hunk 的 `above` 恰好指向一块更晚建、随 Pop 已变空的 hunk，且当前对象放得下那块空 hunk。** 没有 Pop 这一步回拨，`top->above` 在"一路往前分配"时永远是 null（因为活跃的总是最新块，它的 above 为空），本分支永远进不去。

由此有个直接推论：**这个分支只有 `memScoped` 这种会 Push/Pop 的栈才用得上**；`memPermanent` 从不 Pop，活跃 hunk 永远是最新的，`top->above` 恒为 null，本分支永远不会命中。它在 NCCL 里是给"有过 group 嵌套、留下空 hunk 缓存"的场景准备的免 malloc 复用优化。

## 在 NCCL 里的实际用法

讲了这么多机制，最终要看它怎么被用。NCCL 给每个通信器 `ncclComm` 配了**两个** `ncclMemoryStack`（`comm.h:525`）：

```cpp
struct ncclComm {
  uint64_t startMagic;
  struct ncclMemoryStack memPermanent, memScoped;
  ...
};
```

两个栈的分工是清楚的。`memPermanent` 用于通信器级别的常驻结构——通道、peer、ring 这些随通信器共存亡的东西。因为它们在 nil frame 里分配，永不被弹，自然活到通信器销毁。`memScoped` 则用于单次 collective 操作里的临时工作内存，靠 Push/Pop 划出作用域，这正是它名字 "scoped" 的由来。

**`memScoped` 的典型用法**在 group 逻辑里（`group.h:123-159`）。通信器加入 group 时压栈，此后该 comm 在这次 group 里批处理的所有任务都分配在这个新 frame 上；离开 group 时一次弹栈，整批临时内存一起回收，零碎片：

```cpp
// 加入 group：压出一个新作用域
ncclMemoryStackPush(&comm->memScoped);
...
// 离开 group：弹栈，整批临时分配一次回收
inline ncclResult_t ncclGroupCommLeave(struct ncclComm* comm, int type) {
  comm->groupNext[type] = reinterpret_cast<struct ncclComm*>(0x1);
  ncclMemoryStackPop(&comm->memScoped);
  return ncclSuccess;
}
```

**`memPermanent` 的典型用法**是通信通道的元数据（`channel.cc:34-57`），用类型化的 `Alloc<T>` 直接拿到清零的数组，随通信器常驻：

```cpp
channel->peers = ncclMemoryStackAlloc<struct ncclChannelPeer*>(&comm->memPermanent, nPeers);
channel->ring.userRanks   = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
channel->ring.rankToIndex = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
```

**`AllocInlineArray` 的典型用法**是构造工作节点（`enqueue.cc:336-351`）。先分配一个 `ncclWorkList` 头部加一个紧随其后的 `ncclDevWorkCollReg` 负载，写完头部后用 `memcpy` 把负载拷进 `header+1` 的位置。一次分配、随 frame 一次回收，紧凑且无需单独管理：

```cpp
workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkCollReg>(&comm->memScoped, 1);
workNode->workType = ncclDevWorkTypeCollReg;
workNode->size = sizeof(struct ncclDevWorkCollReg);
memcpy((void*)(workNode+1), (void*)&workReg, workNode->size);  // 负载紧跟头部
```

这个"头部 + 内联负载"的模式在 `enqueue.cc` 和调度器里反复出现。需要自定义对齐时（比如 16 字节对齐的 GPU kernel 参数块），则回到原始的 `void*` 重载，例如 `enqueue.cc:216`：

```cpp
plan->kernelArgs = (struct ncclDevKernelArgs*)ncclMemoryStackAlloc(&comm->memScoped, plan->kernelArgsSize, /*align=*/16);
```

## 设计要点回顾

把上面散落的要点收拢一下，`ncclMemoryStack` 的设计可以归结为几条相互配合的取舍。

它把 LIFO 放在了 frame 粒度而不是对象粒度，这是它"释放极快"的根源——一整批对象靠回拨 `bumper` 一次性逻辑回收，只有少数外逃的大对象需要真正 `free`。hunk 加 bump 指针让快路径退化成一次对齐加法和一次比较，`COMPILER_EXPECT` 进一步锁住分支预测。Pop 时 hunk 不立即 `free`，而是经过 `above` 链缓存起来复用，避免反复向系统要内存。遇到大对象时，Unhunk 外逃机制让它脱离 bump 流单独 `malloc`，但用一个位于 hunk 内的小代理登记，Pop 时按链回收——既不撑爆 hunk，也不破坏线性分配的简洁。

nil frame 作为永不弹出的根 frame，一方面用崩溃保证栈始终至少有一层有效，另一方面让"nil frame 内分配 = 栈生命周期分配"成为一个干净的语义，NCCL 用它实现通信器常驻内存。Push 把旧 frame 的快照存进当前 hunk 自身，零额外堆分配，旧状态随 Pop 自然回滚。所有 `Alloc` 都清零，调用方默认拿到干净内存。最后，`memPermanent` 与 `memScoped` 的双栈约定，把"通信器级常驻"和"任务级临时"在数据结构层面分开，把人从手动管理生命周期的负担里解放出来。

## 源码索引

- 头部注释与前置声明：`src/include/utils.h:120-140`
- 结构体定义：`src/include/utils.h:230-251`
- Construct：`src/include/utils.h:253-261`
- allocate 快路径：`src/include/utils.h:263-273`
- 三个 Alloc 重载：`src/include/utils.h:275-297`
- Push / Pop：`src/include/utils.h:299-315`
- allocateSpilled 慢路径：`src/misc/utils.cc:260-335`
- Destruct：`src/misc/utils.cc:337-355`
- 通信器双栈字段：`src/include/comm.h:525`
- 作用域 Push/Pop 用法：`src/include/group.h:123-159`
