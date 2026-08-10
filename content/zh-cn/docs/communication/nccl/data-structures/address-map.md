---
title: ncclIntruAddressMap：侵入式指针哈希表
description: 类型擦除的侵入式哈希表，把 key 和 next 字段留在调用方对象里；Knuth 乘法哈希、懒初始化、2:1 扩容再哈希、空表自动回收。
weight: 4
---

前三篇讲了 [`ncclMemoryStack`](memory-stack) 管分配、[`ncclMemoryPool`](memory-pool) 管回收、[`ncclIntruQueue`](intru-queue) 管排队。但有些场景不是"排队"，而是"按 key 查找"——手里攥着一个设备内存句柄，想知道它对应的 `ncclDevrWindow` 在哪。用排过序的数组可以，但插入/删除都不是常数；用 `std::unordered_map` 又会为每个键值对额外 malloc 一个节点，热路径上不划算。`ncclIntruAddressMap` 就是来补这个缺口的，一个**侵入式（intrusive）的哈希表**。

所谓"侵入式"，思路和 `ncclIntruQueue` 一脉相承：哈希表本身不为每个条目分配节点，而是把"指向同桶下一个条目"的 `next` 指针**留在用户对象自己的结构体里**，由调用方通过模板参数告诉表"用哪个字段当 key、哪个字段当 next"。表自己只持有一个桶数组（`void** table`），其余零分配。区别只在于：队列是按入队顺序串成一条链，而这里是按 key 哈希分进各桶、每桶一条链。源码拆成两部分——类型安全的外壳内联在 `src/include/utils.h:575-712`，类型擦除的内核实现在 `src/misc/utils.cc:383-604`。

> 相关阅读：和 [`ncclIntruQueue`](intru-queue) 是亲兄弟，都用成员指针模板把 next 字段外置给调用方。区别是队列只存 head/tail 两个指针、表额外存一个桶数组并按 key 寻桶。

## 整体模型：untyped 内核 + typed 外壳

这个结构最值得先看的设计是**类型擦除的两层结构**。先把类型无关的内核挖出来（`utils.h:625-629`）：

```cpp
struct ncclIntruAddressMap_untyped {
  int hbits;    // log2 of table size
  int count;    // number of entries
  void** table;
};
```

就三个字段。`hbits` 是桶数的 log2（桶数 = `1<<hbits`），`count` 是当前条目数，`table` 是桶数组，每个桶指向桶内链表的头节点。这个内核完全用 `void*` 操作，不关心 key 是什么类型、next 挂在对象哪个位置——那些信息以"字节偏移"的形式作为参数传进来，所以所有实操作都在 `_untyped` 后缀的函数里，集中在 `utils.cc`。

类型安全的外壳是个模板（`utils.h:632-640`）：

```cpp
template<typename Obj, typename Key, Key Obj::*keyField, Obj* Obj::*nextField>
struct ncclIntruAddressMap {
  static_assert(sizeof(Key) <= sizeof(uintptr_t),
    "ncclIntruAddressMap: Key type size must be <= sizeof(uintptr_t). "
    "Keys larger than a pointer cannot be safely converted to uintptr_t.");
  ncclIntruAddressMap_untyped base;
};
```

四个模板参数把一切都钉死了：对象类型 `Obj`、键类型 `Key`、`Obj` 里当 key 的成员指针 `keyField`、`Obj` 里当 next 的成员指针 `nextField`。一条 `static_assert` 在编译期卡住"key 比 a 指针还大"的用法——因为内核会把 key 强转成 `uintptr_t` 来哈希和比较，key 大过一个字就没法无损塞进去了。

外壳在组合 `base` 之外不存任何东西，它真正的工作是把成员指针**翻译成字节偏移**，再调内核。翻译的手法（`utils.h:679`）是这份代码里最值得停下来想一眼的几行：

```cpp
Obj dummy;
int keyFieldOffset = (char*)&(dummy.*keyField) - (char*)&dummy;
int nextFieldOffset = (char*)&(dummy.*nextField) - (char*)&dummy;
```

注释说得很直白：本想用 `offsetof` 宏，但它对含继承等"非 C 类型"的类不工作，所以改用这个比 `offsetof` 更通用的等价写法。`dummy.*keyField` 是"dummy 对象里 keyField 指向的那个成员"的引用，取地址再减去 dummy 自己的地址，就是该成员在结构体里的字节偏移。拿到偏移后，内核就能用 `(char*)object + offset` 光着类型去读写那个字段了。三个 typed 包装函数 Insert/Find/Remove（`utils.h:672-711`）干的全是同一套：算两个偏移、把 key `reinterpret_cast<uintptr_t>`、把 `&map->base` 连同偏移一起喂给 `_untyped` 版本。

这就是类型擦除的来回：**外壳负责类型安全和编译期校验，内核负责实际工作并以 `void*`+偏移的形式与对象解耦**。好处是所有哈希表逻辑只写一份、不随模板实例化膨胀，坏处是内核里看不到任何类型信息，读写 key/next 全靠 `memcpy` 和裸指针算术——下面会看到为什么会这样。

## 哈希函数：Knuth 乘法哈希

桶怎么选由 `ncclHashPointer` 决定（`utils.cc:385-390`），注释里写"Uses shadowpool's algorithm"：

```cpp
uint64_t ncclHashPointer(int hbits, void* key) {
  uintptr_t h = reinterpret_cast<uintptr_t>(key);
  h ^= h>>32;
  h *= 0x9e3779b97f4a7c13;
  return (uint64_t)h >> (64-hbits);
}
```

这是个典型的指针哈希。指针本身做 key 有个麻烦：低几位往往高度聚集（对齐导致低位是 0），高位又常常反映固定的地址空间布局（比如某个段基址），中间才是较随机的部分。这套做法分三步。先 `h ^= h>>32` 把高 32 位掺到低 32 位里，打散那些固定的高位布局。再乘以那个魔法常数 `0x9e3779b97f4a7c13`——这是 Knuth 提出的乘法哈希常数（黄金比例相关），乘完后高位会聚拢 key 的全部熵。最后右移 `64-hbits` 位，只取最高的 `hbits` 位当桶号；之所以取高位，是因为乘法哈希的随机性集中在乘积的高位。

最终效果是：任意一个指针，都能被相当均匀地散到 `[0, 1<<hbits)` 的桶里，且计算只有一次异或、一次乘、一次移位，极便宜。这个函数被两处共用——除了这里的侵入式表，`allocator.cc:323/421/454` 里 `ncclShadowPool` 那套非侵入式的设备对象池也用它，所以函数声明放在 `utils.h:577` 当公共设施，实现在 `utils.cc`。

## 类型擦除下的字段读写：为什么全是 memcpy

内核和对象之间只靠偏移交流，但对象里 key/next 字段的真实类型（比如 `ncclWindow_vidmem*`）外壳知道、内核不知道。如果内核直接写 `*(uintptr_t*)((char*)obj+offset) = key`，在真实类型不是 `uintptr_t` 而是指针时，会触发严格别名（strict aliasing）违例，UB 下编译器可能优化出错误结果。所以内核用三个小助手（`utils.cc:397-418`）做类型无关的读写，全部走 `memcpy`：

```cpp
// 读 key:读 keySize 字节,零扩展成 uintptr_t
static inline uintptr_t readKey(void* object, int keySize, int keyFieldOffset) {
  void* keyPtr = (char*)object + keyFieldOffset;
  uintptr_t result = 0;
  memcpy(&result, keyPtr, keySize);
  return result;
}
// 读 next:读一个机器字
static inline void* readNextPtr(void* object, int nextFieldOffset) {
  void* nextPtr = (char*)object + nextFieldOffset;
  void* result = nullptr;
  memcpy(&result, nextPtr, sizeof(void*));
  return result;
}
// 写 next
static inline void writeNextPtr(void* object, int nextFieldOffset, void* value) {
  void* nextPtr = (char*)object + nextFieldOffset;
  memcpy(nextPtr, &value, sizeof(void*));
}
```

`memcpy` 是 C 标准里"对任何类型都合法"的类型双关手法，编译器知道这块内存可能被任意类型访问，不会按某一种类型做越界别名优化。注意 `readKey` 里 `result` 先置 0 再只 `memcpy` 前 `keySize` 字节——这是"读窄类型 key 并零扩展成 uintptr_t"。key 比一个字窄时（比如 32 位 key），只有低 32 位被实际写入，高位保持 0；插入端传进来的 `key` 也是 `reinterpret_cast<uintptr_t>` 来的，同样高位是 0，所以两边比较时高位都干净，不会错配。这套 memcpy 粘合层就是"外壳强类型 + 内核无类型"能安全共事的底座。

## Insert：懒初始化、2:1 扩容、头插

Insert 是最复杂的操作（`utils.cc:420-506`）。先做运行时校验（map 非空、object 非空、`0 < keySize <= sizeof(uintptr_t)`），然后是**懒初始化**——表构造时并不分配桶数组，第一次插入才建：

```cpp
if (map->hbits == 0) {
  map->hbits = 4;                                         // 16 个桶起步
  map->table = (void**)calloc((size_t)1<<map->hbits, sizeof(void*));
  ...
}
```

`hbits==0` 既表示"空表"也表示"还没初始化"，于是零初始化（`{}`）天然就是合法的初始态，全局 static 实例自动满足，无需构造函数。起步 `hbits=4` 即 16 个桶，用 `calloc` 一次清零，桶里全是 nullptr。

接着是**扩容判断**。注释强调维持 2:1 的"对象:桶"比例：

```cpp
if (map->count+1 > 2<<hbits) {            // 插完会超过 2 倍桶数
  int newHbits = hbits + 1;               // 桶数翻倍
  ...
}
```

注意 `2<<hbits` 等价于 `2 * (1<<hbits)`，即桶数的两倍。也就是说，**当条目数将超过桶数的 2 倍时，桶数翻倍**，把装载因子（条目/桶）控制在 2 以下。2:1 这个比例有点意思：Java HashMap 是 0.75，多数链地址哈希表维持在 1 左右。这里放宽到 2，是为了在"每个桶平均 2 个节点"时就扩容，配合指针 key 良好的哈希分布，桶链短、查找快，同时扩容频率比 1:1 低一半。代价是稳态下平均桶链略长（≤2 各节点），对查找是常数级、可接受。

扩容时有个细节值得记一下：**不用 `realloc`，而是全新 `malloc` 再手工再哈希**。注释直说原因——`realloc` 可能在原地扩展内存，那样新桶区和旧桶区重叠，再哈希过程中边读旧桶边写新桶会互相踩坏。所以走 `malloc` 新表、清零、逐个旧桶遍历链表、对每个节点用新 `hbits` 重新哈希进新桶、最后 `free` 旧表：

```cpp
for (int i = 0; i < oldSize; i++) {
  void* obj = map->table[i];
  while (obj) {
    void* next = readNextPtr(obj, nextFieldOffset);       // 先存下一条(next 要被覆写)
    uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
    uint64_t b = ncclHashPointer(newHbits, (void*)objKey);
    writeNextPtr(obj, nextFieldOffset, newTable[b]);       // 头插进新桶
    newTable[b] = obj;
    obj = next;
  }
}
```

再哈希用的是每个对象的**真实 key**（从 keyFieldOffset 读出来），不是它原来在哪个桶——这一点很关键，因为桶号会随 `hbits` 变。`while(obj)` 循环里必须先用 `next` 抓住下一条，再覆写当前节点的 next 字段，否则覆写之后就找不到下一个了——这是链表遍历改链的经典套路。

扩容完，剩下的就是**头插**（`utils.cc:490-503`）：

```cpp
uint64_t b = ncclHashPointer(map->hbits, (void*)key);
void* currentNext = readNextPtr(object, nextFieldOffset);
if (currentNext != nullptr) {
  INFO(NCCL_INIT, "Intrusive map: inserting object %p with non-NULL next pointer %p ...");
}
writeNextPtr(object, nextFieldOffset, map->table[b]);     // 新节点 next 指向原桶头
map->table[b] = object;                                    // 新节点变桶头
map->count += 1;
```

头插，O(1)，和队列 Enqueue 一样只动指针。不过插入前有个检查：如果待插对象的 next 字段已经非空，会打一条 INFO 日志提示"它可能已经挂在某条链上了"。这是侵入式结构的一个固有风险——next 字段是调用方对象的一部分，对象被复用前如果没把 next 清空，表无法分辨"这个 next 是上次挂在别的链上留下的"还是"就是个脏值"。`ncclWindowMap` 的调用方在 `dev_runtime.cc:1068` 会显式 `winHost->next = nullptr` 再插入，正是为了规避这条告警、也避免对象身兼多链时链表逻辑错乱。

## Find 与 Remove：线性扫桶链

Find（`utils.cc:508-547`）很直接：校验、空表直接返回成功且 `*object=nullptr`（查找空表不算错误）、否则哈希取桶号、沿桶链线性比对 key：

```cpp
*object = nullptr;
if (map->hbits == 0) return ncclSuccess;                  // 空表:未命中,非错误
uint64_t b = ncclHashPointer(map->hbits, (void*)key);
void* obj = map->table[b];
while (obj) {
  uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
  if (objKey == key) { *object = obj; return ncclSuccess; }
  obj = readNextPtr(obj, nextFieldOffset);
}
return ncclSuccess;                                        // 没找到也是成功,*object 仍是 nullptr
```

值得留意的是 Find 的错误语义：**"没找到"不是错误，返回 `ncclSuccess`，只是把输出置 nullptr**。这和"参数非法"（map/object 为 NULL、keySize 越界）返回 `ncclInvalidUsage` 是分开的。所以调用方判 Find 结果要靠"返回值成功 + 输出非空"两个条件，不能只看返回值。桶链平均长度受 2:1 比例约束在 2 以内，所以查找基本是 O(1)。

Remove（`utils.cc:549-604`）多了两件事——改链、以及**空表自动回收**。校验和空表短路同 Find（空表 Remove 也幂等成功）。命中时根据位置改链：删的是桶头就让 `table[b]` 跳到下一个；删的是中间就让前驱的 next 跳过当前：

```cpp
if (prev == nullptr) map->table[b] = next;                // 删桶头
else writeNextPtr(prev, nextFieldOffset, next);           // 删中间/尾
map->count -= 1;
```

改完链，count 减一之后有个值得专门讲的设计——**count 归零时自动回收桶表**（`utils.cc:588-594`）：

```cpp
if (map->count == 0) {
  free(map->table);
  map->hbits = 0;
  map->count = 0;
  map->table = nullptr;
}
```

也就是说，删掉最后一个条目后，表把桶数组 `free` 掉，三个字段全部清零——**回到零初始化态**。这呼应了头部契约里关于"销毁"的那段说明（`utils.h:597-601`）：没有显式 Destruct 函数，因为只要调用方"把插进去的对象都删干净"，表就会自己回到零态，真正销毁变成了 no-op。`ncclIntruAddressMapDestruct`（`utils.h:645-653`）确实存在，但注释明说它"只在抛弃一个非空表、避免泄漏桶表时才需要"。

这个设计的动机是和侵入式语义配套的：表不拥有条目对象（对象由调用方另行 `malloc`/管理），它只拥有自己的桶数组。条目全清后，桶数组也没有保留价值——下次再插时会重新懒初始化一个新的 16 桶小表，比留着一个大表占内存合理。所以"删空即释放"让表的生命周期和"是否在用"严格对齐，不占任何闲置资源。

最后，没找到要删的 key 时 Remove 也返回成功（注释："remove is idempotent"）——重复删同一个 key 不会报错，这对幂等清理路径友好。

## 使用契约：调用方必须删净、非线程安全

头部那段大注释（`utils.h:582-622`）把这些约束写得很显眼，值得单独拎出来。一是**构造**：这是个 POD，全局 static 实例靠自动零初始化就合法，局部变量用 `= {}` 显式零初始化即可，不需要构造函数。二是**销毁**：没有强制 Destruct，因为删空即自动回收；真正要做的是"调用方在抛弃表之前把所有插进去的对象都 Remove 掉"。如果直接丢掉一个非空表，泄漏的不是那些对象（它们归调用方管），而是**桶数组本身**——`table` 那块 `malloc` 出来的内存没人 `free` 了。契约原话："Caller MUST remove all inserted objects before abandoning the map. Failure to remove all objects will leak memory (the internal bucket table)."

三是**对象生命周期**：对象必须在表里存活期间一直活着，因为表读它的 key/next 字段；key 和 next 字段会被表读写修改，调用方别在对象挂在表里时动这两个字段。删出表之后再 `free` 对象本体，顺序不能反。

四是**线程安全**：明确说"Not thread-safe, caller must provide external synchronization"。所有操作都不做原子或加锁，多线程并发访问同一个表必须调用方自己上锁。这一点在真实用法里会兑现——下面看 `ncclWindowMap` 怎么处理。

五是**校验**两档：编译期 `static_assert(sizeof(Key) <= sizeof(uintptr_t))` 卡掉过大的 key；运行期对 map/object 指针非空、keySize 合法做检查，违例返回 `ncclInvalidUsage` 并 WARN。运行期校验相对重一点，但都是一次比较、在错误用法下才触发，热路径上无负担。

## 在 NCCL 里的实际用法：ncclWindowMap

目前 NCCL 里 `ncclIntruAddressMap` 只有一个实例化点，在 `src/dev_runtime.cc:44`：

```cpp
// Global window map using intrusive address map
// Uses ncclDevrWindow directly (vidmem as key, next pointer embedded in struct)
static std::mutex ncclWindowMapMutex;
static ncclIntruAddressMap<ncclDevrWindow, struct ncclWindow_vidmem*,
                           &ncclDevrWindow::vidmem, &ncclDevrWindow::next> ncclWindowMap;
```

它是个进程级全局表，把"对称内存窗口（symmetric memory window）"按其设备侧句柄 `vidmem`（类型 `ncclWindow_vidmem*`，一个指针）索引到 host 侧的 `ncclDevrWindow` 结构体。看 `ncclDevrWindow` 的定义（`src/include/dev_runtime.h:21-31`）正好印证了侵入式约定——key 和 next 是它自己的成员：

```cpp
struct ncclDevrWindow {
  struct ncclDevrMemory* memory;
  void* userPtr;
  ...
  struct ncclWindow_vidmem* vidmem; // key for intrusive map
  struct ncclDevrWindow* next;      // next for intrusive map
  struct ncclComm* comm;            // comm for intrusive map window <> comm look up
};
```

注释直接写明 `vidmem` 是 key、`next` 是为侵入式表准备的。最后的 `comm` 字段也特意标注是"为了从 window 反查 communicator"——也就是说这张表不只是 key→window，还顺带让拿到 window 的人能回溯到它隶属的 comm，省去再开一张表的麻烦。

因为表声明是全局 static、而 NCCL 一个进程里可能有多个 communicator 同时注册/反注册窗口，线程安全就成了硬约束。调用方用前面那个 `static std::mutex ncclWindowMapMutex` 兜底：注册时（`dev_runtime.cc:1064-1073`）在锁内设好 `comm`、清 `next`、再 Insert 并打日志；反注册时（`dev_runtime.cc:969-972`）同样在锁内 Remove。这正是契约"caller must provide external synchronization"的落地——侵入式表自己不管锁，但用法上用一把全局互斥量把整张表的 Insert/Remove 串起来。注意 key 是 `*outWinDev`（注册产出的设备侧窗口句柄），删的时候也是用同一个 key 调 Remove，所以窗口句柄就是它在这张表里的"身份证"。

这张表的真实生命周期形态是这样：窗口随 collective 注册而 Insert、随反注册而 Remove。一个进程活跃时窗口数是有限的（受设备内存和 comm 数约束），桶表会在 16→32→… 之间按 2:1 比例缓慢扩张；进程退出或所有窗口反注册后，count 归零、桶表自动 `free`、回到零态。没有显式 Destruct 调用——契约允许的"删净即可"在这里兑现。

## 放进侵入式家族：和 ncclIntruQueue 的异同

把 `ncclIntruAddressMap` 和上一节的 [`ncclIntruQueue`](intru-queue) 放一起看，侵入式家族的共性就很清楚了。两者都用成员指针模板（`T *T::*next` / `Key Obj::*keyField` + `Obj* Obj::*nextField`）把"链表语义"外置给调用方对象，表/队列本身只持有指向入口的少量指针，节点零额外分配。`ncclIntruQueue` 只有 head/tail 两个指针，把所有节点串成一条 FIFO；`ncclIntruAddressMap` 多一个桶数组，按 key 哈希把节点散进各桶、每桶一条链。前者解决"排队"，后者解决"按 key 查找"，合起来覆盖了热路径上两类最常见的关系组织需求。

差异也很值得记。队列的节点顺序由入队时机决定、是可预测的 FIFO；表的节点顺序由哈希决定、对外不可预测。队列几乎所有操作 O(1)（除 Delete 是 O(n)）；表查找 O(1) 但靠哈希质量 + 装载因子共同保证，所以才有 2:1 扩容和 Knuth 乘法哈希这套配套。队列不需要"按值定位节点"，所以没有 key 概念；表的核心就是按 key 定位，于是多了 key 偏移、key 比较和类型擦除下读窄 key 的零扩展处理。还有一处工程上的差异：队列是纯头文件内联模板、零运行时依赖；表因为内核要放在 `.cc` 里（避免每个实例化都展开一份哈希/扩容逻辑），所以有 `Insert_untyped` 这套类型擦除边界，也因此才需要 memcpy 粘合层来安全读写对象字段。

一个细节上的呼应：两者都把"节点是否已在容器内"的判断责任交给调用方。队列 Enqueue 直接覆写 `x->*next = nullptr`，不管这节点之前挂在哪；表 Insert 会检查 next 是否非空并打 INFO 告警，但也仅止于告警，不阻止。侵入式的代价就在这里——表/队列无法独占节点，调用方必须保证一个对象同一时刻不在两个同字段链上，否则链表逻辑会断。`ncclWindowMap` 靠 `winHost->next = nullptr` 兜底，`ncclIntruQueue` 靠"一个节点备多个 next 字段挂多链"的设计规避，殊途同归。

## 四件容器放一起：stack / pool / queue / map

把这一节加进来，热路径的容器版图就齐了。`ncclMemoryStack` 是**地基**，持有大块 hunk、做 bump 分配。`ncclMemoryPool` 是**回收层**，在 stack 之上维护 per-type 空闲链表，让单个对象跨次复用。`ncclIntruQueue` 是**排队层**，把任务节点串成 FIFO 交给调度器。`ncclIntruAddressMap` 是**索引层**，按 key 快速定位对象，服务那些"排队解决不了、需要按值查"的场景。

索引层和前三层不算同一条流水线——stack/pool/queue 串成"分配→排队→回收"的任务生命周期，而 map 更像是一个旁挂的索引服务：注册窗口时往表里塞一条，后续拿到 key 时查表找出对象，反注册时删掉。它和 `ncclMemoryPool` 共享同样的"零额外分配"哲学（都用对象自带字段当链表节点），和 `ncclIntruQueue` 共享同样的成员指针模板手法，但定位在完全不同的子系统（对称内存窗口管理 vs 任务调度）。所以理解它的价值，更多是看 NCCL 在"需要 O(1) 按 key 查找 + 零热路径分配"时，用什么统一手法解决——就是这个类型擦除的侵入式哈希表。

## 设计要点回顾

`ncclIntruAddressMap` 是个侵入式哈希表，用成员指针模板把 key 和 next 字段外置给调用方对象，自己只持有一个桶数组，条目零额外分配。结构上分两层：typed 外壳负责类型安全和编译期 `sizeof(Key) <= sizeof(uintptr_t)` 校验，把成员指针算成字节偏移喂给 untyped 内核；内核以 `void*`+偏移+`memcpy` 操作对象字段，避开严格别名，实现写一份、不随实例化膨胀。哈希用 Knuth 乘法常数 `0x9e3779b97f4a7c13` 配合 `h>>32` 预混和取高位，对指针 key 散布均匀。

表懒初始化（`hbits=4` 即 16 桶起步，零初始化即合法初始态）、按 2:1 对桶比例在条目超 2 倍桶数时翻倍扩容、扩容走全新 malloc + 逐节点再哈希（不用 realloc 以防新旧桶区重叠踩坏）。Insert 头插 O(1)，对 next 非空打告警但不阻止；Find 沿桶链线性比对 key，未命中返回成功+nullptr；Remove 改链后 count 归零自动 free 桶表回到零态，使得"删净即销毁"，无需显式 Destruct。契约要求调用方删净所有条目再抛弃表（否则泄漏桶表）、对象存活期不碰 key/next 字段、并发访问自行加锁。NCCL 唯一实例 `ncclWindowMap` 在 `dev_runtime.cc`，以 `vidmem` 指针索引 `ncclDevrWindow`、用一把全局 mutex 兜底线程安全，窗口注册/反注册时 Insert/Remove。

## 源码索引

- 哈希函数声明与实现：`src/include/utils.h:577`、`src/misc/utils.cc:385-390`
- 使用契约注释：`src/include/utils.h:582-622`
- untyped 内核结构体：`src/include/utils.h:625-629`
- typed 外壳模板：`src/include/utils.h:632-640`
- Destruct：`src/include/utils.h:645-653`
- typed Insert/Find/Remove 包装：`src/include/utils.h:672-711`
- untyped Insert（懒初始化 + 扩容 + 头插）：`src/misc/utils.cc:420-506`
- untyped Find：`src/misc/utils.cc:508-547`
- untyped Remove（改链 + 空表回收）：`src/misc/utils.cc:549-604`
- readKey / readNextPtr / writeNextPtr 助手：`src/misc/utils.cc:397-418`
- 唯一实例 ncclWindowMap：`src/dev_runtime.cc:43-44`
- 窗口结构体（key/next 字段）：`src/include/dev_runtime.h:21-31`
- 注册时 Insert（加锁）：`src/dev_runtime.cc:1064-1073`
- 反注册时 Remove（加锁）：`src/dev_runtime.cc:969-972`
- 哈希函数的另一个共用者（shadow pool）：`src/allocator.cc:323, 421, 454`
