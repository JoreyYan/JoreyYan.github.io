# 一种会遗忘的 Transformer

### 从相关性的产生、传播与粗粒化，到门控线性注意力这个极简特例

---

## 〇、问题不是要不要 Transformer，而是它该怎样遗忘

上一篇文章从邓煜对玻尔兹曼方程的工作出发，讨论了一个普遍问题：一个微观系统会不断通过相互作用产生 correlation，但宏观理论不可能保留全部微观历史。它必须回答：

$$\boxed{\text{哪些 correlation 重要，哪些可以遗忘？}}$$

把这个问题放到机器学习里，最直接的想法不是在 Transformer 外面再加一个独立的 interaction、history、closure、cumulant 或 slow variable 模块。

更统一的理解是：

$$\boxed{\text{Attention 是微观相互作用的计算形式，Transformer 是 correlation 演化的统一载体。}}$$

邓煜式的思想应该体现在 Transformer 内部：

> correlation 如何产生、如何传播、如何被压缩，以及哪些模式能够跨越很长时间继续存在。

普通 Transformer 已经很擅长产生和传播 correlation，却没有明确解决它们的耗散与保留。它通常把原始 token 全部留在上下文里，让后续层随时回头读取。

所以真正要构造的，不是一个更大的上下文，而是一种能够自动完成

$$\boxed{\text{correlation 产生}\;+\;\text{筛选}\;+\;\text{粗粒化}\;+\;\text{慢变量演化}}$$

的 Transformer。

下面先把这套结构从头建立起来。到最后再看会发现：门控线性注意力已经实现了它的一个极简特例。

---

## 一、Attention 本身就是微观相互作用

令一组 token 的状态为

$$X_t=[x_1^t,\ldots,x_N^t]\in\mathbb R^{N\times d}.$$

Attention 权重为

$$A_{ij}^t=\operatorname{softmax}_j\left(\frac{q_i^{t\top}k_j^t}{\sqrt d}\right),$$

token 的更新为

$$x_i^{t+1}=x_i^t+\sum_jA_{ij}^tVx_j^t.$$

这可以直接解释成

$$\boxed{A_{ij}^t=\text{微观单元 }i,j\text{ 的相互作用强度}},$$

以及

$$\boxed{Vx_j^t=\text{这次相互作用给 }i\text{ 带来的状态变化}}.$$

因此，一层 Attention 就是一次所有 token 之间的软碰撞。多头 Attention 则提供多种相互作用通道：身份、空间、因果、目标和约束可以沿不同的 head 传播。

不一定需要给每个 head 人工命名，但它们在数学上就是多个 interaction kernel。

---

## 二、深度传播的是高阶 correlation

第一层里，

$$x_1\leftrightarrow x_2,\qquad x_2\leftrightarrow x_3$$

分别产生局部的二体 correlation。第二层以后，$x_1$ 会通过 $x_2$ 间接获得 $x_3$ 的信息，于是三个 token 之间形成更高阶的依赖。

经过 $L$ 层，

$$X^L=F_L\circ\cdots\circ F_1(X^0),$$

每个 token 都会包含大量 interaction path：

$$i\rightarrow j\rightarrow k\rightarrow\cdots.$$

从这个角度看，

$$\boxed{\text{attention paths}\approx\text{collision histories}.}$$

Attention 产生微观 correlation，深度负责传播并组合它们。Transformer 本身已经是一个 correlation 演化系统，不需要再发明一套笨重的图结构重复实现这件事。

真正缺少的是下一步：**哪些 correlation 应该衰减，哪些应该留下来成为宏观状态？**

---

## 三、普通 Transformer 缺少的是耗散

标准 Transformer 的基本逻辑是

$$\text{更多 token}+\text{更多层}\longrightarrow\text{更复杂的 correlation}.$$

但自然系统通常不是这样演化的。更常见的过程是

$$\text{产生大量 correlation}\longrightarrow\text{绝大多数快速衰减}\longrightarrow\text{只留下少数慢模态}.$$

如果所有历史 token 一直保留，模型并不需要判断哪些信息真正重要。它可以把上下文当作一个不断增长的外部存储，需要时再回头查找。

因此我们需要的不是单纯增加 context length，而是把 Transformer 改造成

$$\boxed{\textbf{能够自动产生、筛选并粗粒化 correlation 的 Transformer。}}$$

它必须同时具有两类状态：

- 短期存在、负责当前局部计算的微观状态；
- 跨越时间持续存在、负责承载宏观信息的慢状态。

---

## 四、统一结构：微观 token 与慢 token

令整个系统的状态为

$$\boxed{Y_t=[X_t;Z_t].}$$

其中

$$X_t\in\mathbb R^{N\times d}$$

是微观 token，代表当前文字、当前观察、具体事件和局部推理的中间状态。

而

$$Z_t\in\mathbb R^{K\times d},\qquad K\ll N$$

是慢变量 token，代表系统从大量微观 interaction 中提炼出来的宏观状态，例如：

- 实体身份；
- 当前世界状态；
- 目标与约束；
- 长期因果关系；
- 未完成的计划；
- 当前的不确定性。

整个系统仍然可以只使用一个 Transformer：

$$Y_{t+1}=Y_t+\operatorname{Transformer}_\theta(Y_t).$$

但它的 Attention 矩阵会自然分成四块：

$$A_t=\begin{bmatrix}A_{XX}&A_{XZ}\\A_{ZX}&A_{ZZ}\end{bmatrix}.$$

原来被拆开的 interaction、粗粒化、反馈和宏观演化，都可以放进这一个 block attention matrix 里。

---

## 五、四块 Attention 分别做什么

### $A_{XX}$：微观 interaction

$$X\rightarrow X.$$

当前 token、事件和观察互相作用，产生大量短期 correlation。这是系统的“碰撞过程”。

### $A_{ZX}$：从微观状态中提炼慢变量

这里 Query 来自 $Z$，Key 和 Value 来自 $X$：

$$Z\leftarrow X,$$

$$Z_{t+1}=Z_t+\operatorname{Attn}(Q=Z_t,K=X_t,V=X_t).$$

慢 token 主动读取当前微观状态，回答：

$$\boxed{\text{哪些微观 correlation 应该被写入宏观状态？}}$$

这就是粗粒化。

### $A_{XZ}$：宏观状态反馈微观计算

$$X\leftarrow Z.$$

当前目标、身份、世界规则和长期约束反过来影响局部理解与推理。

例如同一句“他答应了”，只有结合慢状态，模型才知道“他”是谁、答应了什么，以及这个承诺是否改变了后续计划。

### $A_{ZZ}$：慢变量自身演化

$$Z\rightarrow Z.$$

世界状态、目标、关系和计划之间继续发生相互作用，形成宏观动力学：

$$\boxed{Z_{t+1}=G_\theta(Z_t,\text{new evidence}).}$$

因此，一个统一的 Attention 系统已经可以包含

$$\boxed{\text{微观碰撞}+\text{粗粒化}+\text{宏观反馈}+\text{宏观演化}.}$$

---

## 六、真正关键的一步：微观 token 必须被删除

如果当前 chunk 处理结束后，模型仍然保留全部 $X_t$，那么 $Z_t$ 不需要真正承载世界状态。模型仍然可以偷懒，回头读取原始 token。

所以每个时间段或 chunk 结束后，必须执行

$$\boxed{X_t\longrightarrow\varnothing}$$

或者只留下一个很小的 working buffer。真正跨时间保留的只有

$$\boxed{Z_t.}$$

下一段输入 $O_{t+1}$ 到来时，

$$X_{t+1}^{0}=\operatorname{Embed}(O_{t+1}),$$

再与上一段留下的慢状态拼接：

$$Y_{t+1}^{0}=[X_{t+1}^{0};Z_t].$$

运行同一个 Transformer 后，

$$[X_t;Z_t]\longrightarrow[X_t';Z_{t+1}],\qquad X_t'\longrightarrow\varnothing.$$

于是整个模型按 chunk 循环：

```python
def process_chunk(tokens, slow_state):
    micro = embed(tokens)
    state = concat(micro, slow_state)

    for _ in range(num_steps):
        state = shared_transformer_block(state)

    micro, slow_state = split(state)
    output = readout(micro, slow_state)

    # micro state is discarded
    return output, slow_state
```

模型主体仍然是 Transformer。真正改变的只有三件事：

$$\boxed{\text{共享迭代}+\text{快慢 token 分区}+\text{强制删除微观状态}.}$$

这会把 Transformer 从一个保留全部上下文的序列处理器，改造成一个有耗散的多尺度动力系统。

---

## 七、快变量与慢变量最终是一个谱问题

设共享的更新规律为

$$Y_{n+1}=F_\theta(Y_n).$$

在局部状态附近线性化：

$$\delta Y_{n+1}\approx J_\theta\delta Y_n,\qquad J_\theta=\frac{\partial F_\theta}{\partial Y}.$$

对 Jacobian 做特征分解：

$$J_\theta v_k=\lambda_kv_k.$$

如果

$$|\lambda_k|\ll1,$$

对应的 correlation mode 会很快衰减：

$$\lambda_k^n\rightarrow0.$$

它是快变量，可以被遗忘。

如果

$$|\lambda_k|\approx1,$$

这个模式会持续很久，是慢变量；当 $\lambda_k=1$ 时，它甚至是一个守恒模态。

因此，$Z_t$ 应该承载接近 $|\lambda_k|=1$ 的模式，而 $X_t$ 主要承载 $|\lambda_k|<1$、会迅速耗散的模式。

这时“什么重要、什么可以忽略”就不再只是一句语言描述，而变成一个动力系统的谱问题：

$$\boxed{\text{少量慢模接近单位圆，大量快模位于单位圆内部。}}$$

为了让 $F_\theta$ 真正像一条反复作用的动力学规律，各次迭代还应该共享参数：

$$Y^{l+1}=F_\theta(Y^l).$$

它不再是 $F_1,F_2,\ldots,F_L$ 组成的一条固定流水线，而是同一个 interaction law 在系统上重复运行。

---

## 八、只有架构还不够：训练目标必须定义什么是慢变量

即使加入 $Z_t$ 并删除 $X_t$，也没有任何保证说 $Z_t$ 一定会学成真正的宏观状态。模型可能只把它当作一个懒惰的摘要。

因此训练目标必须直接约束 correlation 的存亡。

### 预测充分性

只给慢状态 $Z_t$，要求预测较远的未来：

$$\mathcal L_{\rm future}=-\log p_\theta(O_{t+\tau}\mid Z_t,A_{t:t+\tau}).$$

这样，真正影响长期未来的 correlation 才必须进入 $Z_t$。

### 压缩

限制

$$K\ll N,$$

并对 $Z_t$ 加噪、量化或信息瓶颈，使模型不能把全部原始 token 原封不动塞进慢状态。

### 闭合

让宏观状态自己 rollout：

$$\hat Z_{t+\tau}=G_\theta^\tau(Z_t),$$

并与未来真实输入压缩得到的 $Z_{t+\tau}$ 匹配：

$$\mathcal L_{\rm closure}=\left\|\hat Z_{t+\tau}-Z_{t+\tau}\right\|^2.$$

这要求慢状态能够近似独立演化。

### 微观扰动不变性

对输入做改写、换序、插入无关信息或更换表达方式时，要求

$$Z_t^{(1)}\approx Z_t^{(2)}.$$

这些变化只影响微观表达，不应该改变宏观状态。

### 关键干预敏感性

如果改变的是目标、身份、约束或因果事实，则要求

$$Z_t^{(1)}\neq Z_t^{(2)}.$$

慢状态既不能什么都记，也不能什么都忽略。它应该对微观扰动稳定，却对真正改变未来的干预敏感。

这组目标才给出了“什么应该进入慢变量”的可测定义。

---

## 九、到这里再看门控线性注意力

现在再比较 Gated Linear Attention，就能看见它在整套结构里的位置。

门控线性注意力可以写成一个矩阵状态递归：

$$\text{GLA:}\qquad S_t=\alpha_tS_{t-1}+v_tk_t^\top,\qquad S\in\mathbb R^{d\times d}.$$

Renormalizing Transformer 的慢状态更新是

$$\text{RT:}\qquad Z_{t+1}=Z_t+\operatorname{Attn}(Q=Z_t,K=X_t,V=X_t),\qquad Z\in\mathbb R^{K\times d}.$$

二者共享三件事：

$$\boxed{\text{固定大小的持久状态}+\text{丢弃原始 token}+\text{遗忘机制}.}$$

而且在最简单的标量门控形式里，有一处对应可以精确写成等号。

GLA 对旧状态的转移是

$$S_t=\alpha_tS_{t-1}+\cdots,$$

所以局部状态转移的特征值就是

$$\boxed{\lambda=\alpha_t.}$$

当 $\alpha_t\rightarrow1$，对应模式被长期保留；当 $\alpha_t\rightarrow0$，旧状态立即衰减。

也就是说：

> **门控线性注意力已经在直接学习上一节要求的时间尺度谱，只不过在这个极简形式里，所有方向共享同一个 $\lambda$。**

Gated DeltaNet 再向前一步：

$$S_t=S_{t-1}\,\alpha_t\left(I-\beta_tk_tk_t^\top\right)+\beta_tv_tk_t^\top.$$

它在统一衰减之外增加一个秩一方向，可以沿当前 key 定向擦除并覆写旧关联。但每一步额外获得的仍然只有一个自由的秩一方向。

---

## 十、GLA 与 Renormalizing Transformer 的实质差别

| | 状态 | 慢状态内部演化 | 从当前输入写入 | chunk 内交互 |
|---|---|---|---|---|
| 标量门控线性注意力 | $S\in\mathbb R^{d\times d}$ | 标量 $\alpha_t$ | 外积 $v_tk_t^\top$ | 无 |
| Gated DeltaNet | 同上 | 标量衰减 + 秩一修正 | 定向擦除后写入 | 无 |
| Renormalizing Transformer | $Z\in\mathbb R^{K\times d}$ | 完整 attention | $Z$ 主动 query $X$ | 有 |

三处差别最重要。

### 1. 慢变量之间可以相互作用

GLA 的状态主要是一组历史外积的加权和，内部没有完整的宏观动力学。RT 中的 $A_{ZZ}$ 允许身份、目标、世界状态和计划彼此作用。

这里更严格地说，真正决定慢模谱的是完整更新映射的 Jacobian

$$J_{ZZ}=\frac{\partial Z_{t+1}}{\partial Z_t},$$

而不是 Attention 权重矩阵本身。$A_{ZZ}$ 提供完整耦合，$J_{ZZ}$ 才描述局部动力学。

### 2. 写入是 pull，而不是 push

GLA 中每个 token 都产生一个外积写入；RT 中是 $Z$ 用自己的 query 主动读取 $X$。

前者是“每个 token 都塞一份”，后者是“当前状态决定自己还需要什么”。

### 3. RT 按 chunk 粗粒化

RT 在 chunk 内保留 $A_{XX}$ 的全注意力，只在跨 chunk 时强制压缩进 $Z$。因此它同时保留局部精确计算和长期固定状态。

当 $K\ll N$ 时，$A_{ZZ}$ 的 $K\times K$ 代价通常是可承受的。

所以可以把关系概括为：

$$\boxed{\text{门控线性注意力是这套慢变量动力学的一个极简特例。}}$$

它已经实现了谱选择和固定状态，但没有实现完整的慢变量相互作用、主动吸收以及 chunk 内微观动力学。

---

## 十一、架构前作与真正可能新的部分

只看“latent state + cross-attention + 跨 chunk 传递 + 丢弃旧 token”这个结构，前作已经很多：

- [Perceiver / Perceiver IO](https://arxiv.org/abs/2103.03206)：latent array 读取输入，再做 latent self-attention，对应 $A_{ZX}$ 与 $A_{ZZ}$；
- [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881)：memory token 跨 segment 传递，是最直接的结构前作；
- [Block-Recurrent Transformer](https://arxiv.org/abs/2203.07852)：state vector 与 token block 通过 self-attention 和 cross-attention 递归更新；
- [Infini-attention](https://arxiv.org/abs/2404.07143)：压缩长期记忆与局部注意力并存；
- [Titans](https://arxiv.org/abs/2501.00663)：短期 attention 与长期神经记忆结合；
- Compressive Transformer、Transformer-XL、Universal Transformer 也分别包含压缩记忆、跨段递归或共享权重迭代。

因此，架构本身更像一条已有路线的统一和加强，而不是从零出现的新结构。

这些模型面对的困难也很具体：

1. 跨 chunk 的梯度必须穿过反复压缩的状态，训练上仍然有 BPTT 的困难；
2. 丢弃 $X$ 后无法回头逐字查找，精确回忆任务容易受损；
3. 普通语言模型目标没有强制 $Z$ 成为对未来充分且近似闭合的慢变量。

所以真正值得强调的不是“又一种 memory architecture”，而是上一节的训练目标：

$$\boxed{\mathcal L_{\rm future}+\mathcal L_{\rm closure}+\text{微观不变性}+\text{关键干预敏感性}.}$$

它们直接回答了过去的 memory token 为什么可能只学成懒惰摘要，以及怎样测量一个状态是否真的承载了该进入慢变量的信息。

---

## 十二、一个必须保留的技术警告

要求某些模式满足

$$|\lambda_k|\approx1$$

正好位于循环网络梯度消失与爆炸的临界区域：

$$|\lambda|<1\Rightarrow\text{梯度逐渐消失},\qquad |\lambda|>1\Rightarrow\text{梯度逐渐爆炸}.$$

LSTM 的 forget gate、GRU 以及门控线性注意力里的 $\alpha_t$，都在尝试让需要长期保留的模式稳定地停在 1 附近。

标量 gate 可以天然限制在 $(0,1)$。RT 把状态内部演化推广成完整的 attention 耦合后，$J_{ZZ}$ 的谱半径不再自动受控。

这不是一个附带问题，而是实现这套结构时最先会遇到的困难：

> **我们既要让少数慢模足够接近 1，又不能让完整的宏观动力学越过稳定边界。**

因此，第七节的谱要求不是一个孤立的新设计。它正是门控机制长期存在的原因；RT 所做的，是把这个问题从简单 gate 推广到完整的慢状态动力学。

---

## 结语

整套构造可以压缩成一句话：

> **Attention 产生 correlation，慢 token 保留重要模式，共享迭代形成动力学，遗忘完成粗粒化。**

Transformer 不再只是不断把 token 变得更复杂，而是经历一个完整的微观—宏观过程：

$$\text{微观 token}\rightarrow\text{Attention 产生 correlation}\rightarrow\text{慢 token 吸收重要模式}\rightarrow\text{快速 correlation 耗散}\rightarrow\text{微观 token 被删除}.$$

门控线性注意力并不是与它无关的另一类模型。恰恰相反，它已经实现了这条路线最核心的极简版本：

$$\boxed{\lambda=\alpha_t.}$$

它用 gate 学习一个 correlation 应该衰减多快。Renormalizing Transformer 则试图把这个标量或结构化衰减，推广成一组能够主动吸收信息、彼此作用并独立演化的慢变量。

所以最后的判断是：

> **架构上，这是一条已有前作的加强路线；真正需要被建立的，是一组能够迫使慢状态学会“什么该留下、什么该耗散”的训练目标。**

这才是从门控线性注意力走向 Renormalizing Transformer 时，最值得保留的理论增量。

---

### 延伸阅读

- [Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient Training*](https://arxiv.org/abs/2312.06635)
- [Yang, Kautz & Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*](https://arxiv.org/abs/2412.06464)
- [Jaegle et al., *Perceiver: General Perception with Iterative Attention*](https://arxiv.org/abs/2103.03206)
- [Bulatov, Kuratov & Burtsev, *Recurrent Memory Transformer*](https://arxiv.org/abs/2207.06881)
- [Hutchins et al., *Block-Recurrent Transformers*](https://arxiv.org/abs/2203.07852)
- [Munkhdalai, Faruqui & Gopal, *Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention*](https://arxiv.org/abs/2404.07143)
- [Behrouz, Zhong & Mirrokni, *Titans: Learning to Memorize at Test Time*](https://arxiv.org/abs/2501.00663)
