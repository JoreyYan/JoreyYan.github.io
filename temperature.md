# 你一直在调温度，只是没人告诉你

### 从优化器到合成数据崩塌：一条涨落–耗散的线索

---

## 〇、核心观察

深度学习里有一个反复出现、却几乎从不被统一命名的量：**温度**。

它至少出现在四个地方，而且每个地方你都在调它：

| 位置 | 你调的旋钮 | 有效温度 |
|---|---|---|
| 优化器 | 学习率 $\eta$、batch size $B$ | $T_{\rm opt}=\eta/B$ |
| 解码 | temperature、top-$p$、best-of-$n$ | $T_{\rm dec}$ |
| RLHF / DPO | KL 系数 $\lambda$ | $T_{\rm rl}=\lambda$ |
| 数据回灌 | 用什么分布生成合成数据 | $T_{\rm eff}$ |

统计物理里，温度不是一个自由参数——它被**涨落–耗散定理**（FDT）钉死在噪声和耗散的比值上：

$$\boxed{\;\text{噪声强度}\;\propto\;\text{摩擦}\times\text{温度}\;}$$

违反它，系统就不会收敛到你以为的那个分布。

而深度学习里这四个温度**互相独立、可以任意配**，没有任何一致性条件把它们绑在一起。本文要说的是：这不是无害的自由度，它是若干个已知现象的共同病根——implicit bias、exposure bias、model collapse。

下面从最具体的开始。

---

## 一、优化器：SGD 字面上就是一个 Langevin 过程

这一层不是类比，是真的。

SGD 的更新是

$$\theta_{k+1}=\theta_k-\eta\,\hat g_k,\qquad \hat g_k=\nabla L(\theta_k)+\xi_k$$

minibatch 梯度等于全梯度加噪声，噪声协方差随 batch size 反比缩小：

$$\mathbb E[\xi_k]=0,\qquad \mathrm{Cov}(\xi_k)=\frac{C(\theta_k)}{B}$$

取连续时间极限，得到一个 SDE：

$$\boxed{\;\mathrm d\theta=-\nabla L(\theta)\,\mathrm dt+\sqrt{\frac{\eta}{B}}\;C(\theta)^{1/2}\,\mathrm dW\;}$$

对照过阻尼 Langevin 方程 $\mathrm d\theta=-\nabla U\,\mathrm dt+\sqrt{2k_BT}\,\mathrm dW$，立刻读出：

$$L\;\leftrightarrow\;U\ (\text{势能}),\qquad \boxed{\;T_{\rm opt}=\frac{\eta}{B}\;}$$

**学习率除以 batch size 就是训练过程的温度。**

这不是修辞。你天天在用的 **linear scaling rule**（$B$ 翻倍则 $\eta$ 翻倍）——它的全部内容就是：**保持 $\eta/B$ 不变，即保持温度不变。** 换 batch size 不改学习率，你改的是系统温度，模型自然表现不同。

### FDT 在这里问什么

Fokker–Planck 方程的稳态解，只有在噪声各向同性时才是 Gibbs 分布：

$$C(\theta)=c\,I\quad\Longrightarrow\quad p_s(\theta)\;\propto\;\exp\!\Big(-\frac{2L(\theta)}{c\,T_{\rm opt}}\Big)$$

即"在损失面上按温度 $\eta/B$ 采样"。这是 FDT 满足时的理想图景。

但真实的梯度噪声协方差是**强各向异性、且依赖 $\theta$ 的**。对交叉熵损失，在极小值附近有

$$C(\theta)\;\approx\;F(\theta)\;\approx\;\nabla^2 L(\theta)$$

（Fisher 信息 ≈ Hessian）。这意味着**噪声沿陡峭方向强、沿平坦方向弱**——和各向同性差得很远。

后果是细致平衡破缺：

$$J_s(\theta)\;=\;-p_s\nabla L-\tfrac{\eta}{2B}\nabla\!\cdot\!\big(C\,p_s\big)\;\neq\;0$$

稳态存在**非零概率流**，稳态分布**不是** $e^{-2L/T}$。SGD 收敛到的甚至不是 $L$ 的极小点，而是一个被噪声结构改写过的有效势的极小点。

**所以：implicit bias 就是 FDT 破缺的产物。** SGD 偏好平坦极小值，不是因为损失面偏好平坦，是因为噪声在陡峭方向更强、把系统从窄谷里踢出去。这是一个可以写下来的机制，不是玄学。

**Adam 让情况更远离平衡。** 加了预条件子 $P=\mathrm{diag}\big(1/(\sqrt{v}+\epsilon)\big)$：

$$\mathrm d\theta=-P\nabla L\,\mathrm dt+\sqrt{\tfrac{\eta}{B}}\;P\,C^{1/2}\,\mathrm dW$$

要让稳态回到 $e^{-2L/T}$，FDT 要求噪声协方差正比于**漂移的预条件子** $P$；而实际得到的是 $PCP^\top$。两者不匹配，且 $P$ 本身随训练变化。**这就是为什么 Adam 的隐式偏置至今没有干净的理论刻画——它的稳态根本不是任何一个 Gibbs 测度。**

---

## 二、采样侧：第二个温度，和训练侧毫无关系

模型是在 $T=1$ 下用交叉熵拟合数据分布的：

$$\theta^*=\arg\min\ \mathbb E_{x\sim p^*}\big[-\log p_\theta(x)\big]\quad\Longrightarrow\quad p_\theta\to p^*$$

但推理时你套一层解码：

$$q(x)\;\propto\;p_\theta(x)^{1/T_{\rm dec}}$$

$T_{\rm dec}<1$ 就是变冷。top-$k$/top-$p$ 是更极端的版本（截断相当于 $T_{\rm dec}\to0$ 作用在尾部）；best-of-$n$ 用奖励模型打分选最好的，等价于一次指数倾斜。

用物理语言说：**这是"学好的势 + 随手配的热浴"。** $U_{\rm eff}=-\log p_\theta$ 是从数据学来的，热浴温度 $T_{\rm dec}$ 是你手调的，两者之间**没有任何一致性条件**。

后果就是我们熟悉的两端失效：

- $T_{\rm dec}\to0$（greedy）：无噪声的确定性梯度下降 → 卡在局部极小 → 重复循环
- $T_{\rm dec}>1$：噪声过强 → 漂离数据流形 → 胡言乱语

top-$p$ / typical / min-$p$ 这一整族解码策略，本质上都是在**手工调一个本该由一致性条件给出的量**。它们全是启发式，因为那个条件在自回归框架里不存在。

**记住这个不等式，它是后面一切的起点：**

$$\boxed{\;T_{\rm dec}<1=T_{\rm train}\;}$$

我们永远在比训练更冷的条件下采样——因为冷采样的单条样本质量更高。

---

## 三、rollout：为什么单步指标会骗你

**rollout** 指让模型用自己的输出作为下一步输入，滚出整条轨迹：语言模型的自回归生成，或粗粒化力场从 $z_0$ 积分出 $z_{0:T}$。

而训练是在**真实历史**上做单步匹配（teacher forcing）。设单步误差

$$\mathbb E_{x\sim p^*}\Big[D_{\rm TV}\big(p_\theta(\cdot\mid x),\,p^*(\cdot\mid x)\big)\Big]\;\le\;\varepsilon$$

则 rollout $T$ 步后的轨迹误差不是 $O(\varepsilon T)$，而是

$$\boxed{\;D_{\rm TV}\big(p_\theta^{(0:T)},\,p^{*(0:T)}\big)=O(\varepsilon T^2)\;}$$

因为误差不只累加，还会把状态推出训练分布，在那里单步误差本身不再受 $\varepsilon$ 控制（Ross–Bagnell 的复合误差分析）。

**$\varepsilon$ 再小，$T$ 一大就被 $T^2$ 吃掉。** 这是 perplexity 与长文行为之间那道裂缝的定量来源。

顺带指出一个完全同构的现象：粗粒化分子动力学里，force matching（匹配参考轨迹上的瞬时受力）拟合得很好，自由能面也对，但 rollout 出来的弛豫时间和转移速率能差几个数量级。同一个病，同一个数学。

而**合成数据，就是 rollout 的产物。**

---

## 四、闭环：把上面几件事乘起来

现在做核心推导。自训练循环是

$$p_n\;\xrightarrow{\ \text{采样}\ }\;\{x_i\}_{i=1}^N\sim q_n\;\xrightarrow{\ \text{拟合}\ }\;p_{n+1}$$

由第二节，$q_n\propto p_n^{\beta}$，其中

$$\beta=\frac{1}{T_{\rm dec}}>1$$

拟合用交叉熵，无限样本下就是矩匹配，$p_{n+1}=q_n$。于是得到一个确定性映射：

$$\boxed{\;p_{n+1}=\frac{p_n^{\beta}}{Z_n}\;}\qquad\Longrightarrow\qquad p_n=\frac{p_0^{\,\beta^{\,n}}}{Z}$$

**指数上是 $\beta^n$，不是 $n\beta$。**

高斯情形有闭式解。$p_0=\mathcal N(\mu,\sigma_0^2)$ 时，$p_0^\beta\propto\mathcal N(\mu,\sigma_0^2/\beta)$，所以

$$\boxed{\;\sigma_n^2=\sigma_0^2\,\beta^{-n}\;},\qquad \mathbb H[p_n]=\mathbb H[p_0]-\frac n2\log\beta$$

**熵随代数线性下降，斜率 $\tfrac12\log\beta$，斜率由你的解码温度决定。** 两三代就能拟合出这条直线并外推——你不需要跑到崩塌才知道会崩。

### 有限采样只是第二个因子

主流叙事把 model collapse 归给有限样本。把它也算进来：每代 $N$ 个样本、MLE 估计方差，$\mathbb E[\hat\sigma_{n+1}^2]=\sigma_n^2\cdot\frac{N-1}{N}$。两个机制相乘：

$$\boxed{\;\sigma_n^2=\sigma_0^2\left[\frac{N-1}{N\beta}\right]^{n}\;}$$

现在看极限：

$$N\to\infty\quad\Longrightarrow\quad \sigma_n^2\to\sigma_0^2\,\beta^{-n}\to0$$

**温度项完全不依赖 $N$。采样再多也救不了。**

这就是这个视角的增量：崩塌不是统计涨落的意外累积，而是"采样比训练更冷"时的**必然方向**。判据是

$$\text{崩塌}\iff\beta>\frac{N-1}{N}\approx1\iff T_{\rm dec}<T_{\rm train}$$

而这个条件我们一直在满足——因为冷采样的样本质量更高。

**质量筛选和分布收缩是同一个动作。** 这解释了为什么"提高合成数据质量"（更低温度、更强过滤、best-of-$n$）反而**加速**崩塌。

---

## 五、一个反例：diffusion 自带 FDT

对比之下这一点很亮。前向 OU 过程 $\mathrm dx=f\,\mathrm dt+g(t)\mathrm dW$ 的反向 SDE 是

$$\mathrm dx=\big[f(x)-g(t)^2\,\nabla\log p_t(x)\big]\mathrm dt+g(t)\,\mathrm d\bar W$$

注意：**score 项前面的 $g^2$ 和噪声项里的 $g$ 是同一个 $g$。** 漂移修正与噪声强度被前向/反向对偶结构性地锁死——这就是一个 FDT 型关系，不是你调出来的。

你可以改 noise schedule，但**不能独立地调噪声和 drift**：一调，采的就不再是同一个分布了。

而自回归采样里，$T_{\rm dec}$ 和 top-$p$ 是完全独立于训练目标的自由旋钮。

**这是 diffusion 在分布保真度上更稳的一个深层原因，也是离散扩散语言模型值得看的理由。**

---

## 六、RLHF：唯一能写下目标平衡分布的地方

$$\max_\pi\ \mathbb E_{y\sim\pi}[r(x,y)]-\lambda\,D_{\rm KL}\big(\pi\,\|\,\pi_{\rm ref}\big)$$

解析最优解是

$$\boxed{\;\pi^*(y\mid x)\;\propto\;\pi_{\rm ref}(y\mid x)\,e^{\,r(x,y)/\lambda}\;}$$

**这就是一个 Gibbs 分布**：$\pi_{\rm ref}$ 是先验测度，$r/\lambda$ 是 $-U/k_BT$，KL 系数 $\lambda$ 就是温度。DPO 的整套推导建立在这上面。

这是 LLM 训练里唯一一处你能写下明确目标平衡分布的地方。于是 FDT 式的问题变得有意义且可定量：**你实际的采样–更新过程，稳态是不是这个 $\pi^*$？**

on-policy / off-policy 不匹配、reward hacking、KL 估计偏差、熵坍缩——都可以重述为"采样动力学的稳态偏离了目标 Gibbs 测度"。我认为这是这条线索在 LLM 里唯一能推到定量的地方。

---

## 七、验证器就是麦克斯韦妖

第四节的映射是**幂次锐化**：方向由 $p_n$ 自己决定，无差别变窄。

引入验证器 $V$（打分 $r$），做拒绝采样或奖励加权，映射变成**指数倾斜**：

$$p_{n+1}(x)\;\propto\;p_n(x)\,e^{\,r(x)/\lambda}$$

即第六节那个 Gibbs 形式。差别是本质的：

$$\text{无验证器：}\ p_{n+1}\propto p_n^\beta\quad(\text{无差别变窄})\qquad\text{有验证器：}\ p_{n+1}\propto p_n e^{r/\lambda}\quad(\text{定向变窄})$$

无差别锐化相当于系统自发降熵，热力学上不能免费发生，代价是尾部信息死掉。定向锐化用信息换熵减，是麦克斯韦妖——但**妖必须付费**。

付费额度可以精确写出来。设 $W$ 为真实世界，马尔可夫链 $W\to(p_n,V)\to p_{n+1}$，由数据处理不等式：

$$\boxed{\;\Delta I\;=\;I(p_{n+1};W)-I(p_n;W)\;\le\;I\big(V;\,W\mid p_n\big)\;}$$

**每一轮自训练的信息增益，上界是验证器相对模型已有知识所携带的真实世界信息量。** 令 $V=\varnothing$，右端为零：闭环无法增加信息。

对照各方案：

| 方案 | 验证器 $V$ | $I(V;W\mid p_n)$ |
|---|---|---|
| 蒸馏 | 教师模型 | 大（但只是转移，非新增） |
| RLVR、单元测试 | 执行环境 | 大 |
| AlphaZero / AlphaGeometry | 规则 / 演绎引擎 | 精确 |
| AlphaFold2 自蒸馏 | pLDDT 过滤 | 弱但非零 |
| self-consistency / 多数投票 | **无** | $0$ |

最后一行值得单说：多数投票确实有效，但它没有增加 $I$，只是降低采样噪声、把 $p_n$ 里已有却被噪声掩盖的知识提取出来。天花板严格等于模型自身。

**归纳：闭环自训练可以锐化，不能扩展。** 你可以把概率质量从 pass@$k$ 搬到 pass@1——这是真实且巨大的收益，今天多数自生成数据 RL 干的就是这件事——但你搬不出 $\mathrm{supp}(p_0)$ 之外。

这也解释了领域差异：形式化证明检查、单元测试是近乎无损的验证器，$I(V;W)$ 极大，合成数据收益巨大；开放域写作、审美几乎没有验证器，$I\approx0$，只剩蒸馏，天花板就是教师。

---

## 八、可检验的推论

全部来自上面的公式：

**① 累积优于替换。** 混入比例 $\alpha$ 的真实数据，映射变成 $p_{n+1}\propto\alpha p^*+(1-\alpha)p_n^\beta$。$p^*$ 是恒温热浴，方差不再几何衰减，收敛到有界偏差而非零。

**② 解码温度是崩塌速率的控制参数。** 斜率 $\tfrac12\log\beta$ 直接由 $T_{\rm dec}$ 决定。最干净的消融实验：固定一切，只扫 $T_{\rm dec}$，测熵随代数的斜率。

**③ 监控熵和有效支撑集大小，不是 loss。** loss 是自由能型指标（由势决定），多样性坍缩是动力学型指标（由噪声决定）。前者可以一路漂亮，后者已经死了。

**④ 崩塌可外推。** 熵曲线是直线，两三代拟合斜率即可预测。它不是某代突然坏掉。

**⑤ 换 batch size 必须同步换学习率。** 保持 $\eta/B$，否则你改的是训练温度。这一条你本来就在做，只是现在知道为什么。

---

## 九、两个落点

**语言模型**：把验收标准从 perplexity 换成 rollout 统计量——长生成的熵、自条件化多代后的方差收缩率、多样性–质量 trade-off 曲线的整体位移。把 $T_{\rm dec}$ 当作**会改变下一代数据分布的系统参数**来管理，而不是推理时的旋钮。

**科学模型**（蛋白质粗粒化为例）：用 CG 模型自己的 rollout 轨迹再训 CG，是标准闭环，$\beta>1$ 成立，必然收缩——而收缩的恰恰是稀有事件尾部，转变态和亚稳态跃迁全在那里。症状极隐蔽：自由能面尚可，速率常数越训越慢。

但这里有个别人羡慕的条件：**全原子 MD 是一个 $I(V;W)$ 极大的真验证器。** 所以正确的循环不是自训练，是主动学习——CG 提议轨迹，全原子在模型最不确定处做短程验证再回灌（MSM 领域的 adaptive sampling）。很少有人从 $\Delta I\le I(V;W)$ 这个角度论证它为什么**必须**存在。

同理，AlphaFold2 自蒸馏没崩，是因为有 pLDDT 这个过滤器；无差别回灌预测结构，结果会和 model collapse 一模一样。

---

## 一句话

$$T_{\rm opt}=\frac{\eta}{B},\qquad \sigma_n^2=\sigma_0^2\Big[\frac{N-1}{N\beta}\Big]^{n},\qquad \Delta I\le I(V;W\mid p_n)$$

> 第一式：**优化器就是 Langevin，学习率除以 batch size 就是温度**，而梯度噪声的各向异性使 FDT 破缺——这就是 implicit bias。
>
> 第二式：**因为我们永远采样得比训练更冷（$\beta>1$），闭环自训练必然坍缩，而且样本量 $N$ 救不了。**
>
> 第三式：**没有验证器时信息增益为零；有验证器时，能换来的能力提升正比于它从真实世界携带的信息量。**

所以评估任何一个合成数据方案，只需问两句：

**信息从哪来（$V$ 是什么）？采样比训练冷多少（$\beta$ 多大）？**

两个都答不上来的，就是在闭环里滚雪球——而雪球会越滚越小。

---

### 延伸阅读

- [S. Ross, G. Gordon, D. Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*](https://proceedings.mlr.press/v15/ross11a.html)
- [I. Shumailov et al., *AI models collapse when trained on recursively generated data*](https://www.nature.com/articles/s41586-024-07566-y)
- [Y. Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*](https://arxiv.org/abs/2011.13456)
