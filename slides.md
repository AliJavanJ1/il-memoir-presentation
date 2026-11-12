---
theme: default
title: Imitation by occupancy matching and forward–backward representations
info: Master's memoir — Ali Javani, supervised by Olivier Cappé, October 2026
layout: cover
transition: slide-left
---

# Imitation by occupancy matching and by forward–backward representations

Proofs, approximations and a controlled comparison of f-PG and FB imitation

Ali Javani · supervised by Olivier Cappé · Master's memoir · October 2026

<!--
⏱ 0:20 (total 0:20)

Hello everyone, and thank you for coming. My presentation 
is about imitation learning: how an agent can learn a behavior from an expert when nobody gives it a reward. I'm going to compare two families of methods.
-->

---
layout: two-cols-header
---

# Our Scope vs Imitation Learning

::left::

- **Setting more specifically**: **state-only** demonstrations — **no reward**, and also **no expert actions** (Only where the expert goes)

<v-clicks>

- **Our scope**:
  1. **Direct occupancy matching using f-divergance**: push the learner's state distribution toward the expert's *online*
  2. Imitation via **Forward–backward (FB)** foundational model: pretrain a reward-free world model *once*, then imitate *offline* with zero new interaction

</v-clicks>

::right::

<img src="/fig1-four-rooms-goal55.png" class="h-72 mx-auto" />

State-only imitation: reach the same states as the expert and stay there

<!--
⏱ 1:25 (total 1:45)

More specifically in our setting, we use state-only demonstrations - no reward and also no expert actions.
For example in the right orange trajectory from our experiments, we can only see where the expert goes.

[click] Our scope can be divided into these two families: ...
-->

---

##  Source Papers & Imitation Methods

<v-clicks>

| Family | Paper | What it learns | In this memoir |
|---|---|---|---|
| f-divergence matching | f-IRL (Ni et al., 2020) | a reward, then a MaxEnt policy | theory only |
| | f-PG (Agarwal et al., 2023) | the policy itself, faster | implemented|
| Forward–backward | FB (Touati & Ollivier, 2021) | where each policy goes | implemented (pretraining) |
| | ER-FB, DM-FB (Pirotta et al., 2024) | only needs a task vector $z$ | implemented |

</v-clicks>

<!--
⏱ 0:30 (total 2:15)

Here are the four methods.

[click] f-IRL learns a reward. I study it in theory only.

[click] f-PG changes the policy directly.

[click] FB learns a model of where every policy goes.

[click] ER-FB and DM-FB use that model to imitate, by choosing only sixteen numbers.
-->

---

# Presentation Structure -> Answer 5 Questions

<v-clicks>

1. **Q1**: Unify notations and reprove each method's core result. Show what is exact, what is approximate.
2. **Q2 · Comparison and Use Cases**
3. **Q3 · Hyperparameters**: which knobs matter, and why?
4. **Q4 · Finite state space (maze)**: how do they perform when everything can be computed exactly?
5. **Q5 · Continuous maze**: what happens with a DQN-style network?

</v-clicks>

✅ exact statement  ·  ⚠️ approximation added by an algorithm  ·  ⭐ tricky step of a proof

<!--
⏱ 0:25 (total 2:40)

What will this talk answer? The five questions of my assignment.

[click] The mathematics, exact versus approximate.

[click] The comparison and use cases.

[click] The hyperparameters.

[click] A finite maze, where everything is exact.

[click] And a continuous maze with a DQN-style network.

On the slides: a check is exact, a warning sign is an approximation, a star is a tricky proof step.
-->

---
layout: two-cols-header
transition: slide-up # next slide goes "down" into the same question
---

# Q1. Some Important Mathematical Objects

::left::

<v-clicks>

- Maze MDP (Experiments): states $s$, 5 actions, kernel $P(s' \mid s,a)$, start cell $s_0$, $\gamma = 0.95$, horizon $H = 100$
- **Occupancy** = discounted share of time a policy $\pi$ spends per cell, as a probability over cells:
  $$d_\pi(x) = \tfrac{1}{Z_H}\textstyle\sum_{t \lt H} \gamma^t \Pr_\pi(s_{t+1} = x), \quad Z_H = \textstyle\sum_{t \lt H}\gamma^t$$

</v-clicks>

::right::

<img src="/heat-occupancy.png" class="h-90 mx-auto" />

<!--
⏱ 0:35 (total 3:15)

Which few objects do we need? I unified the notation of the four papers around very few objects.

[click] A maze MDP: cells, five actions, transitions P, one start cell, discount 0.95, horizon 100.

[click] The occupancy of a policy: the discounted share of time it spends in each cell. It is a probability. On the right, the expert spends most of its time at the goal.
-->

---
layout: two-cols-header
---

## Successor measure: where π goes from any start

::left::

<v-clicks>

- **Definition**, for any start state and action:
  $$M^\pi(s,a,x) = \textstyle\sum_{t \ge 0} \gamma^t \Pr_\pi(s_{t+1} = x \mid s_0 = s, a_0 = a)$$
  total mass $1/(1-\gamma) = 20$
- Finite maze: matrix (from solving bellman eq)
  $$M^\pi(s,a,\cdot) = P(\cdot \mid s,a)\,(I - \gamma P_\pi)^{-1}$$
- **The link between the two families**: occupancy = successor measure from the start
  $$d_\pi \approx (1-\gamma)\, M^\pi(s_0,\cdot) \qquad (\text{error} \le \gamma^H \approx 0.006)$$

</v-clicks>

::right::

<img src="/heat-successor.png" class="h-90 mx-auto" />

<!--
⏱ 0:35 (total 3:50)

The second object is the successor measure.

[click] It says where the policy goes from any start state and action. On the right, from the orange cell, most of the future mass is again at the goal. Its total mass is 20.

[click] In a finite maze it is just a matrix: the next-state row times the inverse of I minus gamma P-pi.

[click] And the occupancy is simply the successor measure from the start cell, times one minus gamma. This link connects the two families.
-->

---

# Can we imitate by matching *where* the expert goes?

<v-clicks>

- ✅ **Return identity.** For every state reward $r$ (read at the next state):
  $$J_H(\pi) = \mathbb{E}_\pi\Big[\textstyle\sum_{t \lt H}\gamma^t r(s_{t+1})\Big] = Z_H \textstyle\sum_x d_\pi(x)\, r(x)$$
- **Proof.** Linearity of expectation, then group the terms cell by cell:
  $$\textstyle\sum_{t}\gamma^t\,\mathbb{E}\,r(s_{t+1}) = \sum_t \gamma^t \sum_x \Pr(s_{t+1}=x)\,r(x) = Z_H \sum_x r(x)\,\underbrace{\tfrac{1}{Z_H}\sum_t \gamma^t \Pr(s_{t+1}=x)}_{d_\pi(x)}$$
- **So** $\ d_\pi = d_{\pi_E} \iff J_H(\pi) = J_H(\pi_E)$ for **every** reward $r$ (take $r = \mathbf{1}_{x}$ for "$\Leftarrow$")
- ⚠️ **Limit**: No guarantee on actions similarity: In a corner, "move into the wall" = "stay".

</v-clicks>

<!--
⏱ 0:45 (total 4:35)

Why is it enough to match where the expert goes?

[click] Because of this identity: the return of any policy, for any state reward, is the reward averaged against the occupancy, times a constant.

[click] The proof is one line. Expectation is linear, so I write each expected reward as a sum over cells, then group the terms by cell. Under the brace is exactly the occupancy.

[click] So if the learner has the expert's occupancy, it has the expert's return for every state reward, including the unknown one.

[click] But it cannot see actions with the same effect. In a corner, moving into the wall equals staying. So I never claim to recover actions.
-->

---
transition: slide-up # next slide goes "down" into the same question
---

# Can a policy gradient descend the divergence? (f-PG)

<v-clicks>

- **Objective** $\ J(\phi) = D_f(d_\phi \Vert q)$, $q$ = expert occupancy. **Signal** $\ g_\phi(x) = f'\big(d_\phi(x)/q(x)\big)$, **signal-to-go** $\ G_k = \sum_{t=k}^{H-1}\gamma^{t-k} g_\phi(s_{t+1})$
- ✅ **Theorem (exact gradient -> policy gradient)**, with the score $\chi_k = \nabla_\phi \log \pi_\phi(a_k \mid s_k)$, so a REINFORCE with a **cost** $g$ (only diff $\gamma^k$):
  $$\nabla J(\phi) = \frac{1}{Z_H}\,\mathbb{E}_\phi\Big[\sum_{k=0}^{H-1} \gamma^k\, \chi_k\, G_k\Big]$$
</v-clicks>

- in experiment: gradient decent -> visits move from red cells to blue ones
  <img src="/heat-fpg-signal.png" class="h-44 mx-auto mt-1" />

<!--
⏱ 0:45 (total 5:20)

Can a policy gradient descend the divergence directly? This is the f-PG idea.

[click] The objective is the divergence between the learner's occupancy and the expert's. Two objects appear: the signal g, f-prime of the ratio learner over expert in each cell, and the signal-to-go G, its discounted future sum.

[click] The theorem: the gradient is exactly a policy gradient, each score weighted by gamma to the k and by the signal-to-go. I derived it for a fixed expert target.

[click] So it is REINFORCE with a cost g. In the picture, the uniform learner visits the red cells too much and the blue path too little. Descent moves visits from red to blue.
-->

---
transition: slide-up # next slide goes "down" into the same question
---

## Proof of the exact gradient (1/2)

<v-clicks>

1. **Likelihood ratio.** Only the policy terms of $\Pr_\phi(\tau) = \mu_0(s_0)\prod_k \pi_\phi(a_k \mid s_k)\,P(s_{k+1}\mid s_k,a_k)$ depend on $\phi$, so for any $h(\tau)$:
   $$\nabla_\phi\, \mathbb{E}_\phi[h] = \mathbb{E}_\phi\Big[h \textstyle\sum_{k \lt H} \chi_k\Big]$$
2. ⭐ **Causality.** Given the past up to $s_k$, the score has mean zero:
   $$\mathbb{E}[\chi_k \mid s_0, a_0, \dots, s_k] = \textstyle\sum_a \pi_\phi(a\mid s_k)\,\nabla \log \pi_\phi(a \mid s_k) = \nabla \sum_a \pi_\phi(a \mid s_k) = \nabla 1 = 0$$
   so $\ \mathbb{E}[Y\chi_k] = \mathbb{E}\Big[\mathbb{E}[Y \chi_k \mid s_0, a_0, \dots, s_k]\Big] = 0$ for every $Y$ known at time $k$
3. **Occupancy gradient.** Take $h = \mathbf{1}[s_{t+1} = x]$. By step 2, all scores with $k \ge t+1$ drop out:
   $$d_\phi(x) = \tfrac{1}{Z_H} \textstyle\sum_{t \lt H} \gamma^t\, \mathbb{E}_\phi\Big[\mathbf{1}[s_{t+1}=x]\Big]$$
   $$\nabla d_\phi(x) = \tfrac{1}{Z_H} \textstyle\sum_{t \lt H} \gamma^t\, \mathbb{E}_\phi\Big[\mathbf{1}[s_{t+1}=x] \sum_{k \le t} \chi_k\Big]$$

</v-clicks>

<!--
⏱ 0:50 (total 6:10)

The proof has five steps.

[click] Step one is the usual likelihood ratio. Only the policy factors depend on phi, so the gradient of an expectation is the expectation times the sum of scores.

[click] Step two is the key trick. Given the past, the next score has mean zero, because the policy probabilities sum to one, and the gradient of one is zero. So a future score is uncorrelated with anything already known.

[click] Step three: apply step one to the event "in cell x at time t plus one". Scores after time t are in its future, so they drop out. This is the occupancy gradient.
-->

---
transition: slide-up # next slide goes "down" into the same question
---

## Proof of the exact gradient (2/2)

<v-clicks>

4. **Chain rule.** $\nabla\big[q(x) f\big(d_\phi(x)/q(x)\big)\big] = f'\big(d_\phi(x)/q(x)\big)\,\nabla d_\phi(x)$, so $\ \nabla J = \sum_x g_\phi(x)\,\nabla d_\phi(x)$
5. ⭐ **Swap the sums.** Use $\sum_x g_\phi(x)\mathbf{1}[s_{t+1}=x] = g_\phi(s_{t+1})$, then sum over $t \ge k$ first:
   $$\nabla J = \tfrac{1}{Z_H}\,\mathbb{E}\Big[\sum_{t}\sum_{k \le t} \gamma^t\, g_\phi(s_{t+1})\,\chi_k\Big] = \tfrac{1}{Z_H}\,\mathbb{E}\Big[\sum_{k} \gamma^k \chi_k \underbrace{\sum_{t \ge k} \gamma^{t-k} g_\phi(s_{t+1})}_{G_k}\Big] \quad \blacksquare$$

</v-clicks>

<!--
⏱ 0:40 (total 6:50)

[click] Step four is the chain rule: the gradient of J is the signal-weighted sum of occupancy gradients.

[click] Step five is the second tricky point. The indicator picks the signal at the visited state. We get a double sum over k at most t. Summing over t first, and splitting gamma to the t into gamma to the k times gamma to the t minus k, gives the signal-to-go.

[click] Three consequences. Keep gamma to the k; without it, it is not the gradient. Baselines are free, by step two. And a constant shift of the signal does nothing.
-->

---

## f-PG: what is exact, what is approximate

<v-clicks>

- ✅ The gradient formula, for the true occupancies. With the exact signal, the Monte Carlo average is unbiased
- ⚠️ **Plug-in estimate signal** $\hat g = f'(\hat d / \hat q)$: biased, because $f'$ is nonlinear.

</v-clicks>

<!--
⏱ 0:40 (total 7:30)

What is exact in f-PG, and what is not?

[click] The formula is exact. With the true signal, averaging trajectories gives an unbiased gradient.

[click] But the signal must be estimated from samples, and because f-prime is nonlinear, it is biased.
-->

---

# f-IRL, f-PG relation?

<v-clicks>

- f-IRL learns a **reward** $r_\theta$ and assumes the agent's trajectories follow a **Gibbs law**:
  $$\rho_\theta(\tau) \propto \bar p(\tau)\,\exp\Big(\tfrac{1}{\alpha}\textstyle\sum_{t} r_\theta(s_t)\Big)$$
- ✅ **Theorem** (under this law): with $u = q / \rho_\theta$ and $h_f(u) = f(u) - u f'(u)$,
  $$\nabla_\theta L_f(\theta) = \tfrac{1}{\alpha T}\,\mathrm{Cov}_{\tau \sim \rho_\theta}\Big(\textstyle\sum_t h_f\big(u(s_t)\big),\ \sum_t \nabla_\theta r_\theta(s_t)\Big)$$
- **relation**: $\ h_f(q/m) = (f^\circ)'(m/q)$ with $f^\circ(u) = u f(1/u)$. Both push each state by the same function of the ratio learner / expert
- ⚠️ Gibbs Law only valid for deterministic env; KDE densities. The learned reward is not unique

</v-clicks>

<!--
⏱ 0:30 (total 8:00)

And f-IRL? Theory only.

[click] It learns a reward, and assumes the agent's trajectories follow a Gibbs law.

[click] Then the reward gradient is a covariance along trajectories. The proof is in a backup slide.

[click] My addition: its weight h-f is the derivative of the learner-first generator. So f-IRL and f-PG push each state by the same function of the ratio.

[click] But f-IRL needs an RL inner loop, and is exact only for an exact Gibbs agent.
-->

---
transition: slide-up # next slide goes "down" into the same question
---

# Can one model predict where every policy goes?

### (Forward Backward paper) two assumptions:

1. **FB** models successor measure over replay buffer state distribution $\nu$  (rank $d = 16$):
  $$M^{\pi_z}(s,a,x) \approx F(s,a,z)^\top B(x)\,\nu(x)$$
2. **For each latent** $z$, there is one $\ \pi_z(a \mid s) = \mathrm{softmax}_a\big(F(s,a,z)^\top z / \tau\big)$ (the 2021 paper: greedy)

<v-clicks>

- For any reward $r$, we can compute a **Task vector**: $\ z_r = \mathbb{E}_{x\sim\nu}\big[r(x)\,B(x)\big]$
- ✅ **Theorem (reward → task vector).** If the model is exact at $z_r$ and the policy is greedy, then $\pi_{z_r}$ is **optimal** for $r$ and $\ Q^\star_r(s,a) = F(s,a,z_r)^\top z_r$

</v-clicks>

<!--
⏱ 0:40 (total 8:40)

Second family. Can one model predict where every policy goes, without any reward?

[click] FB writes the successor measure as a product: a forward vector F of state, action and latent z, times a backward vector B of the future state, size sixteen.

[click] Each latent z defines a policy: the softmax of F transpose z.

[click] Any reward becomes a task vector by one average over the replay data. No planning.

[click] The theorem: if the model is exact at z-r, the greedy policy of z-r is optimal. Control is solved in advance, for every reward.
-->

---
transition: slide-up # next slide goes "down" into the same question
---

## Proof of reward → task

<v-clicks>

1. Linearity of Expectation:
   $$Q^\pi_r(s,a) = \mathbb{E}\textstyle\sum_{t \ge 0}\gamma^t r(s_{t+1}) = \sum_x M^\pi(s,a,x)\, r(x)$$
2. **Exactness at $z$**:
   $$Q^{\pi_z}_r(s,a) = \textstyle\sum_x F(s,a,z)^\top B(x)\,\nu(x)\, r(x) = F(s,a,z)^\top z_r$$
3. ⭐ At $z = z_r$: greedy $\ \pi(s) \in \arg\max_a F(s,a,z_r)^\top z_r = \arg\max_a Q^\pi_r(s,a)$. A policy greedy for its own $Q$ ⇒ optimal. $\ \blacksquare$
4. ⚠️ **Softmax** policy (in practice): $\ V^\star - V^{\pi} \le \tau \log|A| / (1-\gamma) = 32.2$ at $\tau = 1$, more than the largest return $20$: **vacuous**

</v-clicks>

<!--
⏱ 0:45 (total 9:25)

The proof has three steps.

[click] A Q-value is the reward integrated against the successor measure: the same linearity as before.

[click] If the model is exact, replace the successor measure by F transpose B times nu. The sum over x is exactly z-r. No greediness yet.

[click] The tricky step. At z-r, the policy maximizes F transpose z-r, which is its own Q-function. A policy greedy for its own Q-function is a fixed point of the Bellman optimality operator, and that fixed point is unique: V-star. So it is optimal.

[click] In practice the policy is a softmax. Then only a temperature bound remains, and here it is larger than any return.
-->

---

## Model Exactness Requirements

<v-clicks>

- In a finite maze the successor measure is a matrix: $\ \mathbf{M}_\pi = \tilde P\, R_\pi$, with rows $P(\cdot \mid s,a)$ in $\tilde P$ and $R_\pi = (I - \gamma P_\pi)^{-1}$ invertible $\ \Rightarrow\ \operatorname{rank} \mathbf{M}_\pi = \operatorname{rank} \tilde P$
- An FB model is a product through $\mathbb{R}^d$: $\ \mathbf{M} = F_z\, B_\nu^\top \Rightarrow \operatorname{rank} \le d$. **Exactness needs $d \ge \operatorname{rank} \tilde P$**
- ⭐ Our "stay" action with slip $0.1$: $\ P_{\text{stay}} = 0.9\, I + 0.1\, U$ ($U$ row-stochastic). strictly diagonally dominant, so invertible $\Rightarrow \operatorname{rank} \tilde P = \#S$
- $\operatorname{rank} \tilde P = 49$ (open), $40$ (four rooms) $\gt d = 16$: **no rank-16 model can be exact**.

</v-clicks>

<!--
⏱ 0:40 (total 10:05)

But can the model be exact in our maze? This short proof explains most of my FB results.

[click] The successor matrix is the transition matrix times an invertible matrix, so it has the rank of the transition matrix.

[click] An FB model goes through sixteen dimensions, so its rank is at most sixteen.

[click] Our stay action has at least 0.9 on the diagonal and at most 0.1 elsewhere, so it is invertible. The rank is the number of states.

[click] That is 49 or 40, far above 16. No rank-sixteen model can be exact, however long we train. The measured error is even larger than the total mass.
-->

---
transition: slide-up # next slide goes "down" into the same question
---

# How do we imitate with a frozen FB model?

<v-clicks>

- **ER-FB** reads $z$ off the expert demonstrations: $\ z_{ER} = \tfrac{1}{N_E}\sum_i \tfrac{1}{H}\sum_{t=1}^{H} B(s^i_t)$. ✅ This is task vector $z_{r_e}$ for the reward $r_e = \rho_e / \nu$ (expert state frequency / replay frequency) - no matching
- **DM-FB** matches successor measures at expert states:
  $$L_{DM}(z) = \mathbb{E}_{x \sim \bar\rho}\,\big\Vert M^{\pi_z}(x,\cdot) - M^{\pi_E}(x,\cdot) \big\Vert^2_{L^2(\nu)^*} = \mathbb{E}_{x \sim \bar\rho}\,\big\Vert F(x,z)^\top B(\cdot) - m_E(x,\cdot) \big\Vert^2_{L^2(\nu)}$$
- What it can compute, with $G_B = \mathbb{E}_\nu[BB^\top]$ and $\bar\psi_E(x) = \mathbb{E}_{\pi_E}\sum_k \gamma^k B(s_{k+1})$ (expert state features):
  $$\ell_{DM}(z) = \mathbb{E}_{x \sim \bar\rho}\big[F(x,z)^\top G_B\, F(x,z) - 2\,F(x,z)^\top \bar\psi_E(x)\big]$$

</v-clicks>

<!--
⏱ 0:40 (total 10:45)

How do we imitate with a frozen FB model?

[click] ER-FB averages B over the expert states. This is the task vector of a reward: expert frequency over replay frequency. So it maximizes a reward; it does not match.

[click] DM-FB really matches: at expert states, it compares the successor measures of the policy and of the expert.

[click] It cannot compute that directly. It computes this quadratic expression, with the Gram matrix of B and the expert's features from the demonstrations, and takes 300 Adam steps on it.
-->

---
transition: slide-up # next slide goes "down" into the same question
---

## Proof: the DM-FB expansion

<v-clicks>

1. **Exact model** $\Rightarrow M^{\pi_z}(x,\cdot)$ has density $F(x,z)^\top B(\cdot)$ with respect to $\nu$. Call $m_E(x,\cdot)$ the expert's density
2. ⭐ **Expand the square** (all integrals against $\nu$):
   $$\Vert F^\top B - m_E \Vert^2_{L^2(\nu)} = F^\top \underbrace{\textstyle\int B B^\top d\nu}_{G_B} F \;-\; 2F^\top \underbrace{\textstyle\int B\, m_E\, d\nu}_{\bar\psi_E(x)} \;+\; \underbrace{\Vert m_E \Vert^2}_{\text{no } z}$$
   using $\ \int B\, m_E\, d\nu = \int B\, dM^{\pi_E}(x,\cdot) = \mathbb{E}_{\pi_E}\sum_k \gamma^k B(s_{k+1})$
3. ✅ Average over $x \sim \bar\rho$: $\ L_{DM}(z) = \ell_{DM}(z) + C_E$, with $C_E$ independent of $z$. $\ \blacksquare$
4. ⭐ **Pythagoras** in $L^2(\nu)$: $\ L_{DM} = \mathbb{E}_x\big[\Vert F^\top B - \Pi_B m_E\Vert^2 + \Vert m_E - \Pi_B m_E \Vert^2\big]$. DM-FB only sees the part of the expert that lives in $\mathrm{span}(B)$

</v-clicks>

<!--
⏱ 0:35 (total 11:20)

Why is this the right loss? Under exactness, it is.

[click] With an exact model, the policy's successor measure has density F transpose B.

[click] Expand the squared distance. The first term gives the Gram matrix, the cross term the expert's features, and the last term has no z.

[click] So the true objective is the computable one plus a constant, even for softmax policies.

[click] And by Pythagoras, DM-FB only sees the part of the expert inside the span of B.
-->

---

## FB family: what is exact, what is approximate

<v-clicks>

| Step | ✅ Exact statement | ⚠️ What the algorithm does |
|---|---|---|
| Pretraining loss | Bellman residual $\tfrac12 \Vert \hat M_z - \mathcal{T}_z \hat M^- \Vert^2_{L^2(\nu)}$ + const | loss value does **not** bound the model error |
| Model | optimal policies needs exact model | rank $16 \lt 40$ or $49$: never exact here |
| Policy | optimality possible for greedy | softmax with $\tau = 1$: vacuous bound |
| ER-FB | maximizes $r_e = \rho_e / \nu$ | might end up in different absorbing state |
| DM-FB | $L_{DM} = \ell_{DM} + C_E$ | minimizes loss using the **model's** estimate & model not exact => f-divergance might not agree |

</v-clicks>

<!--
⏱ 0:35 (total 11:55)

The exact-versus-approximate line for FB.

[click] The pretraining loss is a Bellman residual, but its value does not tell us the model error.

[click] Optimality needs an exact model: impossible here.

[click] The softmax makes the bound vacuous.

[click] ER-FB maximizes a reward; it can end far from the expert's distribution.

[click] And DM-FB minimizes the model's estimate. When the model is wrong, the two can disagree. Remember this for the results.
-->

---
transition: slide-up # next slide goes "down" into the same question
---

# Q2. How are the two families alike?

<v-clicks>

- Same input: expert **states** only.
- Same kind of object: linked by $\ d_\pi \approx (1-\gamma)\, M^\pi(s_0,\cdot)$
- Both are exact only for true (population) quantities. In practice both rely on **plug-in estimates** whose errors are not related to optimized quantity.

</v-clicks>

<!--
⏱ 0:20 (total 12:15)

How are the two families alike?

[click] Same input: states only.

[click] Same kind of object, linked by the formula from the start.

[click] And both rely on estimates whose errors their objective cannot see.
-->

---

## How they differ, and when to use which?

<v-clicks>

| | f-PG | FB + ER-FB / DM-FB |
|---|---|---|
| Interaction | rollouts **for every task** | replay **once**, none per task |
| Learned per task | the whole policy | task vector ($z$) |
| Exact guarantee | the gradient formula | only for an exact model |
| Fails first | biased density estimate, variance | model error (DM-FB can exploit it) |
| Use when | one task, cheap simulator | many tasks, costly interaction, accurate model |

</v-clicks>

f-IRL: only when you want a **reward** to inspect or to transfer (inner RL loop, reward not unique)

<!--
⏱ 0:35 (total 12:50)

The key difference is where the approximation sits.

[click] f-PG needs new rollouts for every task. FB collects data once.

[click] f-PG learns a whole policy; FB learns sixteen numbers.

[click] f-PG's guarantee is its gradient; FB's needs an exact model.

[click] So f-PG fails through noisy estimates, FB through model error.

[click] Use f-PG for one task with a cheap simulator. Use FB for many tasks with costly interaction, if the model is accurate. And f-IRL if you want the reward itself.
-->

---
layout: two-cols-header
---

# Q3. Which knobs matter, and what do we predict?

::left::

<v-clicks>

- **Choice of $f$** acts only through $f'(v)$, $v = d/q$. JS: bounded. KL$(d\Vert q)$: needs smoothing. KL$(q\Vert d)$: mass-covering, so expect **more stocastic** policies
- **Rank $d$**: exactness needs $d \ge 40$ or $49$. We use $16$, so expect **model error**
- **Orthonormality penalty** $\tfrac14\Vert G_B - I\Vert_F^2$ fixes a freedom of $F, B \mapsto CB, C^{-T}F$ softly, so expect better **conditioning**, not accuracy


</v-clicks>

::right::

<img src="/fprime-shapes.png" class="w-full mt-10" />

<!--
⏱ 0:35 (total 13:25)

Which knobs matter, and what do we predict before running anything?

[click] The choice of f acts only through f-prime, on the right. KL of q over d punishes missing expert cells, but barely punishes extra visits. So I expect a more spread-out policy.

[click] Rank sixteen is below 40, so I expect model error.

[click] Temperature and the norm of z act as one inverse temperature.

[click] The orthonormality penalty only fixes a symmetry, so I expect better conditioning, not accuracy.
-->

---
layout: two-cols-header
---

# Q4. How did we test the methods fairly? (finite maze)

::left::

<v-clicks>

- Two 7×7 mazes (open: 49 cells, four rooms: 40), 5 actions, slip $0.1$, 3 goals each. Expert: certified optimal policy
- 4 demonstrations × 100 steps, **states only**. Same budget per task: **50,000 transitions** (f-PG on-policy, FB task-blind replay)
- In finite maze, an **exact oracle** computes every occupancy, return and model error. Monte Carlo estimates in continues
- Held-out $\mathrm{JS}(d_\pi, q_E)$ (max $\log 2 \approx 0.69$) and success. Thresholds fixed in advance: **JS ≤ 0.10 and success ≥ 0.80**. 3 seeds, protocol frozen before the final runs

</v-clicks>

::right::

<img src="/fig1-finite-mazes.png" class="w-full mt-8" />

<!--
⏱ 0:35 (total 14:00)

How did I test the methods fairly?

[click] Two small mazes, three goals each, ten percent slip. The expert is a certified optimal policy.

[click] Every method gets the same four demonstrations, states only, and the same fifty thousand transitions.

[click] An exact oracle computes every occupancy and every model error, so there is no sampling noise.

[click] I measure the divergence to the true expert occupancy, and success. Thresholds, seeds and settings were all fixed before the final runs.
-->

---
layout: two-cols-header
transition: slide-up   # next slide goes "down" into the same question
---

# What happened in the finite mazes?

::left::

<v-clicks>

- **f-PG: 18 / 18** runs meet both thresholds, first at 20k–30k transitions. JS $0.001$–$0.03$, success $\ge 0.998$
- **ER-FB: 2 / 18**. Often reaches the goal (success ≥ 0.8 in 14 runs) but rarely **stays**: ends in the goal with probability $0.001$–$0.51$ (expert: $0.92$)
- **DM-FB: 0 / 18**, and worse than its own starting point ER-FB in 16 of 18 runs
- f-PG is stable across seeds (SD ≤ 0.014 nats). FB moves with the seed and the pretraining snapshot (up to 0.53 nats)

</v-clicks>

::right::

<img src="/fig3-final-finite.png" class="h-96 mx-auto" />

<!--
⏱ 0:45 (total 14:45)

What happened in the finite mazes? On the right, every run: divergence on top, success below, thresholds dashed.

[click] f-PG, in blue, met both thresholds in all eighteen runs, after only twenty to thirty thousand transitions.

[click] ER-FB, in green, often reaches the goal but does not stay. Only two runs pass.

[click] DM-FB, in orange, passes in none, and is usually worse than ER-FB, its own starting point.

[click] f-PG gives the same result for every seed. FB depends on the seed and on the pretraining snapshot.
-->

---

## Ablations and cost (four rooms, goal (5,5))

<img src="/fig7-ablations.png" class="h-40 mx-auto" />

<v-clicks>

- **Choice of $f$**: KL$(d\Vert q)$ ≈ JS. KL$(q\Vert d)$ is slower (40k–50k) and more stochastic (entropy 0.38 vs 0.12 nats), as predicted
- **No orthonormality**: almost uniform policies, and still 0 runs meet the thresholds: the penalty is not the bottleneck

</v-clicks>

<!--
⏱ 0:30 (total 15:15)

Two ablations and the cost.

[click] KL of q over d is slower and more random, as predicted. The learning rate was tuned for JS only.

[click] Without the orthonormality penalty, the Gram matrix degrades, but the result barely changes. So it is not the bottleneck.

[click] f-PG takes under a second per task. FB pretraining takes fifty seconds, then imitation is almost free. But FB never reached f-PG's quality.
-->

---

# Why did FB imitation fail in the finite mazes?

<v-clicks>

- The model **cannot** be exact (rank 40 / 49 > 16). The oracle measures it: TV error $36$–$80$ (true mass $20$), $\max|Q - F^\top z_r|$ up to $15$ (values ≤ 20)
- DM-FB did its job: the **model's** objective fell in all 27 adaptations (97 % of the drop within 100 steps)…
- …but the **true** divergence rose in 16 of 18 finite runs. Our reading: the optimizer exploits model error (consistent with the data, not proved)
- ER-FB: an exact model would maximize $r_e = \rho_e/\nu$, which peaks at the goal; the wrong model reaches the goal without staying.

</v-clicks>

<img src="/fig6-dmfb-latent-steps.png" class="h-48 mx-auto" />

<!--
⏱ 0:40 (total 15:55)

Why did FB imitation fail in the finite mazes?

[click] First, the model cannot be exact, and the oracle shows its error is larger than the total mass.

[click] Second, bottom left: DM-FB lowered its model objective in every run. The optimizer did its job.

[click] But bottom right: the true divergence went up in sixteen of eighteen runs. My reading: the optimizer moved to policies the model represents worst. Consistent with the data, not proved.

[click] ER-FB reaches the goal but leaves it. The clean test is a model with d at least 49, which I did not run.
-->

---

# Q5. What changes in a continuous maze?

<img src="/fig2-continuous-maze.png" class="h-40 mx-auto" />

<v-clicks>

- State = coordinates in $[0,1]^2$, the same 5 moves (length $1/7$) plus Gaussian noise. Budget: **100,000 transitions**
- **No method meets the thresholds.** f-PG is closest: JS $0.24$–$0.41$, success $0.92$–$0.99$. ER-FB and DM-FB: JS $0.35$–$0.69$
- **Reaching ≠ staying**: rollouts that end in the goal: expert 72–76 %, f-PG 6–14 %, ER-FB ≤ 0.4 %

</v-clicks>

<!--
⏱ 0:35 (total 16:30)

What changes in a continuous maze? Same four rooms, but the state is a position, with small noise.

[click] The budget doubles.

[click] Here is the DQN-style approximation: FB's forward network outputs one vector per action, as a DQN outputs one value per action.

[click] No method meets the thresholds. f-PG is the closest.

[click] But reaching is not staying: three quarters of expert rollouts end at the goal, against at most fourteen percent for f-PG.
-->

---
layout: two-cols-header
---

# Why did things not work in the continuous maze?

::left::

<v-clicks>

- **FB**: the backward Gram matrix is almost singular, $\lambda_{\min}(\hat G_B) \approx 2 \times 10^{-5}$ (finite: 0.93–0.96). Untested causes: raw cordinate $(x,y)$ inputs (the 2021 paper used RBF features), 20k updates, $d = 16$
- FB imitation stays flat over pretraining snapshots (mean JS 0.51–0.56); the model error cannot be computed here
- **f-PG** was still improving at the end, stopped because **budget-limit**. its policy stays very stochastic, so it passes through the goal

</v-clicks>

::right::

<img src="/fig4-fpg-curves-continuous.png" class="h-72 inline-block" />
<img src="/fig5-fb-curves-continuous.png" class="h-72 inline-block" />

<!--
⏱ 0:30 (total 17:00)

Why did things not work in the continuous maze?

[click] For FB, the backward features are nearly degenerate: one direction is almost zero on the data. Possible causes, untested: raw inputs, too few updates, the rank.

[click] On the right, FB stays flat during pretraining.

[click] f-PG, on the left plot, was still improving, so the budget limited it. And its policy stays random, so it passes through the goal.
-->

---
layout: center
class: text-center
---

# Thank you.
# Any questions?

---

# Backup · Why is the f-IRL gradient a covariance?

<v-clicks>

1. **Gibbs derivative.** $\log\rho_\theta(\tau) = \log\bar p(\tau) + R_\theta(\tau)/\alpha - \log Z(\theta)$ with $R_\theta = \sum_t r_\theta(s_t)$, and $\nabla\log Z = \mathbb{E}_{\rho_\theta}[\nabla R_\theta]/\alpha$, so
   $$\nabla \rho_\theta(\tau) = \tfrac{1}{\alpha}\,\rho_\theta(\tau)\big(\nabla R_\theta(\tau) - \mathbb{E}\,\nabla R_\theta\big) \ \Rightarrow\ \nabla\rho_\theta(s) = \tfrac{1}{\alpha T}\,\mathrm{Cov}\big(\eta_\tau(s), \nabla R_\theta\big)$$
   where $\eta_\tau(s)$ = number of visits of $\tau$ to $s$
2. ⭐ **Chain rule** with $u = q/\rho_\theta$: $\ \nabla\big[\rho_\theta\, f(q/\rho_\theta)\big] = \big(f(u) - u f'(u)\big)\nabla\rho_\theta = h_f(u)\,\nabla \rho_\theta$
3. **Bilinearity of Cov, then counting**: $\ \sum_s h_f(u(s))\,\eta_\tau(s) = \sum_t h_f(u(s_t))$ $\ \blacksquare$

</v-clicks>

<!--
Backup slide, only if asked about f-IRL.

Why is the f-IRL gradient a covariance?

[click] Under the Gibbs law, the log-probability of a trajectory is the cumulative reward over alpha, minus log Z. The gradient of log Z is the mean reward gradient over alpha. So the gradient of a trajectory's probability is its probability times the centred reward gradient. Summing visit counts gives a covariance.

[click] The chain rule on each term rho times f of q over rho gives f minus u f-prime, which is h-f, times the gradient of rho.

[click] Finally, the covariance is bilinear, and summing h-f over visit counts is the same as summing it along the trajectory.
-->

---

# Backup · Why is the softmax bound vacuous?

<v-clicks>

- **Log-sum-exp**: for $p = \mathrm{softmax}(x/\tau)$, $\ \sum_a p_a x_a \ge \max_a x_a - \tau\log|A|$
- A softmax fixed point $\pi = \mathrm{softmax}(Q^\pi/\tau)$ therefore gives $\ \mathcal{T}V^\pi - \tau\log|A| \le V^\pi \le \mathcal{T}V^\pi$
- Residual bound $\ \Vert V - V^\star\Vert_\infty \le \Vert \mathcal{T}V - V\Vert_\infty/(1-\gamma)\ \Rightarrow\ 0 \le V^\star - V^\pi \le \tau\log|A|/(1-\gamma)$ $\ \blacksquare$
- At $\tau=1$, $|A| = 5$, $\gamma = 0.95$: $\ \log 5 / 0.05 \approx 32.2 \gt 20$ = the largest possible return, so the bound says nothing

</v-clicks>

<!--
Backup slide, only if asked about the softmax policies.

Why is the softmax bound vacuous?

[click] For a softmax, the average of the scores is at least the maximum minus tau log of the number of actions.

[click] So, for a softmax fixed point, the value is squeezed between the Bellman optimality operator minus that gap, and the operator itself.

[click] The standard residual bound turns this into a value gap of at most tau log A over one minus gamma.

[click] With our numbers this is 32.2, but returns are at most 20, so the bound is empty. Tau equal one was chosen in the pilot to keep the policies stochastic, not for this bound.
-->

---


## Backup · What does FB pretraining minimize?

Replay $(s,a,s_1)\sim D$, latent $z\sim D_z$, and independent future state $x\sim\nu$.

<v-clicks>

- **Online prediction** and **frozen continuation target**:
  $$\hat m(x):=F_\xi(s,a,z)^\top B_\omega(x),$$
  $$u_{s_1}(x):=\gamma\sum_{a_1}\pi_z^-(a_1\mid s_1)F^-(s_1,a_1,z)^\top B^-(x).$$
- The reward-free loss is $\mathcal L=\mathbb E_{D,D_z}\ell$, where
  $$\boxed{\ell=\frac12\,\mathbb E_{x\sim\nu}\big[(\hat m(x)-u_{s_1}(x))^2\big]-\hat m(s_1).}$$
- $F^-,B^-$ and their softmax policy $\pi_z^-$ stay **fixed during differentiation**. The last term inserts the first transition

</v-clicks>

<!--
The inner product predicts a successor density relative to nu, the replay next-state law. The observed next state s1 is coupled to s,a; the future-state sample x is independent. Our next-state convention gives the immediate transition weight one, so the negative term has no extra gamma.

Source: report, Section 6.5, Eqs. (F.3)-(F.5).
-->

---
transition: slide-up   # next slide goes "down" into the same question
---

## Backup · Why is this a Bellman objective?

Fix $(s,a,z)$. Assume $p_\nu:=dP(\cdot\mid s,a)/d\nu\in L^2(\nu)$.
The **Bellman target density** is $y:=p_\nu+\mathbb E_{s_1\mid s,a}u_{s_1}$.

<v-clicks>

1. **Change measure**:
   $$\mathbb E_{s_1\mid s,a}\hat m(s_1)=\int\hat m(x)p_\nu(x)\nu(dx).$$
2. **Expand, then complete the square**, using $L^2(\nu)$ norms:
   $$\mathbb E_{s_1}\ell=\frac12\|\hat m\|^2-\langle\hat m,y\rangle+C_1=\frac12\|\hat m-y\|^2+C.$$
3. Since $C$ is independent of $(\xi,\omega)$ for frozen targets,
   $$\boxed{\nabla_{\xi,\omega}\mathcal L=\nabla_{\xi,\omega}\mathcal R,\qquad\mathcal R:=\frac12\mathbb E_{D_{SA},D_z}\|\hat m-y\|_{L^2(\nu)}^2.}$$

</v-clicks>

<!--
The squared continuation term contributes C1=one half E||u_s1|| squared. Its cross term gives minus the inner product with E u_s1. The negative next-state term supplies minus the inner product with p_nu. Completing the square gives C=C1-one half ||y|| squared. These constants depend only on the kernel and frozen targets. Bounded outputs and dominated parameter derivatives justify exchanging integrals and derivatives.

Source: report, Proposition 6.6(a) and Proposition F.22(a).
-->

---
transition: slide-up   # next slide goes "down" into the same question
---

## Backup · What is exact during training?

<v-clicks>

- The expected online gradient includes the **orthonormality penalty**:
  $$G_B:=\mathbb E_\nu[BB^\top],\qquad\mathcal J_{\mathrm{step}}:=\mathcal L+\frac{\lambda_{\mathrm{orth}}}{4}\|G_B-I_d\|_F^2.$$
- **Independent minibatches** give unbiased gradients for the sampling law used. With a stored buffer, this is the empirical buffer loss
- **Moving targets:** differentiate the online networks, then Polyak-update the targets. This is a semi-gradient procedure
- **Exact model + equal copies** makes the FB semi-gradient zero. A decreasing training loss gives no guarantee of exactness or convergence

</v-clicks>

<!--
R_orth depends only on B, so it contributes only to the backward-network gradient. The implementation's stop-gradient regularizer estimates this penalty's gradient, rather than its value. The residual equivalence on the previous slide applies with targets fixed; their later updates change the omitted constant. Exact-model stationarity concerns the FB loss alone, with exactness on all required pairs and latents. The regularized update also needs a zero penalty gradient to be stationary.

Source: report, Algorithm 2, Proposition F.22(b)-(c), Remark F.23 and Proposition F.24.
-->

---

# Backup · So what did we learn?

<v-clicks>

- **Q1** Matching occupancies = matching every state-reward return (not actions). f-PG has an exact gradient; FB is exact only for exact models, impossible here by rank
- **Q2** Same input and objects; the difference is **where the approximation sits**: f-PG samples the real environment, FB trusts a learned model
- **Q3** $f$ shapes the signal (KL$(q\Vert d)$: slower, more stochastic); rank decides exactness; the orthonormality penalty gives conditioning, not accuracy
- **Q4** Finite: f-PG 18/18, ER-FB 2/18, DM-FB 0/18. DM-FB lowers its model objective while the true divergence rises
- **Q5** Continuous: no method meets the thresholds; f-PG is closest but does not stay; FB's Gram matrix is nearly singular
- **Limits and next steps**: 3 seeds, small mazes, narrow tuning. Next: FB with $d \ge 49$, track the true DM objective with the oracle, RBF inputs, longer f-PG runs

</v-clicks>

<!--
⏱ 0:45 (total 17:45)

So what did we learn? One line per question.

[click] Matching occupancies gives the expert's return for every state reward. f-PG has an exact gradient; FB needs an exact model, impossible here.

[click] The real difference is where the approximation sits: real samples for f-PG, a learned model for FB.

[click] f shapes the signal; rank decides exactness.

[click] Finite: f-PG reliable; DM-FB improved its model objective while moving away from the expert.

[click] Continuous: no method passed.

[click] Three seeds, small mazes. Next, I would run FB with rank at least 49, where the theory can apply.
-->
