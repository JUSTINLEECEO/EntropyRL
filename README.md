<h1 align="center">EntropyRL</h1>

EntropyRL is a cutting-edge reinforcement learning (RL) framework designed to dynamically regulate sentence-level policy entropy, mitigating hallucination in vision-language models (VLMs). Built on the GRPO framework, EntropyRL introduces dynamic entropy modulation based on hallucination metrics, ensuring fine-grained control and robust training.

---

## Key Features

- **Token-Level Granularity**: Achieves precise regulation of policy entropy through sentence-level clipping bounds in the GRPO optimization target.
- **Dynamic Entropy Modulation**: Adapts entropy dynamically during rollout based on hallucination metrics (e.g., CHAIR scores).
- **Extensibility**: Seamlessly integrates with the EasyR1 framework and supports MSCOCO datasets with CIDEr rewards.

---

## Getting Started

### Installation

To install EntropyRL, follow these steps:

```bash
git clone https://github.com/JUSTINLEECEO/EntropyRL.git
cd EntropyRL
pip install -e .
```

### Training

To train a model using EntropyRL with CHAIR-enabled evaluation:

```bash
bash examples/chair_mscoco.sh
```

---

## Methodology

EntropyRL builds upon the GRPO optimization target, introducing dynamic entropy modulation by replacing static clipping bounds with sample-specific bounds informed by hallucination metrics.

### GRPO Objective 

The GRPO objective is defined as:
```math
\begin{aligned}\mathcal{J}_{\mathrm{GRPO}}(\theta)&=\mathbb{E}_{(q,\alpha)\thicksim\mathcal{D},\{o_i\}_{i=1}^G\thicksim\pi_{\theta_{\mathrm{old}}}(\cdot|q)}\\&\left[\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\mathrm{clip}\left(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right)-\beta D_{\mathrm{KL}}(\pi_\theta||\pi_{\mathrm{ref}})\right)\right]\end{aligned}
```
in which $r_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}$.

### Dynamic Clipping

EntropyRL replaces the static $\varepsilon_{\mathrm{high}}$ with a dynamic, sample-specific value informed by the CHAIR metric:
```math
\mathrm{clip}\left(r_{i,t}(\theta),1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high},s}\right)\hat{A}_{i,t}
```

For example, $\varepsilon_{\mathrm{high},s}$ can be interpolated as:
```math
\varepsilon_{\mathrm{high},s} = \begin{cases}
0.3, & \mathrm{CHAIR.i} < 10^{-4} \\
0.5, & \mathrm{CHAIR.i} \ge 10^{-4}
\end{cases}
```

---

## Workflow

1. **Base Framework**: EntropyRL uses EasyR1 (GRPO) as the foundational codebase and MSCOCO with CIDEr as the reward function.
2. **CHAIR Evaluation**: During rollout, a CHAIR evaluation server computes hallucination scores (`CHAIR.i`) for each sample, after which an array containing the CHAIR.i scores is returned to the main process.
3. **Dynamic Adjustment**: The main process adjusts $\varepsilon_{\mathrm{high},s}$ dynamically based on `CHAIR.i` scores received and feeds the updated values into the GRPO optimization target.

---

## Configuration

- **Dataset**: MSCOCO with CIDEr reward.
- **CHAIR Server**: A server endpoint that evaluates hallucination scores (`CHAIR.i`) for rollout samples.
- **Dynamic Clipping**: Implement interpolation rules for $\varepsilon_{\mathrm{high},s}$ in `adjust_clip_ratio_high_by_chair_score`.

---

## References

- [EasyR1](https://github.com/hiyouga/EasyR1): The foundational framework for EntropyRL.
- [CHAIR](https://github.com/LisaAnne/Hallucination): A metric for evaluating hallucination in VLMs.
- [MSCOCO caption evaluation toolkit](https://github.com/salaniz/pycocoevalcap): Different tools for MSCOCO caption evaluation.



