# 3. Methodology: Uncertainty-Gated Multi-Scale Inference (MCG-TTA)

Motivated by the computational and latency constraints of clinical edge devices [12, 25, 26], and the observation that compact convolutional architectures relying on parallel dilated convolutions can exhibit sensitivity to input scale variations [5, 21], this work introduces **Multi-Scale Consistency Gated Test-Time Augmentation (MCG-TTA)**. The underlying hypothesis is that scale-induced ambiguity in compact multi-dilation models concentrates in low-confidence predictions, and that these predictions can be dynamically stabilized at inference time by probing the model with multi-scale views aligned with its internal dilation structure—requiring zero additional learned parameters.

In this section, we formulate the general image classification problem and base model representation, analyze the theoretical scale sensitivity of compact atrous convolutions, present the formal algorithmic pipeline of MCG-TTA across its three decision stages, derive an analytical edge latency model, and define the volumetric patient-level diagnostic aggregation.

---

## 3.1. Problem Formulation and Base Network Representation

Let $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ denote a 2D brain MRI slice, where $H, W$ represent spatial dimensions and $C$ denotes channel dimensionality. A deep neural network $f_\theta$ parameterized by learned weights $\theta$ maps the input slice to a vector of logits $\mathbf{z} = f_\theta(\mathbf{I}) \in \mathbb{R}^K$ over $K$ mutually exclusive diagnostic categories. 

The predicted class posterior probabilities are computed via the softmax operator:
$$
\mathbf{p} = \sigma\left(f_\theta(\mathbf{I})\right) \in \Delta^{K-1}, \quad \text{where } p_c = \frac{\exp(z_c)}{\sum_{j=1}^K \exp(z_j)}
\tag{1}
$$
The model's predictive confidence is quantified by the maximum posterior probability:
$$
p_{\max} = \max_{c \in \{1, \dots, K\}} p_c
$$
with initial predicted class $c_0 = \arg\max_c p_c$.

To support execution on resource-constrained hardware, we adopt a compact multi-scale convolutional architecture as the base backbone (specifically LiteSpeed-Net [21]). The backbone incorporates multi-path atrous (dilated) depthwise convolutions (MPAC blocks [21]) to expand the effective receptive field without parameter inflation, allowing the network to capture both local tissue textures and broader anatomical context.

---

## 3.2. Scale Sensitivity in Compact Multi-Dilation Convolutions

While atrous convolutions provide an efficient mechanism for expanding the receptive field without downsampling feature resolutions [5], their discrete sampling mechanics introduce a distinct structural vulnerability under input distribution shifts.

An atrous convolution with dilation rate $d > 1$ samples input features on a sparse regular lattice with stride $d$. In deep, overparameterized architectures, the resulting sparse sampling artifacts—often referred to as the *gridding effect* [5]—are mitigated by dozens of subsequent parameterized layers that continuously blend feature maps across deep residual stages. In contrast, compact vision models designed for low-power edge environments [21] possess limited parameter depth, restricting their capacity to interpolate across discrete spatial sampling gaps.

When deployed across clinical environments with differing acquisition settings—such as variations in slice thickness, scanner geometry, or lesion physical dimensions—pathological margins undergo spatial scale shifts relative to the training distribution. Consequently, fixed dilated sampling points can misalign with fine anatomical boundaries, such as thin infiltrative borders or micro-structural margins [5, 21].

Based on these observations, we formulate two design principles:
1. **Uncertainty Concentration**: Spatial sampling instability manifests primarily in predictions exhibiting high predictive uncertainty ($p_{\max} < \tau$, where $\tau$ denotes an empirical confidence threshold).
2. **Architecture-Informed Scale Probing**: Because this instability stems from discrete spatial sampling intervals, uncertain predictions can be probed and stabilized at test time by applying targeted scale perturbations that mirror the internal dilation offsets of the architecture, without altering the model's learned parameters.

---

## 3.3. Proposed Method: Multi-Scale Consistency Gated Test-Time Augmentation (MCG-TTA)

To improve inference robustness while strictly bounding computational overhead, we formulate **Multi-Scale Consistency Gated Test-Time Augmentation (MCG-TTA)**. Rather than applying test-time augmentation unconditionally across all inputs—which scales inference latency linearly with the number of views [58]—MCG-TTA dynamically routes samples through a three-stage decision pipeline.

```
                     Input MRI Slice I
                             |
                             v
                  [Base Forward Pass f_\theta(I)]
                             |
                   Compute p_max = max_c p_c
                             |
                    /-----------------\
                   /   p_max >= \tau?  \
                  \  (High Confidence) /
                    \-----------------/
                       /           \
               YES (Confident)   NO (Uncertain)
                     /               \
                    v                 v
            [Accept Base]     [Generate Scale Probes]
              Output y_base     I_{s-} = Scale(I, 1 - \delta)
                                I_{s+} = Scale(I, 1 + \delta)
                                      |
                                      v
                               [Probe Forward Passes]
                               p_{s-} = f_\theta(I_{s-})
                               p_{s+} = f_\theta(I_{s+})
                                      |
                                      v
                             /-----------------\
                            / Scale Agreement?  \
                            \   c_0 = c_- = c_+ /
                             \-----------------/
                                /           \
                        YES (Consensus)    NO (Disagreement)
                              /               \
                             v                 v
                     [Triad Average]     [Escalate Multi-View TTA]
                             |                 |
                             \-------+---------/
                                     |
                                     v
                           Calibrated Prediction y_hat
```

### Stage 1: Confidence-Gated Screening
For an incoming slice $\mathbf{I}$, the base model computes posterior probabilities $\mathbf{p} = \sigma(f_\theta(\mathbf{I}))$. The maximum probability $p_{\max}$ is compared against a confidence threshold $\tau \in (0, 1)$:
- If $p_{\max} \ge \tau$, the prediction is deemed confident and stable. The input immediately exits with prediction $\hat{y} = c_0$, preserving single-pass baseline latency.

### Stage 2: Architecture-Informed Scale Probing
If $p_{\max} < \tau$, the prediction is flagged as uncertain. Instead of applying arbitrary geometric transformations, MCG-TTA synthesizes dual multi-scale probe views:
$$
\mathbf{I}_{s^-} = \mathcal{T}_{\text{scale}}(\mathbf{I}, \; 1 - \delta), \quad \mathbf{I}_{s^+} = \mathcal{T}_{\text{scale}}(\mathbf{I}, \; 1 + \delta)
\tag{2}
$$
where $\delta$ denotes a fractional scale perturbation parameter.

*Theoretical Inductive Match*: In multi-path atrous convolutions, parallel branches operate at discrete dilation steps (e.g., $d_1 = 1, d_2 = 2$). The scale perturbation factor $\delta$ shifts spatial boundaries across the discrete grid spacing $\Delta x = d$, enabling the parallel depthwise branches to re-sample lesion margins along complementary spatial coordinates without distorting diagnostic anatomy.

The model evaluates both probed views to obtain $\mathbf{p}_{s^-} = \sigma(f_\theta(\mathbf{I}_{s^-}))$ and $\mathbf{p}_{s^+} = \sigma(f_\theta(\mathbf{I}_{s^+}))$, with class predictions $c_- = \arg\max_c p_{s^-, c}$ and $c_+ = \arg\max_c p_{s^+, c}$.

### Stage 3: Consistency Verification and Selective Escalation
We evaluate semantic consistency across the triad of predicted labels $\{c_0, c_-, c_+\}$:
1. **Scale Consensus ($c_0 = c_- = c_+$)**: All scale views agree on the predicted class, indicating that uncertainty is driven by ambient contrast rather than structural grid misalignment. Calibrated posteriors are obtained via triad averaging.
2. **Scale Disagreement ($\exists \, a, b \in \{0, -, +\} \text{ s.t. } c_a \ne c_b$)**: Disagreement confirms that the input lies on an unstable decision boundary sensitive to discrete sampling. MCG-TTA selectively escalates to a broader multi-view augmentation pool $\mathcal{V}$ (including horizontal reflections and mild rotations).

The unified calibrated posterior vector $\bar{\mathbf{p}}$ is defined as:
$$
\bar{\mathbf{p}} = \begin{cases}
\dfrac{1}{3}\left(\mathbf{p} + \mathbf{p}_{s^-} + \mathbf{p}_{s^+}\right), & \text{if } c_0 = c_- = c_+ \quad \text{(Scale Consensus)} \\[1.2ex]
\dfrac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \sigma\left(f_\theta(\mathcal{T}_v(\mathbf{I}))\right), & \text{otherwise} \quad \text{(Escalation)}
\end{cases}
\tag{3}
$$
The final slice-level diagnosis is given by $\hat{y} = \arg\max_c \bar{p}_c$. The complete procedure is detailed in Algorithm 1.

```
================================================================================
Algorithm 1: Multi-Scale Consistency Gated Test-Time Augmentation (MCG-TTA)
================================================================================
Input: Query MRI slice I, trained model f_\theta, confidence threshold \tau,
       scale perturbation factor \delta, escalation augmentation pool \mathcal{V}.
Output: Calibrated diagnostic class prediction \hat{y}.

 1: Compute base probabilities: p = \sigma(f_\theta(I))
 2: Compute confidence: p_max = \max_c p_c,  c_0 = \arg\max_c p_c
 3: if p_max \ge \tau then
 4:     return \hat{y} = c_0                                    // Fast-path exit
 5: end if
 6: Generate dual scale probes:
 7:     I_{s-} = Scale(I, 1 - \delta)
 8:     I_{s+} = Scale(I, 1 + \delta)
 9: Compute probe probabilities:
10:     p_{s-} = \sigma(f_\theta(I_{s-})),  c_- = \arg\max_c p_{s-, c}
11:     p_{s+} = \sigma(f_\theta(I_{s+})),  c_+ = \arg\max_c p_{s+, c}
12: if c_0 == c_- and c_0 == c_+ then
13:     \bar{p} = \frac{1}{3}(p + p_{s-} + p_{s+})              // Triad consensus
14: else
15:     \bar{p} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \sigma(f_\theta(\mathcal{T}_v(I))) // Disagreement escalation
16: end if
17: return \hat{y} = \arg\max_c \bar{p}_c
================================================================================
```

---

## 3.4. Analytical Latency and Computational Complexity Modeling

A primary objective of MCG-TTA is to bound execution latency for edge deployment. We formulate an analytical model characterizing expected runtime relative to baseline single-pass inference.

Let $T_{\text{base}}$ denote the inference latency of a single standard forward pass through the compact model. Let $\gamma = P(p_{\max} < \tau)$ denote the probability that an incoming slice triggers the uncertainty gate, and let $\beta = P(\text{scale disagreement} \mid p_{\max} < \tau)$ denote the conditional probability of scale disagreement. Triggered samples execute 3 forward passes under scale agreement, and an additional $|\mathcal{V}| - 3$ passes under disagreement.

The expected relative latency factor $\lambda = \frac{\mathbb{E}[T]}{T_{\text{base}}}$ is expressed as:
$$
\lambda = 1 + \gamma \cdot \left[ 2 + \beta \cdot (|\mathcal{V}| - 3) \right]
\tag{4}
$$

### Latency Bounds and Parameter Invariance
- **Single-Pass Inference**: Executes a single forward pass with constant relative latency $\lambda_{\text{base}} = 1.0$.
- **Unconditional Multi-View TTA**: Executes all $|\mathcal{V}|$ views uniformly across all samples, incurring a constant overhead factor $\lambda_{\text{full}} = |\mathcal{V}|$ [58].
- **MCG-TTA**: Bounded by $\lambda \le 1 + \gamma \cdot (|\mathcal{V}| - 1)$. As long as most inputs are confident ($\gamma \ll 1$) and exhibit scale agreement ($\beta \ll 1$), $\lambda$ remains close to unity, avoiding the prohibitive $|\mathcal{V}|\times$ multiplication of brute-force TTA.

Crucially, because MCG-TTA operates strictly at inference time by routing perturbed input views through the static model weights $\theta$, it introduces **zero additional learned parameters** ($|\Theta_{\text{MCG-TTA}}| = |\Theta_{\text{base}}|$).

---

## 3.5. Volumetric Patient-Level Diagnostic Aggregation

In clinical neuro-oncology, definitive diagnoses are rendered per subject across 3D volumetric examinations rather than isolated 2D slices. 

For a patient examination consisting of $S$ diagnostic axial slices $\{\mathbf{I}_s\}_{s=1}^S$, patient-level diagnostic probabilities are computed via unweighted mean-probability voting across all slices:
$$
\mathbf{P}_{\text{patient}} = \frac{1}{S} \sum_{s=1}^S \bar{\mathbf{p}}_s, \quad \hat{Y}_{\text{patient}} = \arg\max_{c} P_{\text{patient}, c}
\tag{5}
$$
Mean-probability aggregation integrates contextual evidence across the volumetric series, reducing slice-level variance and providing a robust subject-level diagnosis.
