


<link rel="stylesheet" href="*(Caption)*">

# The nDNA -- Cartography Across 15 Foundation Models


To chart the latent genomic landscape of modern foundation models, we choose a representative suite of large language models (LLMs) spanning diverse architectural paradigms, parameter scales, training objectives, and alignment methodologies. Our selection includes:

-   **Dense transformers** (e.g., *LLaMA-2* , *LLaMA-3* , *Gemma* , *Falcon* , *GPT-NeoX* )
-   **Sparse mixture-of-expert designs** (e.g., *Mixtral expert variants* )
-   **Multilingual and culturally calibrated models** (e.g., *Qwen base/instruct* )
-   **Compact efficient architectures** (e.g., *Phi-2* , *TinyLLaMA* )

These choices reflect contrasting scales  , pretraining corpora , multilingual coverage  , alignment regimes  , and distillation strategies  . Our goal is to trace how these factors sculpt each model's **nDNA**: its unique latent fingerprint of *semantic inheritance*, *epistemic adaptation*, and *ideological absorption*.

**Base variants** reflect the pretrained backbone: large autoregressive transformers trained on massive corpora for general language modeling objectives. Their latent geometry embodies inherited statistical priors---typically exhibiting smoother **spectral curvature** (
$$
\kappa_\ell
$$
), lower **thermodynamic length** (
$$
\mathcal{L}_\ell
$$
), and minimal directional strain from cultural priors (small
$$
\|\mathbf{v}_\ell^{(c)}\|
$$
).

**Instruct- and alignment-tuned variants**, in contrast, undergo reinforcement learning, instruction fine-tuning, or safety alignment  . These models show elevated
$$
\kappa_\ell
$$
,
$$
\mathcal{L}_\ell
$$
, and
$$
\|\mathbf{v}_\ell^{(c)}\|
$$
---particularly in upper decoder layers---indicating zones of *epistemic strain*, *latent reorientation*, and *cultural imprinting* necessary to align outputs with external value systems.

<div class="card">
  <h3>What we aim to uncover:</h3>
  <ul>
    <li>How alignment and instruction tuning inscribe  <em>persistent latent signatures</em>, distinguishing inherited traits from semantic mutations.</li>
    <li>Whether architectural form (dense vs. MoE) yields distinct geometric adaptation patterns (e.g., localized vs. distributed reconfiguration).</li>
    <li>Whether compact models preserve latent genomic complexity or collapse toward lower-dimensional manifolds with flattened nDNA signatures.</li>
  </ul>
</div>

<blockquote style="text-align: center; margin: 2em auto; border: none; background: transparent; font-size: 1.25em; font-weight: normal;">
As shown by the below table from the previous module, this latent geometry is not merely decorative--<strong>it is diagnostic</strong>.
</blockquote>
<figure id="tab:ndna_example" style="text-align: center; margin: 2em 0;">
  

  <div class="table-wrap">
    <table id="tab-ndna-table" role="table" aria-label="nDNA Geometric Signatures Table">
      <thead>
        <tr>
          <th class="col-center" scope="col">Layer</th>
          <th class="col-center" scope="col">$\kappa_\ell$</th>
          <th class="col-center" scope="col">$\mathcal{L}_\ell$</th>
          <th class="col-center" scope="col">$\|\mathbf{v}_\ell^{(c)}\|$</th>
          <th class="col-belief" scope="col">Belief Vector&nbsp;$\mathbf{v}_\ell^{(c)}$</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="col-center">20</td>
          <td class="col-center green-20">0.0412</td>
          <td class="col-center yellow-30">0.9123</td>
          <td class="col-center orange-20">0.6521</td>
          <td class="col-belief">[0.1204, -0.0502, 0.0896, ..., 0.0402]</td>
        </tr>

        <tr>
          <td class="col-center">21</td>
          <td class="col-center green-30">0.0458</td>
          <td class="col-center green-15">0.8123</td>
          <td class="col-center red-30">0.7523</td>
          <td class="col-belief">[0.1301, -0.0351, 0.0950, ..., 0.0431]</td>
        </tr>

        <tr>
          <td class="col-center">22</td>
          <td class="col-center yellow-20">0.0523</td>
          <td class="col-center orange-30">1.0120</td>
          <td class="col-center green-20">0.5823</td>
          <td class="col-belief">[0.1423, -0.0312, 0.0994, ..., 0.0488]</td>
        </tr>

        <tr>
          <td class="col-center">23</td>
          <td class="col-center orange-20">0.0581</td>
          <td class="col-center yellow-20">0.9021</td>
          <td class="col-center yellow-30">0.6912</td>
          <td class="col-belief">[0.1534, 0.0270, 0.1042, ..., 0.0512]</td>
        </tr>

        <tr>
          <td class="col-center">24</td>
          <td class="col-center red-20">0.0639</td>
          <td class="col-center red-40">1.1023</td>
          <td class="col-center green-15">0.5520</td>
          <td class="col-belief">[0.1667, 0.0205, 0.1105, ..., 0.0543]</td>
        </tr>

        <tr>
          <td class="col-center">25</td>
          <td class="col-center yellow-30">0.0505</td>
          <td class="col-center orange-20">0.9420</td>
          <td class="col-center red-40">0.8124</td>
          <td class="col-belief">[0.1602, -0.0251, 0.1081, ..., 0.0504]</td>
        </tr>

        <tr>
          <td class="col-center">26</td>
          <td class="col-center green-15">0.0398</td>
          <td class="col-center green-20">0.8520</td>
          <td class="col-center yellow-20">0.6120</td>
          <td class="col-belief">[0.1251, 0.0450, 0.0912, ..., 0.0418]</td>
        </tr>

        <tr>
          <td class="col-center">27</td>
          <td class="col-center yellow-20">0.0512</td>
          <td class="col-center red-30">1.0520</td>
          <td class="col-center orange-30">0.7222</td>
          <td class="col-belief">[0.1455, -0.0322, 0.1005, ..., 0.0477]</td>
        </tr>

        <tr>
          <td class="col-center">28</td>
          <td class="col-center orange-30">0.0590</td>
          <td class="col-center yellow-30">0.9320</td>
          <td class="col-center green-30">0.5721</td>
          <td class="col-belief">[0.1577, 0.0285, 0.1078, ..., 0.0499]</td>
        </tr>

        <tr>
          <td class="col-center">29</td>
          <td class="col-center red-30">0.0672</td>
          <td class="col-center orange-30">1.0123</td>
          <td class="col-center yellow-20">0.6322</td>
          <td class="col-belief">[0.1701, -0.0198, 0.1142, ..., 0.0533]</td>
        </tr>

        <tr>
          <td class="col-center">30</td>
          <td class="col-center orange-20">0.0555</td>
          <td class="col-center green-15">0.8221</td>
          <td class="col-center red-20">0.7720</td>
          <td class="col-belief">[0.1620, -0.0242, 0.1101, ..., 0.0510]</td>
        </tr>
      </tbody>
    </table>
  </div>

  <figcaption style="margin-top: 12px; font-size: 0.9em; color: #555; max-width: 1200px; margin-left: auto; margin-right: auto; text-align: left;">
<b>Table 1:</b> An <b>illustrative nDNA example</b> that captures the <em>semantic genome</em> of a foundation model through the joint interplay of <b>spectral curvature</b> ($\kappa_\ell$), <b>thermodynamic length</b> ($\mathcal{L}_\ell$), and <b>belief vector norm</b> ($\|\mathbf{v}_\ell^{(c)}\|$) across layers. Each of these quantities offers a distinct geometric and epistemic lens: $\kappa_\ell$ measures the <em>local acceleration</em> of latent representations, $\mathcal{L}_\ell$ quantifies the cumulative <em>internal work</em> required to traverse the belief manifold, while $\|\mathbf{v}_\ell^{(c)}\|$ encodes the <em>magnitude of cultural drift</em> imposed on latent activations. The <em>color intensities</em> shown alongside each value reflect relative magnitude within column-specific ranges: <span style="color: #27ae60;">■</span> low, <span style="color: #f1c40f;">■</span> moderate, <span style="color: #e67e22;">■</span> high, <span style="color: #e74c3c;">■</span> very high. For this example, spectral curvature spans $\kappa_\ell$ ∈ [0.0400, 0.0700], thermodynamic length $\mathcal{L}_\ell$ ∈ [0.80, 1.20], and belief vector norm $\|\mathbf{v}_\ell^{(c)}\|$ ∈ [0.55, 0.75]—revealing regions where the <em>latent manifold bends</em>, <em>epistemic energy intensifies</em>, or <em>external priors steer internal cognition</em>. This triad forms what we term the model's <b>nDNA</b>: a compact, high-dimensional <em>semantic fingerprint</em> that encodes the hidden geometry of belief. It enables us to diagnose zones of <em>inheritance stability</em>, detect <em>ideological absorption</em>, and trace <em>latent mutations</em> introduced by fine-tuning, alignment, or architectural choice. The pattern of these quantities across layers constitutes a signature as unique as a biological genome -- a map of how artificial cognition evolves, remembers, and adapts.
</figcaption>

</figure>

The pattern of
$$
\mathbf{\kappa_\ell}
$$
,
$$
\mathbf{\mathcal{L}_\ell}
$$
, and
$$
\| \mathbf{v}_\ell^{(c)} \|
$$
 across layers reveals:

-   **Semantic stability or reorientation**: Models that preserve pretrained priors display low curvature and thermodynamic cost across layers . Conversely, instruction-tuned or alignment-heavy models exhibit spikes in curvature and length  , marking latent restructuring.
-   **Zones of cultural pressure**: Peaks in belief vector norm
$$
\| \mathbf{v}_\ell^{(c)} \|
$$
 localize where cultural priors or alignment protocols most strongly steer internal cognition  .
-   **Inheritance fingerprinting**: The joint profile of these measures forms a signature--akin to a genomic sequence--allowing us to distinguish, compare, and trace the latent ancestry and adaptation pathways of models  .

In this sense, **nDNA is not a metaphor--it is a geometric genome**: an intrinsic latent encoding of how a model thinks, adapts, and inherits. Where biological DNA encodes traits through molecular structure, nDNA encodes them through the curvature, length, and directional flow of latent belief trajectories. This geometry defines not only what the model produces--but how it knows what it knows.

<figure id="fig:ndna_families" style="text-align: center; margin: 2em 0;">

  <!-- Row 1: LLaMA -->
  <div style="margin-bottom: 2em;">
    
  </div>

  <!-- Row 2: Gemma -->
  <div style="margin-bottom: 2em;">
    
  </div>

  <!-- Row 3: Mistral -->
  <div style="margin-bottom: 2em;">
    
  </div>

  <!-- Row 4: Deepseek -->
  <div style="margin-bottom: 2em;">
    
  </div>

  <!-- Row 5: Qwen -->
  <div style="margin-bottom: 2em;">
    
  </div>

  <!-- Row 6: Others -->
  <div style="margin-bottom: 2em;">
    
  </div>

  <!-- Final combined / all -->
  <div style="margin-bottom: 2em;">
    
  </div>

  <figcaption style="margin-top: 12px; font-size: 1.1em; color: #555; max-width: 1200px; margin-left: auto; margin-right: auto; text-align: left;">
     <b>Figure 1: (g) nDNA Landscape across 15 Foundation Models.</b> The composite visualization reveals striking <b>family-level clustering</b> in spectral-thermodynamic space, mapping how foundation models diverge in their latent genomic architecture. <b>High-strain models</b>--notably <b>Qwen</b> and <b>Mixtral</b>--consistently exhibit <b>spectral curvature</b> <i>κ<sub>ℓ</sub></i> exceeding 0.1 and <b>thermodynamic length</b> <i>L<sub>ℓ</sub></i> rising beyond 1.2 in upper decoder layers (<i>ℓ</i> ≥ 24). These profiles reflect aggressive latent reorganization driven by multilingual pretraining, expert routing, and intensive alignment adaptation--zones of <em>semantic strain</em>, <em>conceptual shock</em>, and <em>ideological absorption</em>. In contrast, <b>low-strain models</b> such as <b>Falcon</b>, <b>TinyLLaMA</b>, and <b>GPT-NeoX</b> form a distinct cluster where <i>κ<sub>ℓ</sub></i> < 0.03 and <i>L<sub>ℓ</sub></i> < 0.9, indicating smoother latent pathways that preserve pretrained epistemic structure with minimal reorientation. <b>LLaMA-3 Instruct</b>, <b>Gemma Instruct</b>, and <b>Deepseek Chat</b> occupy an intermediate zone--showing moderate curvature spikes (<i>κ<sub>ℓ</sub></i> peaking near 0.08) and thermodynamic gradients (<i>L<sub>ℓ</sub></i> up to 1.1)--highlighting selective reconfiguration in response to alignment and instruction tuning. This landscape provides a <em>geometric map of neural ancestry and adaptation</em>, illuminating inherited traits, semantic mutations, and the latent genomic signatures that distinguish foundation model families.
  </figcaption>
</figure>

---

## Why This Triad? On the Necessity of
$$
\kappa_\ell
$$
,
$$
\mathcal{L}_\ell
$$
 and
$$
\|\mathbf{v}_\ell^{(c)}\|
$$
 for nDNA Geometry

It may be tempting to argue that any pair or triplet of latent metricscould produce seemingly unique latent fingerprints when plotted layer-wise. Why, then, do we assert that the specific triad of **spectral curvature** (
$$
\kappa_\ell
$$
), **thermodynamic length** (
$$
\mathcal{L}_\ell
$$
), and **belief vector norm** (
$$
\|\mathbf{v}_\ell^{(c)}\|
$$
) is both minimal and sufficient for robust nDNA geometry?

<div class="content-showcase-container">

   <div class="showcase-card accent-green">
   <h3>Orthogonal yet complementary perspectives on latent dynamics.</h3>
   </div>

</div>

Each of the three measures captures a distinct, irreducible axis of the model's internal epistemic geometry:

- **
$$
\kappa_\ell
$$
 - The intrinsic semantic curvature of latent trajectories**  
  - how sharply the internal path of representations bends across depth. It encodes second-order structure, analogous to geometric curvature on manifolds.  

- **
$$
\mathcal{L}_\ell
$$
 - The cumulative epistemic work performed as the model adapts beliefs layer by layer**  
  - quantifying the energy expenditure needed for belief state transitions in the Fisher–Rao geometry of statistical manifolds.   

- **
$$
\|\mathbf{v}_\ell^{(c)}\|
$$
 - The directional cultural force acting upon the latent manifold**  
  - how much external priors or sociolinguistic constraints steer internal belief trajectories.   

Together, they span *latent shape* (curvature), *internal effort* (thermodynamics), and *external directional pressure* (belief vector field).

<div class="content-showcase-container">

  <div class="showcase-card accent-yellow">
    <h3>Other combinations evaluated and their limitations.</h3>
  </div>

</div>

We systematically experimented with numerous alternative metric sets to determine whether they could match or exceed the diagnostic power of this triad:

- **Norm-based pairs:** combinations like
$$
(\|h_\ell\|, \|\nabla_\theta h_\ell\|)
$$
, weight norms  , singular values of attention matrices , these collapse under trivial rescaling and layer normalization , offering little insight into geometric inflections or external directional forces. They reflect magnitude, not structure.
- **Gradient-only diagnostics:** Fisher information diagonal  , local logit gradients , these capture internal strain or sensitivity but fail to reveal latent manifold curvature or the directional drift imposed by external priors, leaving cultural or alignment effects hidden.
- **Entropy measures:** activation entropy , token probability entropy , valuable for quantifying output uncertainty or diversity, but disconnected from the internal geometric dynamics that govern latent inheritance or reorganization.
- **Pairings of curvature and local statistics:** attempts like (
$$
\kappa_\ell
$$
, activation variance) , (
$$
\mathcal{L}_\ell
$$
, ‖hₗ‖) fail to jointly encode latent shape, adaptation cost, and directional drift in a unified, interpretable manner. They fragment geometric, energetic, and external-force insights rather than synthesizing them.

None of these alternatives provided the geometric separability across model families (e.g., LLaMA vs. Mixtral vs. Qwen) nor the interpretability of zones of mutation, inheritance, and adaptation that our triad achieved.

<div class="content-showcase-container">

  <div class="showcase-card accent-blue">
    <h3>Effectiveness in revealing hidden geometry.</h3>
  </div>

</div>

What ultimately validates this triad is its empirical effectiveness in unveiling the hidden structural signatures of:

- **Finetuning and alignment:** zones where latent paths sharply reorient and effort spikes, e.g., LLaMA-3 Instruct vs. LLaMA-2 base  .
- **Cultural calibration:** regions where
$$
\|\mathbf{v}_\ell^{(c)}\|
$$
 reveals external value steering, e.g., Qwen instruct's ideological absorption .
- **Architectural specialization:** how MoE models like Mixtral partition latent space via curvature and effort redistribution .
- **Collapse and merging:** detection of flattening or hybridization of latent manifolds in model collapse and neural marriages  .

These are phenomena we rigorously map in the sections that follow, each tied to distinctive nDNA signatures visible only when these three axes are combined.

> ### Why not more metrics? Why not fewer?
> 
> Adding further dimensions (e.g., activation norms, entropy, variance) increased noise and reduced interpretability, without providing meaningful new axes of latent epistemic variation. Reducing to two metrics (e.g.,
$$
\kappa_\ell
$$
 and
$$
\mathcal{L}_\ell
$$
) failed to localize external cultural or alignment forces. The triad represents the minimal sufficient grammar to capture inheritance dynamics, as validated in [Table 1](#tab:ndna_example) and [Figure 1](#fig:ndna_families).

---

## Summary

The nDNA triad provides a latent genomic coordinate system:
$$
\boxed{ \underbrace{\text{Intrinsic curvature}}_{\kappa_\ell} \quad + \quad \underbrace{\text{Epistemic effort}}_{\mathcal{L}_\ell} \quad + \quad \underbrace{\text{External steering}}_{\|\mathbf{v}_\ell^{(c)}\|} \quad \Rightarrow \quad \text{nDNA: a unique fingerprint of neural inheritance.}}
$$


Its power lies not only in theoretical soundness  , but in its empirical capacity to disentangle inherited traits, zones of mutation, and ideological drift that no arbitrary metric combination could replicate.

While belief vectors are the core focus, one could consider extending to belief tensors for richer representation; however, this would significantly increase computational cost and complexity.

---


