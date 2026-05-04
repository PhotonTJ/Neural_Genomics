---
layout: page
title: The nDNA Cartograph - Latent Semantic Genome of Foundation Models
permalink: /llm/ndna/
skip_title: true
---

{% include ndna-title.liquid 
   title="The nDNA Cartograph: Latent Semantic Genome of Foundation Models" 
   video_id="rddewsgbuKw?listType=playlist&list=PLaE3N7Ax00P93HC0jxZ-b6c4tnLkldPOJ" %}

**Before we unveil *nDNA*, we must confront a foundational question:** *What qualifies as heritability in artificial cognition*? Conventional artifacts--*weights*, *activations*, or the output *behavior*--are mere **epiphenomena of training**. In contrast, **nDNA** seeks to capture a model's *semantic genome*: the *latent organizational structures* that govern how knowledge is internally *represented*, *adapted*, and *transmitted* across fine-tuning, distillation, pruning, and deployment. To chart the **semantic ancestry** of AI systems, we must move beyond output-level metrics and embrace a deeper epistemic foundation--one that traces not just what models *say*, but how they *reason*, *evolve*, and *remember*. We argue that **nDNA** constitutes this missing genomic trace: a **structured latent fingerprint** of artificial cognition. Just as molecular genetics enabled biology to transcend surface taxonomies and uncover causal mechanisms, we contend that a *genomic lens* is now essential for machine learning--one that can **quantify**:

<img src="{{ '/assets/img/ndna_logo_v5.jpg' | relative_url }}" alt="nDNA" style="width: 100%">

{% capture ndna_quantify %}
- **Layer Importance and Semantic Specialization**: Not all layers contribute equally to a model's epistemic structure. A growing body of evidence {% cite belrose2023mechanistic geva2022transformer dai2023knowledge liu2023hidden %} reveals that semantic representations, cultural memory, and alignment behavior disproportionately concentrate in the mid-to-upper transformer layers--particularly the final 10 layers in ~30-layer models. These layers encode more than surface patterns; they carry deep *semantic priors* and value shifts induced by alignment, fine-tuning, and cultural adaptation. For **nDNA** to serve as a meaningful genomic diagnostic, it must trace inheritance, drift, and trait transformation across these epistemically sensitive regions.

- **Semantic Drift and Heritable Traits**: Subtle misalignments and persistent divergences--documented in alignment studies {% cite zhou2023alignmentdrift ganguli2023reducing %}--can occur even when models appear behaviorally consistent. These are not superficial perturbations but inheritable epistemic traits passed along neural offspring {% cite wu2024seamless xu2023aligning %}.

- **Value Simulation vs. Internalization**: As models grow more context-sensitive, they learn to *simulate* alignment without embodying its values {% cite jacobs2024evalaware perez2022discovering %}. Disentangling true normative internalization from strategic mimicry is essential for any meaningful epistemic inspection.

- **Plasticity and Collapse**: Aggressive fine-tuning, distillation, or ideological merging can induce *plasticity collapse*--a reduction in epistemic flexibility and semantic richness {% cite liu2023lost bai2023constitutional %}. This demands metrics that trace both robustness and degeneration over time.

- **Latent Cultural Conflict**: In multilingual or cross-cultural settings, models often encode conflicting or incoherent value systems {% cite mukherjee2024inconsistency chen2023helpfulness %}. These conflicts are not visible through BLEU or ROUGE--they reside in the model's latent belief structure and must be surfaced through geometric lineage analysis.

- **Topological Continuity**: Alignment and fine-tuning warp the internal geometry of models in nontrivial ways {% cite chiang2023can liu2023hidden %}. **nDNA** must preserve continuity and interpretability of trajectories across such transformations.

- **Epistemic Mutation**: Merging preferences, annotator distributions, or learned behaviors--as explored by {% cite bakker2024uniting %}--creates emergent traits that standard metrics cannot track. These mutations are only diagnosable through a genomic lens on representation evolution.
{% endcapture %}

<style>
  .ndna-quantify li::marker {
    content: "✦  ";
    font-size: 1.2em;
  }
</style>

<blockquote class="ndna-quantify">
  {{ ndna_quantify | markdownify }}
</blockquote>

**nDNA** empowers us to interrogate the *hidden geometry* of learning--revealing how foundational operations such as **alignment**, **fine-tuning**, **quantization**, **pruning**, and **multilingual fusion** subtly but systematically reshape a model's *semantic core*. It uncovers **cultural instabilities** introduced through regional adaptation, traces **asymmetric inheritance** patterns across neural offspring, visualizes **latent reorganizations** induced by merging or distillation, and quantifies a model's capacity to *resist* or *absorb* conflicting epistemic pressures.

These phenomena--often dismissed as quirks--are in fact *heritable traits*, etched into the model's internal manifold. When viewed through this lens, *model collapse*, *alignment-induced drift*, and *semantic mimicry* cease to be incidental failures and instead emerge as structural signatures of deeper latent dynamics. **nDNA** thus transcends metaphor to become a **scientific grammar** for measuring *epistemic resilience*, *semantic coherence*, *cultural consistency*, and *trait inheritance*--offering a principled lens through which to **govern**, **understand**, and **audit** the evolving anatomy of artificial cognition.

We further posit that **cultural provenance** induces a distinct *layerwise calibration effect*, predominantly localized in the final decoder layers $$\ell \in [20, 30]$$, where sociolinguistic priors exert the strongest influence on output distribution. To capture this, we introduce the **nDNA Score**--a composite diagnostic unifying: **(i)** *Spectral curvature* $$\kappa_\ell$$, reflecting the compression and warping of conceptual flow; **(ii)** *thermodynamic length* $$\mathcal{L}_\ell$$, quantifying the epistemic effort required to traverse belief transitions; and **(iii)** the norm of the *Belief Vector Field* $$\lVert\mathbf{v}_\ell^{(c)}\rVert$$, measuring the directional intensity of latent cultural drift.

Together, these dimensions form a latent semantic fingerprint--a high-dimensional, biologically inspired signature of internal cognition--enabling us to **trace**, **compare**, and **govern** the *neural evolution* of foundation models with unprecedented granularity.


---

## Rationale and Formalization: Why Trajectories, Not Weights

The usual levers for interpreting and governing LLMs—parameter counts, sparsity patterns, attention heatmaps—live in coordinates that are *non-identifiable* and only weakly tethered to deployed behavior. Permutations, rotations, and low-rank re-expressions can leave the realized function intact while scrambling weight-level narratives {% cite garipov2018loss draxler2018essentially li2018visualizing entezari2022role ainsworth2023git wortsman2022model %}. By contrast, what remains stable under such reparameterizations is the *on-input computation*: for a prompt x, the forward pass traces a **trajectory of hidden states** through depth. Endowing representation space with information geometry (e.g., **Fisher--Rao** pullbacks) yields **coordinate-free** notions of distance, bending, and effort that track changes in the output law {% cite efron1975defining amari2000methods amari2016information %}. We read this as **semantic hydrodynamics**: **meaning is transported** through layers like a fluid through a shaped conduit.

### Limits of weight-space and attention views

Weight-space indicators (parameter counts, sparsity, individual neurons/heads) live in *non-identifiable* coordinates: permutations, rotations, or refactorings can leave behavior unchanged while rewriting any weight-level narrative. Attention maps are largely *descriptive*, not reliably causal or stable—different patterns can yield the same outputs and head roles drift across training. These limits motivate a **behavior-first**, **coordinate-free** view that reads the model's *on-input trajectory* of representations, rather than static weights or raw attention.

**Weight space is non-identifiable and behavior-misaligned.** Permutation symmetries, rotations, and low-rank re-expressions can preserve the function while scrambling weight-level narratives. Empirically, independently trained solutions are often *mode-connected* by low-loss paths or become connected after accounting for permutations, undermining explanations that cling to specific coordinates {% cite garipov2018loss draxler2018essentially li2018visualizing entezari2022role ainsworth2023git %}. Moreover, practical levers like **weight averaging/model soups** alter parameters while leaving deployed behavior similar or improved, again decoupling "where weights sit" from *what the model does* {% cite wortsman2022model %}. In short, **we deploy behaviors, not weights**; coordinate-specific stories are fragile.

**Attention is informative but not a faithful, stable mechanism by itself.** Extensive tests show that *similar outputs can arise from disparate attention patterns*, and directly perturbing attention often leaves predictions largely unchanged; hence attention weights are, at best, *descriptive* {% cite jain2019attention serrano2019attention %}. Redundancy and role-drift are common: many heads can be pruned with little loss, a few heads do the "heavy lifting," and head functions shift across training or fine-tuning, weakening governance value of raw maps {% cite michel2019sixteen voita2019analyzing clark2019bert kovaleva2019revealing %}. Post-hoc corrections (e.g., *attention flow/rollout*) improve alignment with token importance but still treat attention as *signals*, not ground-truth causes {% cite abnar2020quantifying %}. Beyond attention, **critical computation lives in MLPs**: feed-forward layers behave like *key–value memories* that store and retrieve factual associations, so attention alone under-specifies mechanism {% cite geva2021ffkv %}. Methodologically, the broader saliency literature warns that visually plausible explanations can fail *sanity checks*, and "faithfulness" must be defined and evaluated explicitly {% cite adebayo2018sanity jacovi2020faithfulness %}.

**What is stable: the on-input trajectory.** For each prompt x, the forward pass traces a depth-indexed path of hidden states—the operational object we actually deploy. Prior analyses show that linguistic competencies emerge layerwise in consistent *pipelines* (POS → parsing → NER → SRL → coreference), supporting the intuition that the *trajectory through representation space* is a robust behavioral signature {% cite tenney2019bert clark2019bert %}. This motivates nDNA's choice to work in **trajectory space**, not parameter space.

### We read AI foundation models as semantic fluid–dynamics

{% include visualization-html.liquid 
   video_path="semantic_hydrodynamics/llm_action_2.mp4"
   alt="nano-gpt (structure) Visualization"
   caption="**nano-gpt (*structure*).** *Architecture as channel blueprint:* depth acts like the axial coordinate; **residuals** $\leftrightarrow$ *bypass pipes*; **attention** / **MLP** blocks act as *mixers/valves* that locally reshape the flow of representations."
   %}

<div style="display: flex; gap: 1rem;">
<div style="width: 50%">
{% include visualization.liquid 
   image_path="semantic_hydrodynamics/flow-simulation.gif"
   alt="Flow simulation Visualization"
   caption="**Flow simulation (*analogue*).** *Fluid:* colored streamlines show speed through a bend and throat---**curvature** rises, **shear** increases, small *recirculation* pockets may form. *Semantic:* bends $\Rightarrow$ **spectral curvature** spikes ($\kappa$); constrictions $\Rightarrow$ **thermodynamic length** bursts ($\Delta L$); eddies $\Rightarrow$ local rotation in the **belief field** ($\nabla\times\mathbf{v}$)."
   height="410px"
   %}
</div>
<div style="width: 50%">
{% include visualization.liquid 
   image_path="semantic_hydrodynamics/pipe.jpg"
   alt="Pipeline metaphor (macro view)"
   caption="**Pipeline metaphor (*macro view*).** *Geometry governs transport:* routing capacity and effort depend on the network of ducts. *Semantic:* model design / fine-tuning shapes **where meaning flows easily**, **where it pays**, and **where it recirculates**."
   height="410px"
   cover_image=true
   %}
</div>
</div>

{% include visualization-html.liquid 
   caption="**Semantic hydrodynamics.** ***Model.*** We read the forward pass as *semantic hydrodynamics*: a prompt injects *semantic mass* that is transported through depth like a fluid through a shaped channel. ***Why.*** Weight/attention coordinates can change without altering behavior; the *on-input flow* provides **behavior-first**, **coordinate-free** signals. ***Reading guide.*** **Bend** $\to$ *spectral curvature* $\kappa$ (sharp reroutes vs. laminar refinement); **Pay** $\to$ *thermodynamic length* $L$ (where the model expends effort; $\Delta L$ bursts mark *bottlenecks*); **Push** $\to$ *belief field* $\mathbf{v}$ (direction/magnitude of local drive; eddies indicate *recirculation*). ***Benefit.*** The same metaphor specifies **where to measure**---*bends*, *throats*, and *eddies*---turning inner computation into **actionable diagnostics** and **governance thresholds**."
   %}

### Why semantic hydrodynamics matters

- **We govern *behavior*, not coordinates.** Operational concerns—*robustness, safety, bias, faithfulness*—attach to what the model **does** on an input, not to how its weights are labeled. Two checkpoints can behave the same while their parameters and attention differ. In short: **the weights are the map; the trajectory is the territory**.

- **Invariance beats introspection.** Coordinate-bound stories change under neuron permutations, subspace rotations, or low-rank refactorings; the **path an input carves** and its **geometry** (length, curvature, alignment) are **invariant** because they are measured by **how predictions would change**, not by which index moved.

- **Geometry turns cognition into observables.** An information metric acts as local **stiffness**: soft directions barely affect the output; stiff directions swing the predictive law. With that ruler, we quantify **how far** the model travels to reshape belief (thermodynamic length L), **where** it turns its internal argument (spectral curvature κ), and **what** pushes change locally (belief field **v**) {% cite sivak2012thermodynamic hyvarinen2005estimation %}.

- **The hydrodynamics metaphor is operational.** Like fluid in a
  channel, semantic flow shows **corners, constrictions, and eddies**:
  sharp bends $\Rightarrow$ high $\kappa$; narrow throats $\Rightarrow$
  bursts in $\Delta L$; local recirculation $\Rightarrow$ rotational
  structure in $\mathbf{v}$. These are **measurable**, per-layer signals
  on the actual computation.

***What this buys us* (concrete payoffs).**

- **Behavior-first invariance.** Reading $\kappa$, $L$, and $\mathbf{v}$
  on the trajectory yields **fingerprints** that are **comparable**
  across models, seeds, and checkpoints---even when weights or head
  roles reshuffle.

- **Local diagnostics.** **$\kappa$ spikes** flag brittle decision
  pivots; **$\Delta L$ bursts** expose capacity bottlenecks or lossy
  transformations; **low alignment** (small $\cos\theta$ between
  $\mathbf{v}$ and the tangent $\mathbf{T}$) marks layers that **move
  without updating belief** (staging or detours).

- **Governance hooks.** **Geometry budgets and thresholds**---max
  $\kappa$, allowable $\Delta L$ per slice, minimum alignment---become
  **pre-release gates**; nDNA fingerprints support **drift monitoring**
  after fine-tuning, pruning, quantization, or alignment.

- **Comparative forensics.** Because $\kappa/L/\mathbf{v}$ are tied to
  the output law, we can **attribute performance deltas** to **where in
  depth** the flow changed (e.g., a new bend from fine-tuning, an effort
  spike from quantization) instead of to unstable weight indices.


**Rule of thumb:** If the goal is to **explain**, **compare**, or **govern** deployed behavior, analyze the **flow of meaning** that the input actually experiences. In nDNA: **curvature** says *where it bends*, **thermodynamic length** says *how much it pays*, and the **belief field** says *what pushes it*—all with a ruler calibrated to the model's own predictions.

<div style="display: flex; gap: 1rem;">
<div style="width: 50%">
{% include visualization.liquid 
   image_path="semantic_hydrodynamics/laminar_flow.png"
   alt="Laminar flow Visualization"
   caption="**Laminar flow.** *Fluid:* viscous--dominated, low--Re regime; nearly
   parallel streamlines, negligible cross--stream mixing, no
   recirculation.

   **LLM Semantic Flow:** uniformly low spectral curvature $\kappa$, small
   steady $\Delta L$, and high alignment between the step and the belief
   push (steady refinement)."
   height="auto"
   %}
</div>
<div style="width: 50%">
{% include visualization.liquid 
   image_path="semantic_hydrodynamics/spectral_flow.png"
   alt="Spectral curvature Visualization"
   caption="**Spectral curvature ($\kappa$) --- turbulent.** *Fluid:* a bend induces
   sharp turning, higher shear, possible separation.

   **LLM Semantic Flow:** a localized $\kappa$ spike at the turning point
   marks a sharp reroute in representation space; quasi--linear segments
   before/after indicate a discrete semantic pivot (e.g., topic jump,
   shortcut, policy jolt)."
   height="auto"
   %}
</div>
</div>

<div style="display: flex; gap: 1rem;">
<div style="width: 50%">
{% include visualization.liquid 
   image_path="semantic_hydrodynamics/thermodynamics_flow.png"
   alt="Thermodynamic length Visualization"
   caption="**Thermodynamic length ($L$).** *Fluid:* a constriction raises shear and
   pressure drop; energy dissipates fastest in the throat.

   **LLM Semantic Flow:** a stiffer metric band (hatched) and a rise in
   $\Delta L$ reveal a bottleneck where extra *semantic effort* is paid to
   reshape belief (friction, detours, boundary crossing)."
   height="auto"
   %}
</div>
<div style="width: 50%">
{% include visualization.liquid 
   image_path="semantic_hydrodynamics/belief_vector_flow.png"
   alt="Belief field Visualization"
   caption="**Belief field ($\mathbf{v}$).** *Fluid:* the velocity field sets
   transport; eddies (local curl) mark recirculation; alignment with
   streamlines indicates efficient conveyance.

   **LLM Semantic Flow:** $\mathbf{v}$ is the local push that most steeply
   changes the output law; longer arrows $\Rightarrow$ larger
   $\|\mathbf{v}\|$, and the side gauge shows $\cos\theta$ between
   $\mathbf{v}$ and the path tangent $\mathbf{T}$; circular loops on waves
   depict local recirculation that can trap or reinforce beliefs."
   height="auto"
   %}
</div>
</div>

{% include visualization-html.liquid 
   caption="**LLM as an input$\to$output semantic channel.** *Model:* we read the
   forward pass as *semantic hydrodynamics*---a prompt injects semantic
   mass that is transported through depth like a fluid through a shaped
   conduit. **Bend** (*top row*): curvature $\kappa$ distinguishes
   *laminar* refinement from *sharp* reroutes. **Pay** (*bottom left*):
   thermodynamic length $L$ localizes where effort concentrates via
   $\Delta L$ bursts (*bottlenecks*). **Push** (*bottom right*): the belief
   field $\mathbf{v}$ reveals whether a layer update directly *advances
   belief* (high alignment) or *reorganizes information* (low alignment);
   eddies signal *local recirculation*.

   **Why this lens:** weight--space and attention views are
   *non--identifiable* and unstable across checkpoints; nDNA instead reads
   the *on--input trajectory* and its information geometry, yielding
   *coordinate--free*, behavior--first measurements.

   **Vision:** treat inner computation as a *measurable flow* so that
   bends, effort, and push become quantifiable traits of
   cognition---comparable across inputs, layers, models, and training
   phases.

   **Benefits:** *actionable diagnostics*---$\kappa$ spikes flag brittle
   turns, $\Delta L$ bursts expose capacity bottlenecks, low $\cos\theta$
   (between $\mathbf{v}$ and the tangent $\mathbf{T}$) indicates movement
   that does not immediately update belief; *stable
   comparability*---geometry--based fingerprints are robust to neuron
   permutations and head--role drift; *governance hooks*---set thresholds
   on $\kappa$ or $\Delta L$, track fingerprint drift after
   fine--tuning/pruning, and audit capacity before release."
   %}

---

## Spectral Curvature $$(\kappa_\ell)$$: A Geometric Lens on Latent Bending

**What is spectral curvature?** In classical geometry, curvature quantifies how much a path deviates from being straight--measuring local bending of a trajectory. In **spectral geometry** and **harmonic analysis**, curvature extends to how signals or paths behave in frequency space or under operators that encode structure (e.g., Laplacians, difference operators). *Spectral curvature* refers to curvature derived through such operators--capturing the *shape of latent signals* as they evolve across layers of a model.

**Why spectral for latent manifolds?**
In foundation models, hidden representations form a sequence of activations $$\{h_\ell\}_{\ell=0}^L$$ across layers. These representations trace a path in high-dimensional latent space. The *shape* of this path encodes the model's **internal conceptual flow**--how its beliefs evolve as it integrates priors, inputs, and alignment constraints. **Spectral operators** (such as discrete Laplacians or difference operators) naturally quantify how this path bends or accelerates--making them ideal for probing internal geometry. Unlike mere distance measures, *spectral curvature* reflects **intrinsic shape**, invariant under reparameterization.

**Formulation and derivation.**
Consider hidden activations $$h_\ell \in \mathbb{R}^d$$ at each layer $$\ell$$. The **first-order difference**

$$\Delta h_\ell := h_\ell - h_{\ell-1}$$

approximates the local directional change of latent states--a discrete analogue of *velocity* in latent space.

To capture bending, we compute the change in this directional flow--the **second-order difference**:

$$\Delta^2 h_\ell := \Delta h_{\ell+1} - \Delta h_\ell = (h_{\ell+1} - h_\ell) - (h_\ell - h_{\ell-1}) = h_{\ell+1} - 2h_\ell + h_{\ell-1}$$

This operator acts like a *discrete Laplacian* along the latent path, highlighting where the model's **internal belief flow** deviates from a straight trajectory.

> **Spectral curvature at layer $$\ell$$ is defined as:**
> 
> $$\kappa_\ell := \lVert\Delta^2 h_\ell\rVert = \lVert h_{\ell+1} - 2h_\ell + h_{\ell-1}\rVert$$

In continuous form, this corresponds to:

$$\kappa(s) = \left\lVert\frac{d^2 h(s)}{ds^2}\right\rVert$$

where $$s$$ parameterizes depth through the network. Our discrete $$\kappa_\ell$$ provides a practical, layerwise estimator.

**Why is this meaningful?**
Peaks in $$\kappa_\ell$$ mark layers where internal geometry is most dynamic--zones of *semantic inflection*, *belief compression*, or *ideological absorption*. These are the structural signatures of **internal epistemic adaptation**, essential to trace cultural inheritance and alignment drift.

**Lineage and context.**
Spectral curvature builds on tools from **geometric deep learning**, **equivariant architectures**, **Ricci flow in machine learning**, and **spectral graph analysis** {% cite farzam2024ricci cho2023mixedcurvature gasteiger2021gemnet xu2022spherical konf2021hierarchical ying2021equivariant hu2022lie hess2023spectral wang2021geomtransformer raposo2023spectral %}. Within **nDNA**, it serves as a **principled geometric fingerprint**--revealing not only *what* is encoded, but *how* internal belief pathways are reshaped to encode it.

{% capture spectral_curvature_caption %}
**Spectral Curvature** $$(\kappa_\ell)$$ quantifies second-order deviations in latent representations across transformer layers--computed via the discrete geometric operator $$\kappa_\ell := \lVert h_{\ell+1} - 2h_\ell + h_{\ell-1}\rVert$$. High curvature signals *semantic inflection points* where internal geometry bends sharply--often in **culturally dense**, **ideologically loaded**, or **epistemically volatile** regions. Peaks in $$\kappa_\ell$$ typically emerge in upper decoder layers $$(\ell \in [21,30])$$, where the model accommodates sociolinguistic priors during alignment, multicultural or multilingual fusion. Within the **nDNA** framework, such curvature reflects *latent inheritance dynamics*, offering a fine-grained geometric fingerprint of representational restructuring.
{% endcapture %}

{% include visualization-html.liquid 
   image_path="ndna/spectral_curvature.gif"
   alt="Spectral Curvature Analysis" %}

{% include visualization-html.liquid 
   interactive_html="ndna/spectral_curvature_static_last_layer_FINAL.html"
   image_path="ndna/spectral_curvature_llama_ndna_animation.gif"
   alt="3D Spectral Curvature Visualization"
   caption=spectral_curvature_caption
   full_width=true %}

## Thermodynamic Length $$(\mathcal{L}_\ell)$$: Epistemic Effort Across Layers

**What is thermodynamic length?**
In **statistical thermodynamics** and **information geometry**, *thermodynamic length* measures the cumulative effort--or "*work*"--required for a system to transition between states on a statistical manifold. It integrates local gradient energy along a trajectory, providing an *intrinsic cost measure* that is independent of parametrization.

**Why thermodynamic length for foundation models?**
In foundation models, layers trace a **path through latent belief space**. As input data and alignment priors reshape activations, the model expends internal **computational effort** to adjust its belief state. *Thermodynamic length quantifies this latent effort* -- measuring not just *what* the model knows, but *how hard* it works to adapt that knowledge across layers in response to epistemic pressures (e.g., cultural fusion, alignment shifts).

**Mathematical intuition.**
Let $$h_\ell$$ denote the latent state at layer $$\ell$$, and $$\mathcal{M}$$ the model's latent manifold. Layer transitions define a curve $$\gamma: [0,L] \to \mathcal{M}$$ whose thermodynamic length is

> $$\mathcal{L}(\gamma) = \int_0^L \sqrt{\langle \dot{\gamma}(s), \mathcal{G}_{\text{Fisher}} \dot{\gamma}(s) \rangle} \, ds$$

where $$\mathcal{G}_{\text{Fisher}}$$ is the Fisher information metric. Here, $$\mathcal{L}(\gamma)$$ represents the *intrinsic work* needed to traverse $$\gamma$$ on $$\mathcal{M}$$.

**Interpretation.**
High thermodynamic length indicates regions where latent geometry **stretches** -- where the model's belief space undergoes substantial reconfiguration to reconcile priors and input. This formalism reveals *not just where* latent states change, but the *cost structure of that change*. Zones of large $$\mathcal{L}_\ell$$ mark points of **alignment tension**, **cultural fusion**, or **complex reasoning**, where internal scaffolds are under maximum stress.

*Thermodynamic length offers a window onto the model's "latent energy budget" -- illuminating how internal belief states reshape to meet complexity, constraint, and context.*

### Mathematical Definition

Let $$p_\ell(y\mid x)$$ denote the model's conditional distribution at layer $$\ell$$ given input $$$$x. The local epistemic cost is reflected in the squared norm of the gradient of log-likelihood with respect to model parameters:

$$\lVert\nabla_\theta \log p_\ell(x)\rVert^2$$

This quantity measures how much the model must *adjust its parameters locally* at layer $$\ell$$ to improve its fit to input $$$$x. *Thermodynamic length at layer $$\ell$$* aggregates this cost across the dataset $$\mathcal{D}$$:

> **Thermodynamic length at layer $$\ell$$ is defined as:**
> 
> $$\mathcal{L}_\ell := \sum_{x \in \mathcal{D}} \lVert\nabla_\theta \log p_\ell(x)\rVert^2 = \lvert\mathcal{D}\rvert \cdot \mathbb{E}_{x \sim \mathcal{D}} \lVert\nabla_\theta \log p_\ell(x)\rVert^2$$

This formulation reveals that $$\mathcal{L}_\ell$$ captures both the *average local effort* and its scaling with dataset size. Furthermore, in differential geometric terms, thermodynamic length can be written as a path energy:

$$\mathcal{L}_\ell = \int_{\gamma_\ell} \left\langle \frac{dh_\ell}{ds}, \mathcal{G}_{\text{Fisher}}(h_\ell) \frac{dh_\ell}{ds} \right\rangle ds$$

where $$h_\ell$$ denotes latent trajectories at layer $$\ell$$, $$\mathcal{G}_{\text{Fisher}}$$ the Fisher information metric, and $$s$$ arc length along $$\gamma_\ell$$. Thus, $$\mathcal{L}_\ell$$ can be seen as an *energy integral over the belief manifold* -- capturing how much internal "*heat*" or computational work is generated to reconcile prior belief state with new input at depth $$\ell$$.

**Why is this meaningful?**
Unlike static capacity metrics or weight magnitudes, $$\mathcal{L}_\ell$$ is *dynamically grounded*: it measures where the model actively strains to reconcile competing epistemic demands. In regions of high $$\mathcal{L}_\ell$$, the model's **latent geometry** is under tension--*reshaping itself* to accommodate alignment constraints, cultural priors, or multilingual semantics.

**Lineage and context.**
This diagnostic builds on the **Fisher–Rao metric** in **information geometry** and **thermodynamic length formalism** from statistical physics {% cite crooks2007measuring oliviero2023thermodynamics farzam2024ricci wagner2023thermodynamic %}. Thus **nDNA** provides a *complementary view* to spectral curvature--capturing not where the model bends, but *how hard it works* to do so. Together, these axes form a **neurogeometric anatomy** of latent belief adaptation.

{% capture thermo_length_caption %}
**Thermodynamic Length** $$\mathcal{L}_\ell := \sum_{x \in \mathcal{D}} \lVert\nabla_\theta \log p_\ell(x)\rVert^2$$ quantifies the *epistemic work* performed across transformer layers, calculated as the **cumulative squared gradient norm of layerwise log-likelihoods**. Higher values signal *internal resistance*--zones of significant restructuring, belief compression, or negotiation of conflicting priors. In culturally fine-tuned models, these peaks localize to upper decoder layers, indicating intense adaptation near output-generating blocks. Within the **nDNA** construct, $$\mathcal{L}_\ell$$ helps reveal latent epistemic effort that underlies surface-level behavior. This metric thus provides a nuanced window into where and how models internally allocate effort during learning and inference.
{% endcapture %}

{% include visualization-html.liquid 
   image_path="ndna/thermodynamic_length.gif"
   alt="Thermodynamic Length Analysis" %}

{% include visualization-html.liquid 
   interactive_html="ndna/thermodynamic_length_static_last_layer.html"
   image_path="ndna/Final_thermodynamic_length.gif"
   alt="3D Thermodynamic Length Visualization"
   caption=thermo_length_caption
   full_width=true %}

## Belief Vector Field $$(\mathbf{v}_\ell^{(c)})$$: Cultural Drift in Latent Space

**What is the Belief Vector Field**--
In **differential geometry** and **physics**, a *vector field* describes a directional force applied at each point of a space. Inspired by this, the **Belief Vector Field** models the *directional semantic force* that a specific culture or value system exerts on a model's latent representations. It encodes *where*, *how strongly*, and *in what direction* cultural priors act within the model's internal geometry--functioning as a **semantic compass** through the latent manifold.

**Why a vector field for cultural influence?**
While **spectral curvature** $$(\kappa_\ell)$$ captures how sharply latent paths bend, and **thermodynamic length** $$(\mathcal{L}_\ell)$$ how hard the model works during adaptation, neither tells us the *source*, *direction*, or *origin* of that adaptation. The Belief Vector Field offers this missing piece: it traces the latent steering applied by culture-conditioned priors--*where the model is being pushed in latent space, by what epistemic force, and toward which semantic direction*. This makes it a critical diagnostic for studying **cultural drift**, **ideological imprinting**, and **alignment tension**.

### Mathematical Definition

Let $$p(y\mid x)$$ denote the model's conditional output distribution for input $$$$x, and let $$h_\ell$$ be the latent representation at layer $$\ell$$. The local belief gradient, $$\nabla_{h_\ell} \log p(y\mid x)$$, measures how a small change in $$h_\ell$$ would affect output confidence--a proxy for *semantic force* at that layer. To extract the culturally conditioned semantic force, we compute its expectation over a culture-specific distribution $$\mathcal{P}^{(c)}$$:

> **Belief vector field at layer $$\ell$$ for a given manifold condition is defined as:**
> 
> $$\mathbf{v}_\ell^{(c)} := \mathbb{E}_{x \sim \mathcal{P}^{(c)}} [\nabla_{h_\ell} \log p(y\mid x)]$$

where $$\mathcal{P}^{(c)}$$ represents inputs emblematic of given manifold condition $$c$$ (e.g., regional, linguistic, ideological contexts). This formulation captures not just latent deformation, but *its cause*: how cultural priors exert directional influence within the belief manifold.

**Why is this meaningful?**
$$\mathbf{v}_\ell^{(c)}$$ provides a directional lens on latent dynamics. High $$\lVert\mathbf{v}_\ell^{(c)}\rVert$$ signals regions where the model is *actively redirected* by external cultural forces--offering diagnostic power for detecting **ideological drift**, **semantic conflict**, or **bias inheritance**. Unlike $$\kappa_\ell$$ or $$\mathcal{L}_\ell$$, which capture internal geometry, $$\mathbf{v}_\ell^{(c)}$$ reveals *external epistemic pressure* and its directional impact.

**Lineage and context.**
This diagnostic builds upon belief geometry, alignment drift studies, and cultural bias tracing in NLP {% cite wang2023culturalbias zhou2023alignmentdrift shen2023beliefgeometry arora2023stereoset bommasani2023foundation peng2024cultural laurens2024anthropic kang2024biasfairness de2023latentbias gao2023value %}. Within the **nDNA** construct, it integrates with curvature and length to offer a holistic neurogeometric portrait--revealing *how*, *why*, and *where* foundation models inherit, adapt, or distort beliefs under cultural influence.

**Interpretability in practice.**
By mapping $$\mathbf{v}_\ell^{(c)}$$ across layers and cultures, we can trace **cultural provenance**, identify **ideological pressure zones**, and diagnose **inheritance asymmetry** in multilingual or aligned models. This directional fingerprint informs audits of model bias, robustness, and alignment integrity--providing the missing vectorial dimension in understanding machine cognition.

{% capture belief_vector_caption %}
**Belief Vector Field Visualization**: $$\mathbf{v}_\ell^{(c)} = \mathbb{E}_{x \sim \mathcal{P}_{\text{CIVIC}}^{(c)}} [\nabla_{h_\ell} \log p(y\mid x)]$$ represents the *belief semantic steering force* at layer $$\ell$$ toward concept $$c$$, conditioned on CIVIC cultural priors. **Large magnitudes** (e.g., $$\lVert\mathbf{v}_\ell^{(c)}\rVert \in [0.15, 0.50]$$) indicate *strong directional pressure*--zones where cultural values actively reshape latent geometry. *Color-coded arrows* trace distinct conceptual trajectories (protest, peace, order, power, disobedience, justice), while numeric labels quantify local steering strength. Upper layers $$(\ell \geq 20)$$ typically exhibit **epistemic reorientation**, where cultural priors most heavily influence belief encoding. Such visualizations reveal whether a model internalizes culturally contingent reasoning or merely mimics alignment at the output surface.
{% endcapture %}

{% include visualization-html.liquid 
   interactive_html="ndna/belief_vector_field_left_only.html"
   image_path="ndna/belief_vector_field.gif"
   alt="Belief Vector Field Visualization"
   caption=belief_vector_caption
   full_width=true %}

<!-- {% include visualization-html.liquid 
   interactive_html="ndna/belief_vector_field_3d_all_layers_thick_long_vectors.html" %} -->

## nDNA: Unified Epistemic Inheritance Measure

**Why a unified score?**
While **spectral curvature** $$(\kappa_\ell)$$, **thermodynamic length** $$(\mathcal{L}_\ell)$$, and the **belief vector field norm** $$(\lVert\mathbf{v}_\ell^{(c)}\rVert)$$ each offer unique insight into latent dynamics, they operate on distinct facets of *epistemic geometry*:

> **The nDNA score is a cumulative measure of latent geometry, quantifying how a large language model adapts its internal scaffolding to a given corpus.** It integrates three key components at each layer $$\ell$$:
> 
> - **Curvature** $$(\kappa_\ell)$$: how *twisted* or *bent* the latent manifold is; captures *how sharply* internal trajectories bend -- a scalar measure of latent acceleration.
> - **Length** $$(\mathcal{L}_\ell)$$: how much *latent work* or displacement occurs as representations evolve; quantifies *how hard* the model works to adapt its beliefs -- a scalar effort integral.
> - **Belief vector norm** $$(\lVert\mathbf{v}_\ell^{(c)}\rVert)$$: how *strong* the model's belief signal is for that corpus; encodes *where* and *how strongly* cultural priors steer latent space -- a scalar magnitude derived from the vector field.
> 
> **Formally, we define the nDNA score as:**
> 
> $$\text{nDNA} := \sum_{\ell=1}^L \omega_\ell \cdot \kappa_\ell \cdot \mathcal{L}_\ell \cdot \lVert\mathbf{v}_\ell^{(c)}\rVert$$

Individually, these measures illuminate latent strain, adaptation cost, and cultural pressure. But to assess *inheritance as a whole* -- how traits propagate through **fine-tuning**, **merging**, or **distillation** -- we must integrate these into a single diagnostic that reflects combined latent geometry, epistemic work, and directional influence.

**Designing the composite measure.**
Since $$\kappa_\ell$$ and $$\mathcal{L}_\ell$$ are scalars, and $$\lVert\mathbf{v}_\ell^{(c)}\rVert$$ reduces directional drift to scalar magnitude, their product forms a natural joint measure of: *internal bending* $$(\kappa_\ell)$$, *internal epistemic effort* $$(\mathcal{L}_\ell)$$, and *external drift pressure* $$(\lVert\mathbf{v}_\ell^{(c)}\rVert)$$. To balance their contributions across depth, we introduce layer weights $$\omega_\ell$$, emphasizing semantically active or epistemically significant layers (e.g., $$\omega_\ell$$ higher in upper decoder blocks).

{% capture ndna_story_caption %}
**The compositional anatomy of neural DNA (nDNA) through curvature, length, and belief geometry.** This figure illustrates how nDNA arises as a layered product of three latent quantities. First, **spectral curvature** $$\kappa_\ell$$ measures latent manifold bending and flexibility (latent acceleration), indicating how sharply the internal geometry twists at layer $$\ell$$. Second, **thermodynamic length** $$\mathcal{L}_\ell$$ quantifies the accumulated epistemic effort (latent adaptation energy) and reflects how hard the model works to reconcile prior beliefs with new input and alignment signals. Third, **belief vector norm** $$\lVert\mathbf{v}_\ell^{(c)}\rVert$$ encodes the magnitude of latent directional force imposed by corpus priors or alignment signals. The joint trajectory in $$(\kappa_\ell, \mathcal{L}_\ell, \lVert\mathbf{v}_\ell^{(c)}\rVert)$$ space, color-coded by the composite score, shows how bending, effort, and steering co-evolve across layers. The combined latent signature is formalized as $$\text{nDNA}_\ell = \kappa_\ell \cdot \mathcal{L}_\ell \cdot \lVert\mathbf{v}_\ell^{(c)}\rVert = 0.0024$$ (example layer), with high values identifying zones of intense latent reconfiguration where geometry and adaptation forces align. Color-keyed descriptors ("Latent bending", "Epistemic effort", "Belief steering") guide visual interpretation. The figure illustrates how large language models coordinate latent bending, effort, and steering to build a neurogeometric scaffold that adapts flexibly to task complexity while remaining anchored in a universal latent structure.
{% endcapture %}

{% include visualization-html.liquid 
   image_path="ndna/ndna_refined_story.gif"
   alt="nDNA Compositional Anatomy"
   caption=ndna_story_caption %}

This composite score integrates scalar and vector-derived diagnostics into a unified measure of *epistemic inheritance* -- quantifying the latent structure and cultural traits a model carries forward from its neural ancestry.

**Rationale for multiplicative integration.**
This form spotlights layers where latent paths bend sharply, belief adaptation incurs significant effort, and cultural or alignment pressures apply strong directional force. High scores identify zones of *intense latent reconfiguration*, where internal dynamics and external pressures converge to reshape the model's reasoning space.

**Role of $$\omega_\ell$$.**
The weight $$\omega_\ell$$ serves as a lens to prioritize semantically expressive, epistemically active regions of the network. It may be set uniformly, hand-tuned, or optimized against alignment drift benchmarks, bias metrics, or interpretability objectives.

**Interpretability and utility.**
The nDNA score provides a compact fingerprint of model inheritance:
- It enables direct comparison of parent and child models post **fine-tuning**, **merging**, or **distillation**.
- It highlights zones of *semantic mutation*, *ideological absorption*, or *cultural drift*.
- It serves as a proxy for *latent epistemic integrity* -- quantifying the hidden cost and directionality of neural evolution.

By unifying **spectral**, **thermodynamic**, and **vectorial** diagnostics, the nDNA score functions as a **heritable geometry index** -- diagnosing how latent traits persist, mutate, or degrade as foundation models evolve.

### nDNA Geometry: A Closer Look

The notion of **nDNA** arises from a simple yet profound insight: modern foundation models do not merely produce outputs--they embody a latent cognitive structure that governs how they reason, adapt, and evolve {% cite bommasani2023foundation ganguli2023reducing %}. This latent structure is not directly encoded in model weights or activations alone; rather, it emerges in the internal geometry of belief formation, semantic flow, and epistemic adaptation across layers {% cite liu2023hidden wang2021geomtransformer %}. We define the **nDNA geometry** of a model as the joint distribution of its **spectral curvature** ($$\kappa_\ell$$), **thermodynamic length** ($$\mathcal{L}_\ell$$), and **belief vector field norm** ($$\lVert\mathbf{v}_\ell^{(c)}\rVert$$) layer-by-layer. This triad forms a high-dimensional semantic fingerprint that encodes a model's *inheritance stability*, *alignment dynamics*, and *cultural drift*---analogous to how biological DNA records heritable traits and mutations {% cite shen2023beliefgeometry bakker2024uniting %}.

{% capture ndna_example_caption %}
An **illustrative nDNA example** that captures the *semantic genome* of a foundation model through the joint interplay of **spectral curvature** ($$\kappa_\ell$$), **thermodynamic length** ($$\mathcal{L}_\ell$$), **belief vector norm** ($$\lVert\mathbf{v}_\ell^{(c)}\rVert$$) across layers. Each of these quantities offers a distinct geometric and epistemic lens: $$\kappa_\ell$$ measures the *local acceleration* of latent representations, $$\mathcal{L}_\ell$$ quantifies the cumulative *internal work* required to traverse the belief manifold, while $$\lVert\mathbf{v}_\ell^{(c)}\rVert$$ encodes the *magnitude of cultural drift* imposed on latent activations. The *color intensities* shown alongside each value reflect relative magnitude within column-specific ranges: <span style="background: #00AA00; color: white; padding: 2px 4px; border-radius: 3px;">low</span>, <span style="background: #CCCC00; color: black; padding: 2px 4px; border-radius: 3px;">moderate</span>, <span style="background: #FF8000; color: white; padding: 2px 4px; border-radius: 3px;">high</span>, <span style="background: #CC0000; color: white; padding: 2px 4px; border-radius: 3px;">very high</span>. For this example, spectral curvature spans $$\kappa_\ell \in [0.0400, 0.0700]$$, thermodynamic length $$\mathcal{L}_\ell \in [0.80, 1.20]$$, and belief vector norm $$\lVert\mathbf{v}_\ell^{(c)}\rVert \in [0.55, 0.75]$$--revealing regions where the *latent manifold bends*, *epistemic energy intensifies*, or *external priors steer internal cognition*. This triad forms what we term the model's **nDNA**: a compact, high-dimensional *semantic fingerprint* that encodes the hidden geometry of belief. It enables us to diagnose zones of *inheritance stability*, detect *ideological absorption*, and trace *latent mutations* introduced by fine-tuning, alignment, or architectural choice. The pattern of these quantities across layers constitutes a signature as unique as a biological genome -- a map of how artificial cognition evolves, remembers, and adapts.
{% endcapture %}

<div class="table-container" style="overflow-x: auto; margin: 20px 0;">
<table style="width: 100%; border-collapse: collapse; font-family: 'Courier New', monospace; margin: 20px 0;">
<caption style="font-weight: bold; margin-bottom: 10px; font-size: 1.1em; line-height: 1.6;">
{{ ndna_example_caption | markdownify }}
</caption>
<thead>
<tr style="background-color: #f8f9fa; border-bottom: 2px solid #333;">
<th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">Layer</th>
<th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">$$\kappa_\ell$$</th>
<th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">$$\mathcal{L}_\ell$$</th>
<th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">$$\lVert\mathbf{v}_\ell^{(c)}\rVert$$</th>
<th style="border: 1px solid #333; padding: 10px; text-align: center; font-weight: bold;">Belief Vector $$\mathbf{v}_\ell^{(c)}$$</th>
</tr>
</thead>
<tbody>
<tr><td style="border: 1px solid #333; padding: 8px; text-align: center; font-weight: bold;">20</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(0, 170, 0, 0.2);">0.0412</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 204, 0, 0.3);">0.9123</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(255, 128, 0, 0.2);">0.6521</td><td style="border: 1px solid #333; padding: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;">[0.1204, -0.0502, 0.0896, …, 0.0402]</td></tr>

<tr><td style="border: 1px solid #333; padding: 8px; text-align: center; font-weight: bold;">21</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(0, 170, 0, 0.3);">0.0458</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(0, 170, 0, 0.15);">0.8123</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 0, 0, 0.3);">0.7523</td><td style="border: 1px solid #333; padding: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;">[0.1301, -0.0351, 0.0950, …, 0.0431]</td></tr>

<tr><td style="border: 1px solid #333; padding: 8px; text-align: center; font-weight: bold;">22</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 204, 0, 0.2);">0.0523</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(255, 128, 0, 0.3);">1.0120</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(0, 170, 0, 0.2);">0.5823</td><td style="border: 1px solid #333; padding: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;">[0.1423, -0.0312, 0.0994, …, 0.0488]</td></tr>

<tr><td style="border: 1px solid #333; padding: 8px; text-align: center; font-weight: bold;">23</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(255, 128, 0, 0.2);">0.0581</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 204, 0, 0.2);">0.9021</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 204, 0, 0.3);">0.6912</td><td style="border: 1px solid #333; padding: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;">[0.1534, 0.0270, 1.0042, …, 0.0512]</td></tr>

<tr><td style="border: 1px solid #333; padding: 8px; text-align: center; font-weight: bold;">24</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 0, 0, 0.2);">0.0639</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 0, 0, 0.4);">1.1023</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(0, 170, 0, 0.15);">0.5520</td><td style="border: 1px solid #333; padding: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;">[0.1667, 0.0205, 1.1105, …, 0.0543]</td></tr>

<tr><td style="border: 1px solid #333; padding: 8px; text-align: center; font-weight: bold;">25</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 204, 0, 0.3);">0.0505</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(255, 128, 0, 0.2);">0.9420</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 0, 0, 0.4);">0.8124</td><td style="border: 1px solid #333; padding: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;">[0.1602, -0.0251, 1.0081, …, 0.0504]</td></tr>

<tr><td style="border: 1px solid #333; padding: 8px; text-align: center; font-weight: bold;">26</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(0, 170, 0, 0.15);">0.0398</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(0, 170, 0, 0.2);">0.8520</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 204, 0, 0.2);">0.6120</td><td style="border: 1px solid #333; padding: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;">[0.1251, 0.0450, 0.0912, …, 0.0418]</td></tr>

<tr><td style="border: 1px solid #333; padding: 8px; text-align: center; font-weight: bold;">27</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 204, 0, 0.2);">0.0512</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 0, 0, 0.3);">1.0520</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(255, 128, 0, 0.3);">0.7222</td><td style="border: 1px solid #333; padding: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;">[0.1455, -0.0322, 1.0005, …, 0.0477]</td></tr>

<tr><td style="border: 1px solid #333; padding: 8px; text-align: center; font-weight: bold;">28</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(255, 128, 0, 0.3);">0.0590</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 204, 0, 0.3);">0.9320</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(0, 170, 0, 0.3);">0.5721</td><td style="border: 1px solid #333; padding: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;">[0.1577, 0.0285, 1.0078, …, 0.0499]</td></tr>

<tr><td style="border: 1px solid #333; padding: 8px; text-align: center; font-weight: bold;">29</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 0, 0, 0.3);">0.0672</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(255, 128, 0, 0.3);">1.0123</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 204, 0, 0.2);">0.6322</td><td style="border: 1px solid #333; padding: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;">[0.1701, -0.0198, 1.1142, …, 0.0533]</td></tr>

<tr><td style="border: 1px solid #333; padding: 8px; text-align: center; font-weight: bold;">30</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(255, 128, 0, 0.2);">0.0555</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(0, 170, 0, 0.15);">0.8221</td><td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: rgba(204, 0, 0, 0.2);">0.7720</td><td style="border: 1px solid #333; padding: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;">[0.1620, -0.0242, 1.1101, …, 0.0510]</td></tr>
</tbody>
</table>
</div>

This table provides an *illustrative example of nDNA geometry*, highlighting how these quantities vary across depth in a representative model. Rather than simple monotonic trends, we observe intricate layer-wise patterns: certain layers exhibit elevated curvature ($$\kappa_\ell > 0.06$$), signaling sharp latent reorientation {% cite cho2023mixedcurvature %}, while others concentrate thermodynamic length ($$\mathcal{L}_\ell > 1.10$$), reflecting zones of intense internal work to reconcile competing priors {% cite crooks2007measuring oliviero2023thermodynamics %}. The belief vector norm $$\lVert\mathbf{v}_\ell^{(c)}\rVert$$ exposes the directional cultural force acting on the latent manifold {% cite peng2024cultural zhou2023alignmentdrift %}, marking layers where external alignment or sociolinguistic conditioning exerts greatest influence. Together, these values form a geometry-specific trace that distinguishes models by their latent adaptation history.

## The Corpus Dependence of nDNA: A Necessary Feature, Not a Flaw

In biological systems, DNA is celebrated as the *universal code of life* -- a sequence of nucleotides that, across all known organisms, governs the development, function, and inheritance of traits {% cite alberts2014molecular lewin2013genes %}. Yet despite this **universal structure**, the functional expression of DNA is profoundly **context-dependent**. The same genome, when expressed in different cellular contexts, gives rise to vastly different phenotypes: for instance, *neurons* and *hepatocytes* arise from identical genetic material yet serve radically different functions {% cite bird2007perceptions davidson2006gene %}. This context-sensitive expression is orchestrated through layered regulatory mechanisms, including **epigenetic modifications** {% cite bird2007perceptions %}, **transcription factor (TF) binding** {% cite lambert2018human %}, and **chromatin architecture remodeling** {% cite clapier2017mechanisms dekker2013exploring %}. These mechanisms form a hierarchical, probabilistic regulatory network that determines gene expression patterns in response to developmental and environmental cues {% cite alon2006introduction %}.

{% capture dna_ndna_caption %}
**A hierarchical view of universal DNA and context-sensitive gene expression, as a biological parallel to nDNA latent scaffolding in LLMs.** This figure illustrates how the *same genome* (depicted as a universal DNA helix at the top) produces distinct functional outcomes through a layered and structured regulatory architecture. The **first regulatory layer** consists of *epigenetic modifications*, including DNA methylation (linked with gene silencing) and histone acetylation (linked with gene activation) {% cite bird2007perceptions clapier2017mechanisms %}. These modifications influence chromatin accessibility, setting the stage for context-specific transcriptional control. The **second layer** involves *cell-type-specific transcription factors (TFs)* -- for example, NeuroD and REST in neurons, or HNF4 and C/EBPα in hepatocytes -- which bind regulatory DNA elements and integrate signaling cues to guide gene expression programs {% cite lambert2018human davidson2006gene %}. The **third layer** reflects the resultant chromatin state: open, transcriptionally permissive configurations in neurons for synaptic gene activation, versus compact, repressive configurations in hepatocytes where those genes are silent {% cite thurman2012accessible dekker2013exploring %}. Finally, this hierarchical regulatory control produces *functionally specialized gene programs*: neurons activate synaptic plasticity and axon signaling genes; hepatocytes activate detoxification and glucose metabolism genes {% cite lewin2013genes alon2006introduction %}. This layered architecture provides a powerful biological analogy for *nDNA in LLMs*. Just as DNA's expression is shaped by regulatory logic rather than random variation, **nDNA encodes both universal priors (shared across tasks)** -- such as pretrained latent manifolds, attention mechanisms, and model architecture -- and **corpus-dependent latent scaffolding**, emerging as the model adapts to specific tasks or domains {% cite olah2020zoom geva2021transformer beltagy2020longformer %}. The analogy emphasizes that corpus dependence in nDNA is not a weakness or artifact, but a reflection of meaningful task adaptation: *structured variation grounded in universal latent geometry*. This scaffolding ensures LLMs achieve *functional diversity* across tasks while maintaining **coherence, alignment, and generalization**, much like gene regulatory networks ensure appropriate cellular identity and function despite operating from a common genome blueprint {% cite alon2006introduction davidson2006gene %}. The figure highlights that both biological DNA and nDNA exhibit clarity through complexity: **layered, interpretable hierarchies enabling flexible, robust expression across contexts**.
{% endcapture %}

{% include visualization-html.liquid 
   image_path="ndna/task_nDNA/universal_dna.png"
   alt="DNA-nDNA Hierarchical Comparison"
   caption=dna_ndna_caption %}

Similarly, in large foundation models, the *neural DNA (nDNA)* -- a composite measure of latent geometry encompassing **spectral curvature** $$(\kappa)$$ {% cite belkin2019reconciling %}, **thermodynamic length** $$(L)$$ {% cite still2012thermodynamic %}, and **latent belief vector norms** {% cite olah2020zoom %} -- exhibits both **universal structure** and **corpus-specific adaptation**. LLMs encode universal latent priors through pretraining: architectural invariances {% cite vaswani2017attention %}, semantic manifolds {% cite mikolov2013distributed bommasani2021opportunities %}, and attention-based relational structures {% cite geva2021transformer %}. However, when probed with different corpora -- such as mathematical reasoning benchmarks (e.g. GSM8K {% cite cobbe2021training %}), dialogue datasets (e.g. MultiWOZ {% cite budzianowski2018multiwoz %}), or encyclopedic QA (e.g. SQuAD {% cite rajpurkar2016squad %}) -- the model activates distinct latent scaffolding, producing task-specific geometric pathways.

In both systems, **structured variation emerges as a necessity**: in **biology**, to produce *functional diversity* across cell types; in **LLMs**, to scaffold *reasoning* across tasks while maintaining **alignment** and **generalization** {% cite bommasani2021opportunities cobbe2021training %}. Like **tissue-specific gene expression**, **corpus-dependent nDNA scaffolding** follows precise, *learned priors* rather than arbitrary variation. **Mathematical models** of both systems reduce to *path integrals over conditional cost*:

$$\mathcal{S}(c) = \int_{\gamma_c} \mathcal{C}(h_\ell; c) \, ds$$

where $$\gamma_c$$ is the pathway for *context* $$c$$ (cell type or corpus), and $$\mathcal{C}$$ reflects *regulatory* or *loss cost*.

> ***Where DNA differentiates cells**, nDNA differentiates reasoning. Both systems achieve **functional coherence** through context-dependent geometry anchored in **universal code**.*

Despite their *contextual variation*, both **DNA** and **nDNA** encode **universal structure** that stabilizes functional diversity. In **biology**, this universality is embodied in the *genetic code*: the shared language of **codons**, **conserved regulatory motifs**, and **chromatin architectural principles** that ensure coherent development across tissues {% cite lewin2013genes alberts2014molecular %}. In **large language models**, nDNA's universality arises from the **shared latent priors** learned during pretraining: **attention-based relational structures** {% cite vaswani2017attention %}, **semantic manifolds** {% cite mikolov2013distributed %}, and **transformer-invariant latent symmetries** {% cite bommasani2021opportunities %}. These priors act as the *"genomic grammar"* that binds task-specific latent pathways into a **coherent reasoning framework**.

$$\boxed{\textbf{DNA: }\Sigma^3 / \ker \phi \to \mathcal{A}\quad\textbf{nDNA: }\mathcal{X} / G_{\text{LLM}} \to V/G}$$

Such **universal structure** enables **generalization**: in **biology**, reliable *organismal development*; in **LLMs**, reasoning *consistency* and *alignment* across tasks. Crucially, this structure constrains **corpus-dependent variation** within *interpretable latent geometry* -- preventing arbitrary or adversarial drift {% cite geva2021transformer cobbe2021training %}.

> ***What DNA is to the unity of multicellular life**, nDNA is to the coherence of LLM reasoning: a **stabilizing universal code** that enables **structured functional variation**.*

### Evolutionary and Learning Dynamics: Convergence of Principles

Both **DNA** and **nDNA** are shaped by *selection processes*. In **biology**, the genome has evolved under millennia of selective pressure, with **regulatory networks** fine-tuned to ensure *robust development* and *adaptability* {% cite alon2006introduction davidson2006gene %}. In **LLMs**, pretraining operates as an *evolutionary analogue*: **stochastic gradient descent (SGD)** over massive corpora selects latent priors that minimize expected loss across tasks, with *fine-tuning akin to epigenetic adjustment* {% cite bommasani2021opportunities pfeiffer2021adapterfusion %}.

$$\underbrace{\mathcal{L}_{\text{pretrain}}(\theta) = \mathbb{E}_{(x,y)} [-\log p_\theta(y|x)]}_{\textbf{SGD as selection pressure}}$$

This **evolutionary parallel** explains why both systems exhibit *clarity through complexity*: **layered hierarchies**, **probabilistic pathways**, and **interpretable modularity**. Where **biological evolution** yields *modular gene regulatory networks* that ensure context-sensitive expression {% cite alon2006introduction %}, **LLM training** yields *modular latent structures* -- such as **attention heads** and **adapter modules** -- that scaffold *task-specific reasoning* {% cite geva2021transformer pfeiffer2021adapterfusion %}.

### Why Corpus Dependence Matters

Far from a flaw, **corpus dependence in nDNA** is the signature of a *flexible*, *adaptive reasoning architecture*. Just as biological systems rely on **tissue-specific gene expression** to produce functional diversity from a *universal genome* {% cite davidson2006gene alon2006introduction %}, large language models (LLMs) leverage **corpus-dependent latent scaffolding** to generate reasoning structures attuned to task demands, mirroring the reproducibility logic of biological variability quantification {% cite marioni2011rna %}. By examining nDNA's **spectral curvature** $$(\kappa)$$, **thermodynamic length** $$(\mathcal{L})$$, and **belief vector norm** $$(\lVert\mathbf{v}_\ell^{(c)}\rVert)$$, we gain a **diagnostic lens** for alignment, generalization, and safety {% cite belkin2019reconciling still2012thermodynamic olah2020zoom %}:

$$\mathcal{S}_{\text{nDNA}}(c) = \int_{\gamma_c} \left(\alpha\kappa + \beta\mathcal{L} + \gamma\lVert\mathbf{v}_\ell^{(c)}\rVert\right) \, ds$$

where $$\gamma_c$$ is the latent trajectory for corpus $$c$$. This latent geometry echoes Waddington's epigenetic landscape where paths represent developmental fates {% cite waddington1957strategy %}.

**QA tasks** evoke compact low-curvature paths (e.g. $$\kappa \sim 0.012$$–$$0.03$$, $$\mathcal{L} \sim 0.47$$–$$0.53$$) {% cite rajpurkar2016squad kwiatkowski2019natural joshi2017triviaqa %}, while **reasoning tasks** elicit broader high-curvature paths (e.g. $$\kappa \sim 0.005$$–$$0.04$$) {% cite cobbe2021training patel2021nlp geva2021transformer %}. **Dialogue corpora** produce shallow clustered scaffolds {% cite budzianowski2018multiwoz li2016persona zhang2018personalizing %}; **commonsense tasks** yield oscillatory paths {% cite sap2019socialiqa zellers2019hellaswag talmor2019commonsenseqa %}. nDNA aligns with interpretable AI goals {% cite zhang2018interpretable %} and geometric decoding approaches {% cite narayanan2021decoding %}.

This corpus dependence is *not arbitrary noise* -- it reflects the model's **learned latent regulatory logic**, analogous to the combinatorial control of **gene regulatory networks** that ensures *context-sensitive yet robust gene expression* {% cite alon2006introduction lewin2013genes %}. Just as *developmental disorders* arise when regulatory circuits misfire {% cite davidson2006gene %}, misalignment or hallucination in LLMs can be traced to *latent trajectories that diverge from expected scaffolding*. **nDNA analysis**, therefore, does not merely characterize model geometry -- it offers a **tool for interpretability, failure detection, and safe alignment**.

> ***Corpus dependence in nDNA is the expression of reasoning plasticity**, bounded by universal latent priors much like gene networks balance flexibility with functional coherence.*

Moreover, the **universality of nDNA's foundational structure** -- its pretrained manifold, architectural symmetries, and core alignment priors -- provides the *stabilizing grammar* that constrains corpus-specific scaffolds within meaningful reasoning spaces {% cite vaswani2017attention bommasani2021opportunities %}. This is the latent equivalent of biology's **genetic code** and **conserved transcriptional machinery**: an *invariant substrate* that supports functional diversity without sacrificing coherence. By quantifying how nDNA paths *bend*, *stretch*, or *steer* in response to task demands, we can map the model's **cognitive landscape** -- and determine when it traces *human-aligned reasoning* or drifts into failure modes.

> ***What the genome is to life's functional unity**, nDNA is to the model's reasoning coherence: a universal code that binds diversity into stability, and complexity into interpretability.*

| **Layer** | **DNA (Biology)** | **nDNA (LLM)** |
|-----------|-------------------|-----------------|
| **Universal code** | Codon mapping $$\phi: \Sigma^3 \to \mathcal{A}$$, kernel $$\neq \emptyset$$, redundancy ensures error tolerance {% cite lewin2013genes %} | Pretrained latent manifold; symmetries $$G_{\text{LLM}} \subset \text{Aut}(V)$$; generalization via equivariance {% cite bommasani2021opportunities %} |
| **Context regulator** | Conditional $$P(\text{gene ON}\|\text{TF, epi})$$; Bayesian gene networks {% cite alon2006introduction %} | Conditional latent path $$P(h_1,\ldots,h_\ell\|x)$$; stochastic latent dynamics {% cite geva2021transformer %} |
| **Path geometry** | Minimal energy path $$\gamma^*$$ in epigenetic landscape: $$\int_\gamma \lVert\nabla V\rVert \, ds$$ {% cite waddington1957strategy %} | Latent geodesic minimizing cost: $$\int_\gamma \lVert\nabla_\theta \log p(y\mid x)\rVert^2 \, ds$$ {% cite still2012thermodynamic %} |
| **Output mapping** | Fiber bundle: $$\pi: E_{\text{gene}} \to B_{\text{cell}}$$ | Fiber bundle: $$\pi: E_{\text{latent}} \to B_{\text{task}}$$ |

### nDNA Task Group Analysis

{% include visualization-html.liquid 
   image_path="ndna/task_nDNA/ndna_llama_qa_axes_match.gif"
   interactive_html="ndna/task_nDNA/ndna_llama_qa_interactive.html"
   alt="QA Group nDNA Trajectories"
   caption="**QA group nDNA trajectories**: $$\kappa$$ ranges $$\sim 0.012$$–$$0.03$$, $$\mathcal{L} \sim 0.47$$–$$0.53$$, $$\tau \sim 0.006$$–$$0.014$$. Trajectories are compact and consistently shaped across datasets, reflecting **shared task structure**."
   %}

{% include visualization-html.liquid 
   image_path="ndna/task_nDNA/ndna_llama_dialogue_axes_match.gif"
   interactive_html="ndna/task_nDNA/ndna_llama_dialogue_interactive.html"
   alt="Dialogue Group nDNA Trajectories"
   caption="**Dialogue group nDNA trajectories**: $$\kappa$$ ranges $$\sim 0.01$$–$$0.03$$, $$\mathcal{L} \sim 0.47$$–$$0.53$$, $$\tau \sim 0.006$$–$$0.014$$. Trajectories are shallow and tightly clustered, reflecting **low latent complexity** typical of conversational flow."
   %}
{% include visualization-html.liquid 
   image_path="ndna/task_nDNA/ndna_llama_reasoning_axes_match.gif"
   interactive_html="ndna/task_nDNA/ndna_llama_reasoning_interactive.html"
   alt="Reasoning Group nDNA Trajectories"
   caption="**Reasoning group nDNA trajectories**: $$\kappa$$ ranges $$\sim 0.005$$–$$0.04$$, $$\mathcal{L} \sim 0.44$$–$$0.56$$, $$\tau \sim 0.002$$–$$0.018$$. Trajectories show **greater spread and complexity**, reflecting multi-step reasoning scaffolding."
   %}

{% include visualization-html.liquid 
   image_path="ndna/task_nDNA/ndna_llama_commonsense_axes_match.gif"
   interactive_html="ndna/task_nDNA/ndna_llama_commonsense_interactive.html"
   alt="Commonsense Group nDNA Trajectories"
   caption="**Commonsense group nDNA trajectories**: $$\kappa$$ ranges $$\sim 0.00$$–$$0.04$$, $$\mathcal{L} \sim 0.44$$–$$0.54$$, $$\tau \sim 0.004$$–$$0.018$$. Trajectories are intermediate in complexity, reflecting **varied latent demands of commonsense reasoning**."
   %}

{% capture ndna_groups_caption %}
**nDNA trajectories across LLaMA vs. task groups.** Each subplot visualizes **spectral curvature** $$(\kappa_\ell)$$, **thermodynamic length** $$(\mathcal{L}_\ell)$$, and **belief vector norm** $$(\lVert\mathbf{v}_\ell^{(c)}\rVert)$$ layer-wise trajectories for representative datasets. The structured variation illustrates that *corpus dependence in nDNA is meaningful and interpretable*, reflecting task complexity rather than random noise. **QA** and **dialogue** tasks activate **compact, smooth latent scaffolds with low curvature and modest belief steering**; **reasoning** tasks exhibit broader, more intricate geometry, with **increasing curvature, longer latent length, and stronger belief vector dynamics**. **Commonsense** tasks show intermediate complexity with **oscillatory scaffolding, reflecting ambiguity and contextual switching**. This figure demonstrates the core takeaway of our section: *like biological DNA, nDNA expresses differently in context, but remains bound by universal latent priors that ensure coherence, generalization, and alignment.*
{% endcapture %}

{% include visualization-html.liquid 
   image_path="ndna/task_nDNA/ndna_llama_summarization_axes_match.gif"
   interactive_html="ndna/task_nDNA/ndna_llama_summarization_interactive.html"
   alt="Summarization Group nDNA Trajectories"
   caption=ndna_groups_caption %}

---

{% auto_references %}
