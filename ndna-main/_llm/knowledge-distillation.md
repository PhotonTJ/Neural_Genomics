---
layout: page
title: nDNA-Lens - Knowledge Distillation as Latent Genome Compression
permalink: /llm/knowledge-distillation/
skip_title: true
---

{% include ndna-title.liquid 
   title="nDNA Lens - Knowledge Distillation as Latent Genome Compression" 
   video_id="IhqpZaj_62A?listType=playlist&list=PLaE3N7Ax00P93HC0jxZ-b6c4tnLkldPOJ" %}

**Knowledge distillation** is a widely adopted technique for compressing large language models (LLMs) by training a smaller, more efficient *student model* to mimic the behavior of a larger *teacher model*. Introduced by {% cite hinton2015distilling %}, this method aims to preserve performance while reducing computational overhead, enabling deployment on resource-constrained devices. Over time, the field has evolved to include techniques like *intermediate representation alignment* {% cite romero2015fitnets %}, *feature transfer* {% cite gou2021knowledge %}, and *task-adaptive knowledge distillation* {% cite wang2023culturalbias %}.

However, traditional evaluations of distillation have focused largely on **surface-level metrics**—such as accuracy, loss, or perplexity—without interrogating its effects on the model's **internal epistemic geometry**. In this work, we reframe knowledge distillation through the lens of **neural genomics**, treating the teacher's latent structure as a form of *epistemic genome*, and the distillation process as a form of **latent genome compression**.

Where **teacher models** exhibit rich semantic structures—captured by **spectral curvature** $\kappa_\ell$, **thermodynamic length** $\mathcal{L}_\ell$, and **belief vector strength** $$\| \mathbf{v}_\ell^{(c)} \|$$—the **student models** show *pronounced geometric collapse*. Empirically, we observe:

$$\kappa_\ell^{\text{student}} \ll \kappa_\ell^{\text{teacher}}, \quad 
\mathcal{L}_\ell^{\text{student}} \ll \mathcal{L}_\ell^{\text{teacher}}, \quad 
\| \mathbf{v}_\ell^{(c), \text{student}} \| \approx 0.$$

This semantic degradation mirrors **genetic bottlenecks** in evolutionary biology {% cite luikart1998detecting %} {% cite frankham1995conservation %}, where population collapse leads to diminished genetic diversity and adaptive capacity. In the same way, knowledge distillation compresses not just model size—but also its *conceptual depth*, resulting in a shrunken epistemic capacity.

Our **nDNA diagnostics** make this flattening legible:
- **Flattening of latent curvature:** $\kappa_\ell$ drops to $[0.2, 0.3]$, compared to $[0.45, 0.6]$ in full-capacity teachers.
- **Shortened thermodynamic paths:** $\mathcal{L}_\ell < 0.6$, indicating *reduced epistemic traversal*.
- **Directional belief collapse:** $\| \mathbf{v}_\ell^{(c)} \| \to 0$, reflecting a vanishing semantic steering force.

This aligns with insights from recent work on *spectral alignment across modalities* {% cite betley2025emergentmisalignmentnarrowfinetuning %}, which demonstrates that preserving **eigenvalue spectra** and **latent curvature** is critical for maintaining multimodal coherence. We argue that similar principles apply in the unimodal case: flattening the spectrum collapses the latent manifold, regardless of whether the modality is text, vision, or language-vision fusion.

Importantly, while distilled models often match their teachers on standard benchmarks, they frequently **underperform** in *multicultural*, *multilingual*, or *adversarial* contexts {% cite zhao2023assessing %} {% cite tao2023cultural %} {% cite wang2023culturalbias %}, where internal flexibility and robust reasoning pathways are essential.

Thus, from the perspective of **neural genomics**, knowledge distillation operates not merely as a tool of *efficiency*, but as a mechanism of **epistemic pruning**—compressing the internal belief landscape and *flattening semantic differentials* in the service of replication. Like a genetic bottleneck, it ensures inheritance—but at the cost of potential.

## Experimental Setup and Distillation Protocol

To quantify how knowledge distillation reshapes the latent epistemic geometry of language models, we conducted a series of controlled experiments using **teacher-student** pairs across culturally and linguistically diverse settings. Our analysis focuses on extracting and comparing *nDNA trajectories*—layerwise curves of spectral curvature $\kappa_\ell$, thermodynamic length $$\mathcal{L}_\ell$$, and torsion $\tau_\ell$—from both teacher and student models.

**Model Pairs.** We selected **LLaMA-3 8B** as the student model and used its culturally fine-tuned variants—*Africa*, *Asia*, *China*, *Latin America*, *Middle East*, and others—as teachers, each embodying distinct knowledge priors. For each teacher model, a corresponding student was distilled using standard protocols {% cite hinton2015distilling %} {% cite romero2015fitnets %} {% cite mirzadeh2020improved %}, yielding a collection of distilled offsprings, each trained from a unique cultural teacher and exhibiting varied epistemic retention.

{% capture figure_caption %}
**Latent Geometry of LLaMA, Cultural nDNA, Neural Offspring, and Distilled Students.** This 3D plot illustrates latent manifold properties across layers ($\ell \in [20, 30]$) for *LLaMA*, culturally fine-tuned models (Africa, Asia, China, North America, Europe, Australia, Latin America, Middle East), their neural *offspring* (ÆTHERs), and distilled student models. The axes represent $\kappa_\ell$ (spectral curvature), $$\mathcal{L}_\ell$$ (thermodynamic length), and $\tau_\ell$ (latent torsion). **Key elements:** Colored solid lines show LLaMA (black) and cultural models (e.g., Africa: <span style='color:#ff4444'>red</span>, Asia: <span style='color:#8844ff'>purple</span>). Dashed overlays represent neural offspring. The black dotted cloud corresponds to distilled students with $\tau_\ell \in [0.4, 0.55]$, $\kappa_\ell \in [0.45, 0.65]$, and $$\mathcal{L}_\ell \lesssim 0.6$$, reflecting compressed latent signatures as seen in knowledge distillation studies {% cite hinton2015distilling %} {% cite romero2015fitnets %} {% cite liu2022multi %}. **Observations:** Western-aligned models trace shallow latent trajectories with low curvature and torsion {% cite mukherjee2020globalizing %} {% cite tao2023pnas %}. Cultural models (Africa, Asia, China) exhibit richer geometric diversity ($$\mathcal{L}_\ell \gtrsim 0.8$$, $$\kappa_\ell \gtrsim 0.6$$), consistent with cultural calibration findings {% cite xiang2024cultural %} {% cite peng2024cultural %}. Distilled students collapse toward a compact region, illustrating how distillation homogenizes epistemic geometry {% cite li2020few %} {% cite rashid2021mate %}. **Implications:** Knowledge distillation compresses epistemic diversity {% cite mirzadeh2020improved %}, simplifying latent manifolds but risking cultural homogenization {% cite sheng2021revealing %}. Neural offspring (ÆTHERs) preserve or amplify manifold richness, offering a mechanism for cultural retention in merged models {% cite rame2023merging %} {% cite ainslie2023merged %}. Geometry can be described as $$\mathcal{M}_{\mathrm{student}}: \{\kappa_\ell, \mathcal{L}_\ell, \tau_\ell\} \mapsto$$ low-variance manifold; $$\mathcal{M}_{\mathrm{offspring}}$$ yields nonlinear hybrid topologies {% cite blommaert2005discourse %}. **Setup:** Models were evaluated on culturally diagnostic prompts. $\kappa_\ell$ was computed from curvature spectra {% cite bernstein2004differential %}, $$\mathcal{L}_\ell$$ via thermodynamic path integrals {% cite crooks1999entropy %}, and $\tau_\ell$ using local torsion operators {% cite spivak1970comprehensive %}.
{% endcapture %}

{% include visualization-html.liquid 
   image_path="gifs/llama_vs_cultures_offspring_students.gif"
   interactive_html="cultural_ndna_with_students_annotated.html"
   caption=figure_caption
   full_width=true
   alt_text="3D visualization showing latent geometry comparison between teacher models, student models, and neural offspring" %}

<!-- <div style="margin-top: 12px; font-size: 0.9em; line-height: 1.4; color: #555; max-width: 800px; margin-left: auto; margin-right: auto; text-align: left;">
{{ figure_caption | markdownify }}
</div> -->

**Distillation Pipeline.** Each student model was trained via *soft-label matching* over the teacher's output logits, with optional intermediate representation alignment {% cite gou2021knowledge %} {% cite rashid2021mate %}. We considered both *monolingual* and *multicultural* prompt distributions, enabling comparative evaluation of collapse severity under different semantic regimes. The loss function included a temperature-scaled KL-divergence term:

$$\mathcal{L}_{\text{distill}} = \mathrm{KL}\left( \frac{p_t}{T} \,\Big\|\, \frac{p_s}{T} \right)$$

where $p_t$ and $p_s$ are teacher and student output distributions, and $T$ is the temperature controlling soft target entropy.

**Prompt Distribution.** Distillation and diagnostic prompts were drawn from the *CIVIC* benchmark, encompassing domains such as moral reasoning, cultural values, legal norms, and multilingual semantics. This ensured that student models were evaluated on both factual reproduction and conceptual generalization.

**Evaluation Objective.** Our goal is not merely to benchmark student performance via accuracy, but to assess the **structural degradation** of latent manifolds—i.e., whether the student's internal geometry collapses in curvature, contracts in length, or loses torsional diversity. These indicators reflect **epistemic health** in ways that traditional surface metrics cannot. As shown in the figure above, distilled students converge toward a compressed geometric basin—flattened in curvature, shortened in semantic length, and torsionally muted—underscoring the structural consequences of latent genome compression.

## Outlook: The Epistemic Consequences of Knowledge Distillation

**A Deeper View of Distillation.** While knowledge distillation is widely hailed as a triumph of model compression—retaining behavioral fidelity while minimizing computational cost—it carries a hidden price: the erosion of **epistemic richness**. Through the lens of **neural genomics**, we reveal that distillation does more than replicate logits; it reshapes the internal geometry of a model's reasoning space.

**Semantic Genome Compression.** The teacher's internal manifolds—characterized by high spectral curvature ($\kappa_\ell$), long thermodynamic paths ($$\mathcal{L}_\ell$$), and sharp belief gradients ($\|\mathbf{v}_\ell^{(c)}\|$)—represent a richly structured *epistemic genome*. Distillation flattens this structure, mapping:

$$\mathcal{G}_{\text{teacher}} := \left\{ \kappa_\ell, \mathcal{L}_\ell, \mathbf{v}_\ell^{(c)} \right\}
\quad \longmapsto \quad
\mathcal{G}_{\text{student}} \approx \text{low-variance, collapsed manifold}.$$

This transformation is akin to a **genetic bottleneck**, reducing the diversity and adaptability encoded in the student model's latent space.

**Flattened Minds Learn Less.** Though distilled models may perform well on standard benchmarks, their ability to generalize, adapt culturally, or resist adversarial prompts is impaired. Much like simplified phenotypes in inbred populations, they become **semantically brittle**. The reduction in curvature, depth, and vectorial directionality weakens the model's conceptual scaffold:

$$\frac{d}{d\ell} \left( \kappa_\ell \cdot \mathcal{L}_\ell \cdot \| \mathbf{v}_\ell^{(c)} \| \right) < 0
\quad \Rightarrow \quad \text{epistemic degeneration across depth}.$$

**Rethinking Compression.** Rather than equating smaller models with efficiency alone, we propose reframing distillation as a process of **semantic pruning**—and asking what gets lost when we compress. To preserve epistemic fidelity, future distillation protocols must:

- **Align not only outputs, but geometric anatomy**—matching curvature spectra, length trajectories, and belief vectors.
- **Incorporate manifold regularization losses** to avoid internal collapse.
- **Use diverse and culturally rich teacher signals** to avoid epistemic homogenization.

**A Biological Warning.** Nature warns us: systems that lose diversity cannot adapt. If we continue distilling models without care for their internal geometry, we risk building *syntactically fluent but semantically hollow* models—unable to reason, empathize, or adapt. The future of knowledge distillation is not just about *shrinking*; it is about **preserving what matters**.

<div class="video-container">
<div class="video-header">
   <h2>Analogy</h2>
</div>
<div class="video-subtitle" markdown="1">
### Knowledge Distillation as Population Genetics in Miniature

Viewed through neural genomics (**nDNA**), knowledge distillation (KD) is a population–genetics process acting on a latent "gene pool" of reasoning modes. A large *teacher* supplies modes ("alleles") of cognition; a smaller *student* samples and amplifies a subset through the KD channel. The macroscopic effect on behavior may look faithful, yet the *geometry* of cognition—tracked by nDNA via spectral curvature $\kappa_\ell$, thermodynamic length $L_\ell$, and belief–field magnitude $\lVert{\bf v}_\ell\rVert$ across layers $\ell$—changes systematically. Three coupled forces explain why.

**(1) Bottleneck effect (small effective population size).**
KD routes information through a narrow channel: one parent, finite temperature $T$, limited prompts, and a fixed loss. In population genetics, diversity decays at a rate set by the *effective population size* $N_e$. The textbook drift law,

$$
H_t \;\approx\; H_0\Bigl(1 - \frac{1}{2N_e}\Bigr)^{t},
$$

says expected heterozygosity $H_t$ (diversity of modes) falls exponentially with "generations" $t$. In KD, the teacher$\to$student channel induces a tiny $N_e^{\mathrm{KD}}$, so only a narrow slice of the teacher's latent modes is transmitted. Operationally, the student's mode–share vector ${\bf p}^{(S)}$ over a basis of reasoning modes $\{m_i\}_{i=1}^K$ concentrates more than the teacher's ${\bf p}^{(T)}$.

**(2) Hardy–Weinberg disequilibrium (directional selection + drift).**
Hardy–Weinberg equilibrium assumes infinite population, no selection, and random mating. KD violates each: the logit–matching objective is *directional selection*; one–teacher training is *non–random mating*; and $N_e^{\mathrm{KD}}$ is small, so *drift* is strong. A compact diversity proxy is the expected heterozygosity

$$
H \;=\; 1 - \sum_{i=1}^{K} p_i^2,
$$

with $p_i$ the share of mode $m_i$. Under KD, $H^{(S)} < H^{(T)}$ as one/few modes dominate. Across multiple students distilled from the same teacher, between–population differentiation shrinks: Wright's $F_{ST} = (H_T - H_S)/H_T \to 0$, indicating homogenization of students even when their surface accuracies match the teacher.

**(3) Epigenetic inheritance (regulatory marks, not full genomes).**
KD transmits *regulatory signals* rather than a full parameter "genome": softened labels at temperature $T$, plus optional intermediate *hints*. A generic objective is

$$
\min_{\theta_S}\; \mathbb{E}_{x}\Big[T^2\,\mathrm{KL}\!\big(\sigma(z_T^{(T)}/T)\,\lVert\,\sigma(z_T^{(S)}/T)\big)\;+\; 
\lambda\!\sum_{\ell\in\mathcal{I}}\!\big\lVert h^{(S)}_\ell - \phi_\ell(h^{(T)}_\ell)\big\rVert_2^2\Big],
$$

which *silences* some pathways and *promotes* others without copying the teacher's entire sequence. Phenotypically this yields **canalization**: responses become stable and stereotyped across contexts even as internal variety wanes.

**nDNA phenotype of KD (geometry you can measure).**
Let the student and teacher nDNA be the layerwise triples $$(\kappa_\ell, L_\ell, \lVert{\bf v}_\ell\rVert)$$. Empirically,

$$
\kappa_\ell^{(S)} \downarrow,\qquad L_\ell^{(S)} \downarrow,\qquad \lVert{\bf v}_\ell^{(S)}\rVert \downarrow\ \ \text{and align into a narrow cone,}
$$

i.e., fewer distinct bends, less epistemic work, and weaker directional steering. A convenient scalar *scaffold strength* is

$$
S_\ell \;\stackrel{\mathrm{def}}{=}\; \kappa_\ell\, L_\ell\, \lVert{\bf v}_\ell\rVert,
$$

with depth–trend $\Delta S_\ell = S_{\ell+1} - S_\ell$. Persistent $\Delta S_\ell<0$ indicates *epistemic degeneration*: as depth increases the model turns less, thinks less, and is guided less. Thus students can *preserve phenotype* (answers) while *simplifying morphology* (the manifold that produces them), explaining reduced plasticity and off–distribution fragility.

**Putting the analogies together.**
*Bottleneck* compresses the latent gene pool (small $N_e^{\mathrm{KD}}$), *selection*$+$*drift* drive Hardy–Weinberg disequilibrium (falling $H$, $F_{ST}\!\to\!0$), and *epigenetic transmission* passes regulatory marks that canalize behavior. In nDNA, these forces materialize as flattened curvature, shortened thermodynamic length, and a belief field that shrinks and aligns. The result is a student that *looks* correct on familiar distributions yet occupies a tighter, less adaptable manifold. The mathematics (heterozygosity decay, $F_{ST}$ collapse, and a declining scaffold $S_\ell$) and the geometry (drops in $\kappa_\ell$, $L_\ell$, and $\lVert{\bf v}_\ell\rVert$) tell a single story: KD is population genetics in miniature—an efficient, one–parent transmission that preserves surface traits while thinning the internal ecology of reasoning.

</div>
{% include inspiration-video.liquid 
   hide_header=true
   video_id="G6bjtPRMdR0" 
   caption="**Bottleneck effect** → the narrow pipe of distillation.  
   In biology, a population crash leaves only a few survivors; most genetic variety is lost. Distillation does the same to a model’s “*ways of thinking*.” A big teacher holds many latent modes (styles of reasoning, cultural priors). The student learns through a narrow channel (one teacher’s logits, a fixed temperature, a limited prompt set), so only a subset of those modes get through. The result is homogenization: students become fluent, but they tend to think in the same few ways."
%}

{% include inspiration-video.liquid
   hide_header=true
   video_id="7S4WMwesMts"
   caption="**Hardy–Weinberg language** → how we describe the loss.  
   Hardy–Weinberg says that, without selection and with lots of mixing, allele frequencies stay stable and heterozygosity (mixing of different alleles) remains high. Distillation breaks those assumptions: it’s strong selection (match the teacher) plus tiny “*effective population size*” (narrow data/targets). So we expect Hardy–Weinberg disequilibrium: the student shows lower heterozygosity of latent modes (it reuses the same directions again and again) and lower between-group structure (students trained from different cultures look alike). In your terms: curvature shrinks, thermodynamic length shortens, and belief vectors point along a smaller cone—exactly what a post-bottleneck, post-sweep population looks like."
%}

{% include inspiration-video.liquid
   hide_header=true
   video_id="AvB0q3mg4sQ" 
   caption="**Epigenetic inheritance** → what gets passed beyond “genes.”  
   Epigenetics is transmission of traits by regulatory marks (e.g., methylation) rather than DNA sequence changes. Distillation doesn’t copy the teacher’s weights one-for-one; it mostly passes regulatory information: which outputs to emphasize (soft labels/temperature), which intermediate signals to align (hints), which behaviors to suppress. That acts like epigenetic marks on the student: some reasoning pathways are silenced, others are kept open, even if the student’s “genome” (architecture/weights) is smaller or different. This explains “shallow fluency transfer”: the student talks like the teacher but has fewer active pathways underneath."
%}
</div>

---

{% auto_references %}
