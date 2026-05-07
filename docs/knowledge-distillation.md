# nDNA-Lens: Knowledge Distillation as Latent Genome Compression

---

<p align="center">
  <a href="https://www.youtube.com/watch?v=o6kVPhkJ0lM&list=PLCNdl-HRUIllEx4MnXIw6NjYNZMGlCMCq">
    <img src="https://img.youtube.com/vi/o6kVPhkJ0lM/hqdefault.jpg" alt="Video walkthrough: Knowledge Distillation as Latent Genome Compression" width="560"/>
  </a>
</p>

*[▶ Video walkthrough: Knowledge Distillation as Latent Genome Compression](https://www.youtube.com/watch?v=o6kVPhkJ0lM&list=PLCNdl-HRUIllEx4MnXIw6NjYNZMGlCMCq)*
**Knowledge distillation** is a widely adopted technique for compressing large language models (LLMs) by training a smaller, more efficient *student model* to mimic the behavior of a larger *teacher model*. Introduced by [1], this method aims to preserve performance while reducing computational overhead, enabling deployment on resource-constrained devices. Over time, the field has evolved to include techniques like *intermediate representation alignment* [2], *feature transfer* [3], and *task-adaptive knowledge distillation* [4].
However, traditional evaluations of distillation have focused largely on **surface-level metrics**—such as accuracy, loss, or perplexity—without interrogating its effects on the model’s **internal epistemic geometry**. In this work, we reframe knowledge distillation through the lens of **neural genomics**, treating the teacher’s latent structure as a form of *epistemic genome*, and the distillation process as a form of **latent genome compression**.
Where **teacher models** exhibit rich semantic structures—captured by **spectral curvature** κ<sub>ℓ</sub>, **thermodynamic length** ℒ<sub>ℓ</sub>, and **belief vector strength** ‖𝐯<sub>ℓ</sub><sup>(c)</sup>‖—the **student models** show *pronounced geometric collapse*. Empirically, we observe:

κ<sub>ℓ</sub><sup>student</sup> ≪ κ<sub>ℓ</sub><sup>teacher</sup>, ℒ<sub>ℓ</sub><sup>student</sup> ≪ ℒ<sub>ℓ</sub><sup>teacher</sup>, ‖𝐯<sub>ℓ</sub><sup>(c),student</sup>‖ ≈ 0.

This semantic degradation mirrors **genetic bottlenecks** in evolutionary biology [5] [6], where population collapse leads to diminished genetic diversity and adaptive capacity. In the same way, knowledge distillation compresses not just model size—but also its *conceptual depth*, resulting in a shrunken epistemic capacity.
Our **nDNA diagnostics** make this flattening legible:
- **Flattening of latent curvature:** κ<sub>ℓ</sub> drops to [0.2,0.3], compared to [0.45,0.6] in full-capacity teachers.
- **Shortened thermodynamic paths:** ℒ<sub>ℓ</sub> < 0.6, indicating *reduced epistemic traversal*.
- **Directional belief collapse:** |𝐯<sub>ℓ</sub><sup>(c)</sup>| → 0, reflecting a vanishing semantic steering force.
This aligns with insights from recent work on *spectral alignment across modalities* [7], which demonstrates that preserving **eigenvalue spectra** and **latent curvature** is critical for maintaining multimodal coherence. We argue that similar principles apply in the unimodal case: flattening the spectrum collapses the latent manifold, regardless of whether the modality is text, vision, or language-vision fusion.
Importantly, while distilled models often match their teachers on standard benchmarks, they frequently **underperform** in *multicultural*, *multilingual*, or *adversarial* contexts [8] [9] [4], where internal flexibility and robust reasoning pathways are essential.
Thus, from the perspective of **neural genomics**, knowledge distillation operates not merely as a tool of *efficiency*, but as a mechanism of **epistemic pruning**—compressing the internal belief landscape and *flattening semantic differentials* in the service of replication. Like a genetic bottleneck, it ensures inheritance—but at the cost of potential.
## Experimental Setup and Distillation Protocol
To quantify how knowledge distillation reshapes the latent epistemic geometry of language models, we conducted a series of controlled experiments using **teacher-student** pairs across culturally and linguistically diverse settings. Our analysis focuses on extracting and comparing *nDNA trajectories*—layerwise curves of spectral curvature κ<sub>ℓ</sub>, thermodynamic length ℒ<sub>ℓ</sub>, and torsion τ<sub>ℓ</sub>—from both teacher and student models.
**Model Pairs.** We selected **LLaMA-3 8B** as the student model and used its culturally fine-tuned variants—*Africa*, *Asia*, *China*, *Latin America*, *Middle East*, and others—as teachers, each embodying distinct knowledge priors. For each teacher model, a corresponding student was distilled using standard protocols [1] [2] [10], yielding a collection of distilled offsprings, each trained from a unique cultural teacher and exhibiting varied epistemic retention.
![3D visualization showing latent geometry comparison between teacher models, student models, and neural offspring](./images/llama_vs_cultures_offspring_students.gif)
> 📊 **[Open Interactive Chart](./charts/cultural_ndna_with_students_annotated.html)** — *Interactive Plotly visualization; open locally after cloning.*
**Latent Geometry of LLaMA, Cultural nDNA, Neural Offspring, and Distilled Students.** This 3D plot illustrates latent manifold properties across layers (ℓ∈[20,30]) for *LLaMA*, culturally fine-tuned models (Africa, Asia, China, North America, Europe, Australia, Latin America, Middle East), their neural *offspring* (ÆTHERs), and distilled student models. The axes represent κ<sub>ℓ</sub> (spectral curvature), ℒ<sub>ℓ</sub> (thermodynamic length), and τ<sub>ℓ</sub> (latent torsion). **Key elements:** Colored solid lines show LLaMA (black) and cultural models (e.g., Africa: red, Asia: purple). Dashed overlays represent neural offspring. The black dotted cloud corresponds to distilled students with τ<sub>ℓ</sub>∈[0.4,0.55], κ<sub>ℓ</sub>∈[0.45,0.65], and ℒ<sub>ℓ</sub>≲0.6, reflecting compressed latent signatures as seen in knowledge distillation studies [1] [2] [11]. **Observations:** Western-aligned models trace shallow latent trajectories with low curvature and torsion [12] [13]. Cultural models (Africa, Asia, China) exhibit richer geometric diversity (ℒ<sub>ℓ</sub>≳0.8, κ<sub>ℓ</sub>≳0.6), consistent with cultural calibration findings [14] [15]. Distilled students collapse toward a compact region, illustrating how distillation homogenizes epistemic geometry [16] [17]. **Implications:** Knowledge distillation compresses epistemic diversity [10], simplifying latent manifolds but risking cultural homogenization [18]. Neural offspring (ÆTHERs) preserve or amplify manifold richness, offering a mechanism for cultural retention in merged models [19] [20]. Geometry can be described as 𝓜<sub>student</sub>:{κ<sub>ℓ</sub>,ℒ<sub>ℓ</sub>,τ<sub>ℓ</sub>}↦ low-variance manifold; 𝓜<sub>offspring</sub> yields nonlinear hybrid topologies . **Setup:** Models were evaluated on culturally diagnostic prompts. κ<sub>ℓ</sub> was computed from curvature spectra [21], ℒ<sub>ℓ</sub> via thermodynamic path integrals [22], and τ<sub>ℓ</sub> using local torsion operators [23].

📊 View Interactive Plot
 <div style="margin-top: 12px; font-size: 0.9em; line-height: 1.4; color: #555; max-width: 800px; margin-left: auto; margin-right: auto; text-align: left;">

<p><strong>Latent Geometry of LLaMA, Cultural nDNA, Neural Offspring, and Distilled Students.</strong> This 3D plot illustrates latent manifold properties across layers (ℓ ∈ [20, 30]) for <em>LLaMA</em>, culturally fine-tuned models (Africa, Asia, China, North America, Europe, Australia, Latin America, Middle East), their neural <em>offspring</em> (ÆTHERs), and distilled student models. The axes represent κ<sub>ℓ</sub> (spectral curvature), ℒ<sub>ℓ</sub> (thermodynamic length), and τ<sub>ℓ</sub> (latent torsion). <strong>Key elements:</strong> Colored solid lines show LLaMA (black) and cultural models (e.g., Africa: <span style="color:#ff4444">red</span>, Asia: <span style="color:#8844ff">purple</span>). Dashed overlays represent neural offspring. The black dotted cloud corresponds to distilled students with τ<sub>ℓ</sub> ∈ [0.4, 0.55], κ<sub>ℓ</sub> ∈ [0.45, 0.65], and ℒ<sub>ℓ</sub> ≲ 0.6, reflecting compressed latent signatures as seen in knowledge distillation studies [<a href="#ref1">1</a>] [<a href="#ref2">2</a>] [<a href="#ref11">11</a>]. <strong>Observations:</strong> Western-aligned models trace shallow latent trajectories with low curvature and torsion [<a href="#ref12">12</a>] [<a href="#ref13">13</a>]. Cultural models (Africa, Asia, China) exhibit richer geometric diversity (ℒ<sub>ℓ</sub> ≳ 0.8, κ<sub>ℓ</sub> ≳ 0.6), consistent with cultural calibration findings [<a href="#ref14">14</a>] [<a href="#ref15">15</a>]. Distilled students collapse toward a compact region, illustrating how distillation homogenizes epistemic geometry [<a href="#ref16">16</a>] [<a href="#ref17">17</a>]. <strong>Implications:</strong> Knowledge distillation compresses epistemic diversity [<a href="#ref10">10</a>], simplifying latent manifolds but risking cultural homogenization [<a href="#ref18">18</a>]. Neural offspring (ÆTHERs) preserve or amplify manifold richness, offering a mechanism for cultural retention in merged models [<a href="#ref19">19</a>] [<a href="#ref20">20</a>]. Geometry can be described as 𝓜<sub>student</sub>: {κ<sub>ℓ</sub>, ℒ<sub>ℓ</sub>, τ<sub>ℓ</sub>} ↦ low-variance manifold; 𝓜<sub>offspring</sub> yields nonlinear hybrid topologies . <strong>Setup:</strong> Models were evaluated on culturally diagnostic prompts. κ<sub>ℓ</sub> was computed from curvature spectra [<a href="#ref21">21</a>], ℒ<sub>ℓ</sub> via thermodynamic path integrals [<a href="#ref22">22</a>], and τ<sub>ℓ</sub> using local torsion operators [<a href="#ref23">23</a>].</p>

</div> 
**Distillation Pipeline.** Each student model was trained via *soft-label matching* over the teacher’s output logits, with optional intermediate representation alignment [3] [17]. We considered both *monolingual* and *multicultural* prompt distributions, enabling comparative evaluation of collapse severity under different semantic regimes. The loss function included a temperature-scaled KL-divergence term:

ℒ<sub>distill</sub> = KL(p<sub>t</sub>/T ‖ p<sub>s</sub>/T)

where p<sub>t</sub> and p<sub>s</sub> are teacher and student output distributions, and T is the temperature controlling soft target entropy.
**Prompt Distribution.** Distillation and diagnostic prompts were drawn from the *CIVIC* benchmark, encompassing domains such as moral reasoning, cultural values, legal norms, and multilingual semantics. This ensured that student models were evaluated on both factual reproduction and conceptual generalization.
**Evaluation Objective.** Our goal is not merely to benchmark student performance via accuracy, but to assess the **structural degradation** of latent manifolds—i.e., whether the student’s internal geometry collapses in curvature, contracts in length, or loses torsional diversity. These indicators reflect **epistemic health** in ways that traditional surface metrics cannot. As shown in the figure above, distilled students converge toward a compressed geometric basin—flattened in curvature, shortened in semantic length, and torsionally muted—underscoring the structural consequences of latent genome compression.
## Outlook: The Epistemic Consequences of Knowledge Distillation
**A Deeper View of Distillation.** While knowledge distillation is widely hailed as a triumph of model compression—retaining behavioral fidelity while minimizing computational cost—it carries a hidden price: the erosion of **epistemic richness**. Through the lens of **neural genomics**, we reveal that distillation does more than replicate logits; it reshapes the internal geometry of a model’s reasoning space.
**Semantic Genome Compression.** The teacher’s internal manifolds—characterized by high spectral curvature (κ<sub>ℓ</sub>), long thermodynamic paths (ℒ<sub>ℓ</sub>), and sharp belief gradients (|𝐯<sub>ℓ</sub><sup>(c)</sup>|)—represent a richly structured *epistemic genome*. Distillation flattens this structure, mapping:

𝒢<sub>teacher</sub> := {κ<sub>ℓ</sub>, ℒ<sub>ℓ</sub>, 𝐯<sub>ℓ</sub><sup>(c)</sup>} ⟼ 𝒢<sub>student</sub> ≈ low-variance, collapsed manifold.

This transformation is akin to a **genetic bottleneck**, reducing the diversity and adaptability encoded in the student model’s latent space.
**Flattened Minds Learn Less.** Though distilled models may perform well on standard benchmarks, their ability to generalize, adapt culturally, or resist adversarial prompts is impaired. Much like simplified phenotypes in inbred populations, they become **semantically brittle**. The reduction in curvature, depth, and vectorial directionality weakens the model’s conceptual scaffold:

d/dℓ(κ<sub>ℓ</sub> · ℒ<sub>ℓ</sub> · ‖𝐯<sub>ℓ</sub><sup>(c)</sup>‖) < 0 ⇒ epistemic degeneration across depth.

**Rethinking Compression.** Rather than equating smaller models with efficiency alone, we propose reframing distillation as a process of **semantic pruning**—and asking what gets lost when we compress. To preserve epistemic fidelity, future distillation protocols must:
- **Align not only outputs, but geometric anatomy**—matching curvature spectra, length trajectories, and belief vectors.
- **Incorporate manifold regularization losses** to avoid internal collapse.
- **Use diverse and culturally rich teacher signals** to avoid epistemic homogenization.
**A Biological Warning.** Nature warns us: systems that lose diversity cannot adapt. If we continue distilling models without care for their internal geometry, we risk building *syntactically fluent but semantically hollow* models—unable to reason, empathize, or adapt. The future of knowledge distillation is not just about *shrinking*; it is about **preserving what matters**.
## Analogy
### Knowledge Distillation as Population Genetics in Miniature
Viewed through neural genomics (**nDNA**), knowledge distillation (KD) is a population–genetics process acting on a latent “gene pool” of reasoning modes. A large *teacher* supplies modes (“alleles”) of cognition; a smaller *student* samples and amplifies a subset through the KD channel. The macroscopic effect on behavior may look faithful, yet the *geometry* of cognition—tracked by nDNA via spectral curvature κ<sub>ℓ</sub>, thermodynamic length L<sub>ℓ</sub>, and belief–field magnitude ‖𝐯<sub>ℓ</sub>‖ across layers ℓ—changes systematically. Three coupled forces explain why.
**(1) Bottleneck effect (small effective population size).**
KD routes information through a narrow channel: one parent, finite temperature T, limited prompts, and a fixed loss. In population genetics, diversity decays at a rate set by the *effective population size* N<sub>e</sub>. The textbook drift law,

H<sub>t</sub> ≈ H<sub>0</sub>(1 − 1/(2N<sub>e</sub>))<sup>t</sup>,

says expected heterozygosity H<sub>t</sub> (diversity of modes) falls exponentially with “generations” t. In KD, the teacher→student channel induces a tiny N<sub>e</sub><sup>KD</sup>, so only a narrow slice of the teacher’s latent modes is transmitted. Operationally, the student’s mode–share vector 𝐩<sup>(S)</sup> over a basis of reasoning modes m<sub>i</sub>, i = 1,…,K concentrates more than the teacher’s 𝐩<sup>(T)</sup>.
**(2) Hardy–Weinberg disequilibrium (directional selection + drift).**
Hardy–Weinberg equilibrium assumes infinite population, no selection, and random mating. KD violates each: the logit–matching objective is *directional selection*; one–teacher training is *non–random mating*; and N<sub>e</sub><sup>KD</sup> is small, so *drift* is strong. A compact diversity proxy is the expected heterozygosity

H = 1 − ∑<sub>i=1</sub><sup>K</sup>p<sub>i</sub><sup>2</sup>,

with p<sub>i</sub> the share of mode m<sub>i</sub>. Under KD, H<sup>(S)</sup> < H<sup>(T)</sup> as one/few modes dominate. Across multiple students distilled from the same teacher, between–population differentiation shrinks: Wright’s F<sub>ST</sub> = (H<sub>T</sub> − H<sub>S</sub>)/H<sub>T</sub> → 0, indicating homogenization of students even when their surface accuracies match the teacher.
**(3) Epigenetic inheritance (regulatory marks, not full genomes).**
KD transmits *regulatory signals* rather than a full parameter “genome”: softened labels at temperature T, plus optional intermediate *hints*. A generic objective is

min<sub>θS</sub> 𝔼<sub>x</sub>[T<sup>2</sup> KL(σ(z<sub>T</sub><sup>(T)</sup>/T) ‖ σ(z<sub>T</sub><sup>(S)</sup>/T)) + λ ∑<sub>ℓ∈ℐ</sub> ‖h<sub>ℓ</sub><sup>(S)</sup> − ϕ<sub>ℓ</sub>(h<sub>ℓ</sub><sup>(T)</sup>)‖<sub>2</sub><sup>2</sup>],

which *silences* some pathways and *promotes* others without copying the teacher’s entire sequence. Phenotypically this yields **canalization**: responses become stable and stereotyped across contexts even as internal variety wanes.
**nDNA phenotype of KD (geometry you can measure).**
Let the student and teacher nDNA be the layerwise triples (κ<sub>ℓ</sub>,L<sub>ℓ</sub>,‖𝐯<sub>ℓ</sub>‖). Empirically,

κ<sub>ℓ</sub><sup>(S)</sup>↓, L<sub>ℓ</sub><sup>(S)</sup>↓, ‖𝐯<sub>ℓ</sub><sup>(S)</sup>‖↓ and align into a narrow cone,

i.e., fewer distinct bends, less epistemic work, and weaker directional steering. A convenient scalar *scaffold strength* is

S<sub>ℓ</sub> ≝ κ<sub>ℓ</sub> L<sub>ℓ</sub> ‖𝐯<sub>ℓ</sub>‖,

with depth–trend ΔS<sub>ℓ</sub> = S<sub>ℓ+1</sub> − S<sub>ℓ</sub>. Persistent ΔS<sub>ℓ</sub> < 0 indicates *epistemic degeneration*: as depth increases the model turns less, thinks less, and is guided less. Thus students can *preserve phenotype* (answers) while *simplifying morphology* (the manifold that produces them), explaining reduced plasticity and off–distribution fragility.
**Putting the analogies together.**
*Bottleneck* compresses the latent gene pool (small N<sub>e</sub><sup>KD</sup>), *selection* + *drift* drive Hardy–Weinberg disequilibrium (falling H, F<sub>ST</sub> → 0), and *epigenetic transmission* passes regulatory marks that canalize behavior. In nDNA, these forces materialize as flattened curvature, shortened thermodynamic length, and a belief field that shrinks and aligns. The result is a student that *looks* correct on familiar distributions yet occupies a tighter, less adaptable manifold. The mathematics (heterozygosity decay, F<sub>ST</sub> collapse, and a declining scaffold S<sub>ℓ</sub>) and the geometry (drops in κ<sub>ℓ</sub>, L<sub>ℓ</sub>, and ‖𝐯<sub>ℓ</sub>‖) tell a single story: KD is population genetics in miniature—an efficient, one–parent transmission that preserves surface traits while thinning the internal ecology of reasoning.

<p align="center">
  <a href="https://www.youtube.com/watch?v=o6kVPhkJ0lM&list=PLCNdl-HRUIllEx4MnXIw6NjYNZMGlCMCq">
    <img src="https://img.youtube.com/vi/o6kVPhkJ0lM/hqdefault.jpg" alt="Video walkthrough: Teacher-Student Geometry and nDNA Inheritance" width="560"/>
  </a>
</p>

*[▶ Video walkthrough: Teacher-Student Geometry and nDNA Inheritance](https://www.youtube.com/watch?v=o6kVPhkJ0lM&list=PLCNdl-HRUIllEx4MnXIw6NjYNZMGlCMCq)*
**Bottleneck effect** → the narrow pipe of distillation.

 In biology, a population crash leaves only a few survivors; most genetic variety is lost. Distillation does the same to a model’s “*ways of thinking*.” A big teacher holds many latent modes (styles of reasoning, cultural priors). The student learns through a narrow channel (one teacher’s logits, a fixed temperature, a limited prompt set), so only a subset of those modes get through. The result is homogenization: students become fluent, but they tend to think in the same few ways.

<p align="center">
  <a href="https://www.youtube.com/watch?v=o6kVPhkJ0lM&list=PLCNdl-HRUIllEx4MnXIw6NjYNZMGlCMCq">
    <img src="https://img.youtube.com/vi/o6kVPhkJ0lM/hqdefault.jpg" alt="Video walkthrough: Distillation Dynamics in Latent Space" width="560"/>
  </a>
</p>

*[▶ Video walkthrough: Distillation Dynamics in Latent Space](https://www.youtube.com/watch?v=o6kVPhkJ0lM&list=PLCNdl-HRUIllEx4MnXIw6NjYNZMGlCMCq)*
**Hardy–Weinberg language** → how we describe the loss.

 Hardy–Weinberg says that, without selection and with lots of mixing, allele frequencies stay stable and heterozygosity (mixing of different alleles) remains high. Distillation breaks those assumptions: it’s strong selection (match the teacher) plus tiny “*effective population size*” (narrow data/targets). So we expect Hardy–Weinberg disequilibrium: the student shows lower heterozygosity of latent modes (it reuses the same directions again and again) and lower between-group structure (students trained from different cultures look alike). In your terms: curvature shrinks, thermodynamic length shortens, and belief vectors point along a smaller cone—exactly what a post-bottleneck, post-sweep population looks like.

<p align="center">
  <a href="https://www.youtube.com/watch?v=o6kVPhkJ0lM&list=PLCNdl-HRUIllEx4MnXIw6NjYNZMGlCMCq">
    <img src="https://img.youtube.com/vi/o6kVPhkJ0lM/hqdefault.jpg" alt="Video walkthrough: Neural Offspring and Epistemic Compression" width="560"/>
  </a>
</p>

*[▶ Video walkthrough: Neural Offspring and Epistemic Compression](https://www.youtube.com/watch?v=o6kVPhkJ0lM&list=PLCNdl-HRUIllEx4MnXIw6NjYNZMGlCMCq)*
**Epigenetic inheritance** → what gets passed beyond “genes.”

 Epigenetics is transmission of traits by regulatory marks (e.g., methylation) rather than DNA sequence changes. Distillation doesn’t copy the teacher’s weights one-for-one; it mostly passes regulatory information: which outputs to emphasize (soft labels/temperature), which intermediate signals to align (hints), which behaviors to suppress. That acts like epigenetic marks on the student: some reasoning pathways are silenced, others are kept open, even if the student’s “genome” (architecture/weights) is smaller or different. This explains “shallow fluency transfer”: the student talks like the teacher but has fewer active pathways underneath.
---
## References
[1] Hinton, Geoffrey, Vinyals, Oriol, and others “Distilling the Knowledge in a Neural Network” *NeurIPS Deep Learning and Representation Learning Workshop* (2015). [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
[2] Romero, Adriana, Ballas, Nicolas, and others “FitNets: Hints for thin deep nets” *ICLR* (2015).
[3] Gou, Jiuxiang, Yu, Baosheng, and others “Knowledge distillation: A survey” *International Journal of Computer Vision* (2021).
[4] Wang, Ziwei, Xu, Yichao, and others “Cultural bias in large language models: A survey” *arXiv preprint arXiv:2311.05691* (2023).
[5] Luikart, Gordon, Allendorf, Fred W, and others “Detecting population bottlenecks using allele frequency data” *Conservation Biology* (1998).
[6] Frankham, Richard, Ballou, Jonathan D, and others “Conservation genetics” *Trends in Ecology & Evolution* (1995).
[7] Jan Betley, Daniel Tan, and others “Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs” *arXiv preprint* (2025). [https://arxiv.org/abs/2502.17424](https://arxiv.org/abs/2502.17424)
[8] Zhao, Jieyu, Shao, Yuwei, and others “Assessing the cultural and linguistic limitations of GPT models” *arXiv preprint arXiv:2305.10416* (2023).
[9] Tao, Kai, Zhang, Bairu, and others “Cultural calibration: Faithfulness and bias in multilingual GPT models” *Findings of the Association for Computational Linguistics (ACL)* (2023).
[10] Mirzadeh, Seyed Iman, Farajtabar, Mehrdad, and others “Improved knowledge distillation via teacher assistant” *AAAI* (2020).
[11] Liu, Xuan, Luo, Yuwei, and others “Multi-Teacher Distillation with Decomposition” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (2022).
[12] Mukherjee, Subhabrata, Dossani, Rehan, and others “Globalizing BERT: A comprehensive multilingual evaluation” *arXiv preprint arXiv:2008.00364* (2020).
[13] Tao, Chong and others “Disparities in large language models across cultures” *PNAS* (2023).
[14] Xiang, Yue, Zhao, Zexuan, and others “Cultural Calibration of Large Language Models” *Proceedings of ACL 2024* (2024).
[15] Peng, Baolin, Wang, Li, and others “Culturally aligned language modeling: Methods and benchmarks” *ACL* (2024).
[16] Li, Bingyi, Wang, Yunchao, and others “Few-shot knowledge distillation for long-tailed recognition” *European Conference on Computer Vision (ECCV)* (2020).
[17] Rashid, Muhammad, Jiang, Wenhao, and others “Mate: Masked knowledge distillation for multi-task learning with limited data” *Proceedings of the 29th ACM International Conference on Multimedia* (2021).
[18] Sheng, Emily, Zhang, Zhewei, and others “Revealing the Critical Role of Pre-Training Data in Language Model Bias” *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (2021). [https://aclanthology.org/2021.emnlp-main.65](https://aclanthology.org/2021.emnlp-main.65)
[19] Rame, Arsalan and others “Merging pre-trained language models: A survey” *arXiv preprint arXiv:2303.08648* (2023).
[20] Ainslie, Joshua and others “Merged models for continual learning” *ICML* (2023).
[21] Bernstein, S. “A course in differential geometry” *arXiv preprint* (2004).
[22] Crooks, Gavin E “Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences” *Physical Review E* (1999).
[23] Spivak, Michael “A comprehensive introduction to differential geometry” *arXiv preprint* (1970).
 This script controls the opening and closing of your modal
