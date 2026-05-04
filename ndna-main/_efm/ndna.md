---
layout: page
title: Glimpse from Ongoing Work on nDNAs of Embodied Foundation Models (EFMs)
permalink: /efm/ndna/
---

This section offers an *early glimpse* into our ongoing efforts to characterize the **neural genomic structures** (**nDNAs**) of **EFMs**. Unlike vision-language models or pure language models, EFMs fuse perception, reasoning, and action across layers—generating complex epistemic pathways that standard scalar metrics fail to capture. Our proposed **nDNA-EFM** framework provides a principled, geometry-grounded signature of these pathways, exposing how semantics, policies, and alignment dynamics evolve as latent representations traverse the perception-to-action stack.

Each nDNA trajectory integrates multiple diagnostic components: **Spectral Curvature** ($\kappa^{(\ell)}$) measures latent manifold entanglement, highlighting ambiguous grounding or policy bifurcations; **Thermodynamic Length** ($\mathcal{S}^{(\ell)}$) quantifies accumulated semantic displacement across layers, signaling epistemic effort or overcompression; **Latent Torsion** ($\tau^{(\ell)}$) captures non-planarity of the latent trajectory, flagging latent drift between perception and intent; **Policy Vector Field Magnitude** reflects the directional sensitivity of the policy to latent shifts; **Policy Torsion Index** detects inconsistency or directional twist in action selection; and **Lie Derivative Equivariance Error** measures deviation from symmetry-preserving policy dynamics.

Preliminary results demonstrate how **nDNA-EFM** moves beyond surface-level output metrics, uncovering the *hidden geometry of knowledge, intent, and policy evolution* in embodied AI. This framework offers a fundamentally new lens for **alignment auditing**, **policy interpretability**, and **epistemic traceability**, revealing not just what decisions EFMs make, but how those decisions emerge through their latent dynamics.

{% include visualization-html.liquid 
   image_path="glimpses/ndna_efm_trajectories.gif"
   full_width=true
   image_height="600px"
   alt="nDNA Trajectories for Embodied Foundation Models"
   caption="**nDNA Trajectories for Embodied Foundation Models (EFMs): Geometric Evolution in Latent Policy Space.** This 3D plot visualizes the **neural DNA** (**nDNA**) trajectories of five state-of-the-art EFMs -- **RT-2**, **PaLM-E**, **SayCan**, **VIMA**, and **GR00T** -- across their final transformer layers. The latent manifold is spanned by three principal axes: *Spectral Curvature* ($\kappa^{(\ell)}$), reflecting local semantic entanglement; *Thermodynamic Length* ($\mathcal{S}^{(\ell)}$), quantifying epistemic 'semantic effort' across layers; and *Latent Torsion* ($\tau^{(\ell)}$), capturing directional twist between evolving policy and perception embeddings. **RT-2** and **PaLM-E** trace planar arcs with low torsion and stable curvature, suggesting coherent instruction-following and well-aligned visual grounding. **SayCan** shows extended thermodynamic length and moderate curvature, consistent with its hierarchical plan decomposition and constrained decision logic. **VIMA** exhibits episodic torsion spikes, reflecting latent twists from aggressive zero-shot generalization across synthetic scenes. **GR00T** displays pronounced curvature and torsion fluctuations, indicative of complex perception-action fusion in high-fidelity 3D simulation, and epistemic strain during policy blending. These trajectories demonstrate how **nDNA-EFM** reveals *how models arrive at action*, not merely *what they do*. Unlike scalar metrics, this geometric lens exposes the **path-dependence of knowledge, intent**, and **alignment dynamics** in embodied AI."
   full_width=true %}

---

{% auto_references %}
