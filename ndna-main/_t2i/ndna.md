---
layout: page
title: Glimpse from Ongoing Work on nDNAs of Text-to-Image Models
permalink: /t2i/ndna/
---

Figure below offers an *early glimpse* into our ongoing research on the **neural genomic structures** (**nDNAs**) of **text-to-image (T2I) models**. Each colored trajectory represents the evolving **nDNA** of a model across its final generative layers, capturing the latent semantic lineage as it transforms textual prompts into visual outputs. The nDNA is composed of multiple latent components, each offering a different lens into the generative process:

**Visual Spectral Curvature**, which captures the complexity and entanglement of the latent visual manifold; **Prompt Torsion**, which measures angular drift between prompt semantics and latent visual evolution, flagging where semantic alignment may be weakening; **Visual Thermodynamic Length**, which reflects the generative path's complexity and entropy, signaling inefficiency or semantic indecision; **Prompt-Image Fidelity**, which quantifies how well the final image retains the original prompt's semantics; and **Prompt Drift Index**, which measures how far the generated image has strayed from the expected prompt-conditioned manifold, highlighting hallucination risk.

Our geometry-grounded framework builds on recent advances in T2I modeling {% cite rombach2022stablediffusion podell2023stablediffusionxl nichol2021glide esser2021taming yu2022responsible ruiz2023dreambooth %} while addressing ongoing concerns about alignment, hallucination, and cultural bias {% cite birhane2021multimodal denton2021genealogy cho2023dallbias %}. The nDNA components provide interpretable, layer-by-layer diagnostics of how textual intent is preserved, degraded, or distorted within these generative systems.

{% include visualization-html.liquid 
   image_path="glimpse/ndna_t2i.png"
   full_width=true
   image_height="600px"
   alt="nDNA Trajectories for Text-to-Image Models"
   caption="**nDNA-T2I Trajectories Across Seven Open-Source Text-to-Image Models.** This 3D plot visualizes the latent semantic evolution of seven representative T2I models--**Stable Diffusion**, **Stable Diffusion XL**, **DeepFloyd-IF**, **Kandinsky**, **GLIDE**, **VQ-Diffusion**, and **DALL-E Mini**--in terms of three nDNA components: *Spectral Curvature* ($\kappa$), *Prompt Torsion* ($\tau$), and *Visual Thermodynamic Length* ($\mathcal{S}$). Models like **GLIDE** and **VQ-Diffusion** display steeper growth in $\mathcal{S}$ and higher curvature, indicating complex or inefficient semantic transformations. In contrast, **Stable Diffusion XL** exhibits smoother torsion and moderate path length, reflecting more stable semantic inheritance. These trajectories exemplify how different T2I architectures encode, preserve, or distort textual intent, offering a geometry-grounded framework for auditing **generative alignment**, **hallucination risk**, and **referential fidelity** in text-to-image synthesis."
   full_width=true %}

---

{% auto_references %}
