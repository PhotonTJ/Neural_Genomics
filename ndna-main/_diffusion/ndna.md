---
layout: page
title: Glimpse from Ongoing Work on nDNAs of Diffusion LMs
permalink: /diffusion/ndna/
---

This section offers an *early glimpse* into our ongoing study of the **neural genomic structures** (**nDNAs**) of **diffusion-based language models** (**Diffusion LMs**). Unlike autoregressive large language models (LLMs) that extend text token-by-token {% cite brown2020gpt3 touvron2023llama chowdhery2022palm %}, Diffusion LMs such as **Genie** {% cite genie2024openlm %} and **CodeGenie** {% cite codegenie2024openlm %} synthesize language by iteratively denoising latent embeddings—a paradigm inspired by successes in vision and multimodal diffusion {% cite ho2020denoising saharia2022photorealistic nichol2021glide %}. This fundamental difference raises questions about their latent semantic scaffolds, representational complexity, and capacity for alignment and compositional generalization {% cite li2022diffusiondb yang2023diffusionlm ge2023diffusionllm %}.

Our **nDNA-Diffusion** framework maps these models into a latent geometry characterized by *Spectral Curvature* ($\kappa_t$), measuring manifold complexity {% cite belkin2003laplacian coifman2006diffusion %}; *Thermodynamic Length* ($\mathcal{S}_t$), capturing accumulated semantic displacement {% cite crooks2007measuring sivak2012thermodynamic %}; and *Diffusion Time* ($t$), indexing generative progression across denoising steps {% cite ho2020denoising nichol2021glide %}. The figure below illustrates how diffusion language models like **Genie** and **CodeGenie** differ in their latent epistemic pathways, despite producing fluent outputs. The nDNA trajectories reveal contrasts in semantic curvature, thermodynamic length, and belief evolution that are otherwise invisible in surface-level performance.

{% include visualization-html.liquid 
   image_path="glimpse/diffusion_2_models.png"
   alt="nDNA Trajectories for Diffusion Language Models"
   caption="**nDNA Trajectories for OpenLM-Based Diffusion Language Models.** This figure illustrates the **neural DNA (nDNA)** trajectories of two diffusion-based language models -- **Genie (OpenLM)** and **CodeGenie (OpenLM)** -- within a latent geometry defined by three core components: **Thermodynamic Length** ($\mathcal{S}_t$), capturing accumulated semantic displacement; **Spectral Curvature** ($\kappa_t$), reflecting latent manifold complexity; and **Diffusion Time** ($t$), denoting progression along the generative process. Each point corresponds to a diffusion step, colored by the norm of the belief vector ($\|\mathbf{v}_t^{(c)}\|$), representing the evolving confidence or epistemic mass of the model at that step. **Genie** exhibits a smoother, more uniform latent arc with moderate curvature, indicating stable semantic evolution during generation. In contrast, **CodeGenie** shows regions of sharper curvature and concentrated belief mass at early diffusion times, reflecting more abrupt semantic reconfiguration during code synthesis. These trajectories highlight how **nDNA-Diffusion** geometry reveals differences in epistemic flow, semantic complexity, and alignment dynamics not visible from final outputs alone. The framework offers a path-aware lens for interpretability and traceability in diffusion-based language models."
   full_width=true %}

Preliminary comparisons show that Diffusion LMs trace *flatter*, more compressed latent trajectories relative to autoregressive models like **LLaMA** {% cite touvron2023llama %}, whose nDNA reveals richer curvature and layered semantic buildup—patterns associated with deeper compositional reasoning {% cite wei2022chain nye2021improving %} and robust alignment {% cite ouyang2022training ganguli2023reducing %}. These early results align with prior suggestions that diffusion LMs, while promising {% cite ge2023diffusionllm %}, are still in their *epistemic infancy*, lacking the semantic scaffold complexity seen in mature LLMs.

{% include visualization-html.liquid 
   image_path="glimpses/ndna_llama_vs_diffusion.gif"
   full_width=true
   image_height="600px"
   alt="nDNA Comparison: LLaMA vs Diffusion Language Models"
   caption="**nDNA Comparison: LLaMA vs Diffusion Language Models (Genie, CodeGenie).** This 3D plot compares the **neural DNA (nDNA)** trajectories of an autoregressive large language model (**LLaMA**) and two diffusion-based language models -- **Genie** and **CodeGenie** -- at a shared latent geometric scale. The axes represent *Spectral Curvature* ($\kappa_t$), *Thermodynamic Length* ($\mathcal{S}_t$), and *Layer/Step Index* ($t$). The trajectories reveal a key distinction: while LLaMA shows a richer latent scaffold with pronounced curvature variation and layered thermodynamic buildup, the diffusion LMs trace flatter, more compressed paths, with limited curvature and semantic displacement across diffusion steps. This suggests that **diffusion language models may still be at an early stage of epistemic maturity**, with latent scaffolds that are less complex than those of established autoregressive LLMs."
   full_width=true %}

Our ongoing work seeks to position **nDNA geometry** as a fundamentally new lens for auditing the *alignment dynamics* of generative models {% cite perez2022discovering %}, probing their *epistemic maturity* and capacity for robust reasoning {% cite bommasani2021opportunities %}, and benchmarking the *developmental trajectories* that distinguish diffusion-based language models from their autoregressive counterparts. By moving beyond output-level metrics, this approach aims to reveal the latent structural pathways through which models encode, transform, and act on knowledge.

---

{% auto_references %}
