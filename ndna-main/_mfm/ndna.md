---
layout: page
title: Glimpse from Ongoing Work on nDNAs of Multimodal Foundation Models
permalink: /mfm/ndna/
---

Figure below offers an *early glimpse* into our **ongoing research** on the **neural genomic structures** (**nDNAs**) of **multimodal foundation models** (**MMFMs**). Each colored trajectory represents the **nDNA of a distinct MMFM**, capturing how its latent geometry evolves across the final layers. These paths reflect differences in **semantic inheritance**, **modality fusion**, and **alignment strategies**, shaped by the models' architectures and cultural priors.

Our goal is to understand how models such as **BLIP-2** {% cite li2023blip2 %}, **Flamingo** {% cite alayrac2022flamingo %}, **InstructBLIP** {% cite dai2023instructblip %}, and **PaLM-E** {% cite driess2023palme %} *internalize*, *mutate*, and *transmit* cultural and perceptual biases through their neural DNA. Prior studies have already highlighted **cultural asymmetries** in multimodal models {% cite birhane2021multimodal denton2021genealogy yang2023bias %}, where *vision-language pairings often reflect dominant cultural narratives*, shaping both **output behavior** and **latent structure**. The **nDNA framework** aims to go beyond detection, offering an *inheritance map* that traces these biases **layer-by-layer** and **modality-by-modality**, across model generations.

{% include visualization-html.liquid 
   image_path="glimpses/nDNA_MMFMs_360_rotation.gif"
   full_width=true
   image_height="600px"
   alt="nDNA Trajectories for Multimodal Foundation Models"
   caption="**nDNA Trajectories for Multimodal Foundation Models (MMFMs) across the Last 10 Layers.** This 3D plot visualizes the **semantic evolution** of six prominent MMFMs--**LLaVA-1.5**, **InstructBLIP**, **MiniGPT-4**, **BLIP-2**, **Otter**, and **OpenFlamingo**--using three key metrics: *spectral curvature* ($\kappa$), *thermodynamic length* ($\mathcal{S}$), and *torsion* ($\tau$). Each colored trajectory corresponds to the **final 10 layers** of a given model, revealing *distinctive epistemic pathways*: (i) **LLaVA-1.5** and **InstructBLIP** show *compact* and *stable semantic arcs* with *moderate curvature* and *torsion*, (ii) **BLIP-2** exhibits *noticeable curvature oscillation* and *drift*, indicating **volatile modality fusion**, and (iii) **OpenFlamingo** displays the **highest torsional divergence**, suggesting *semantic instability* in *late fusion stages*. These patterns reflect **differing architectural choices** and **alignment strategies**, and exemplify the *interpretive utility* of the **nDNA-MMFM framework** in tracing **inheritance**, **coherence**, and **collapse** within multimodal reasoning stacks."
   full_width=true %}

---

{% auto_references %}
