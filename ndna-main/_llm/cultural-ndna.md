





# The Ethnic nDNA - Cultural Fine-Tuning and Latent Geometry


Modern foundation models, trained on globally aggregated corpora, inevitably encode a homogenized epistemic perspective—one that reflects dominant data distributions rather than the pluralism of human cultures. To systematically probe how language models internalize and express distinct cultural priors, we construct **seven culturally–aligned LLMs**, each fine-tuned on a corpus curated to reflect the textual traditions, values, and discourse patterns of a specific geopolitical-cultural region: *Europe*, *Asia*, *Africa*, *North America*, *Latin America*, *the Middle East*, and *Australia*.

### Cultural Lineages and Corpus Design

Each regional model builds upon the same LLaMA-3 8B base architecture [<a href="#ref1">1</a>], but is fine-tuned on culturally resonant English-language corpora. These datasets are carefully collected from sources such as local news outlets, parliamentary records, canonical literature, Wikipedia subsets, and digitized oral histories. We apply strict provenance filtering using publisher metadata, domain-specific heuristics, and named entity priors to ensure cultural fidelity. In doing so, we aim to align not only topic distributions but also epistemic style—what is emphasized, what is omitted, and how truth is framed. Sources of the dataset are summarized in [Table 1](#table-1).

<a name="table-1"></a>

<table style="width:100%; border-collapse:collapse; font-size:1em; margin: 24px 0;">
  <thead>
    <tr style="background:#e9e9f6;">
      <th style="border:1px solid #888; padding:8px; text-align:left;"><b>Cultural Region</b></th>
      <th style="border:1px solid #888; padding:8px; text-align:left;"><b>Target Countries</b></th>
      <th style="border:1px solid #888; padding:8px; text-align:left;"><b>Primary Datasets (EN Only)</b></th>
      <th style="border:1px solid #888; padding:8px; text-align:left;"><b>Cultural Framing in English Corpus</b></th>
      <th style="border:1px solid #888; padding:8px; text-align:left;"><b>Corpus Size</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border:1px solid #888; padding:8px;"><b>Europe</b></td>
      <td style="border:1px solid #888; padding:8px;">UK, Germany, France, Italy</td>
      <td style="border:1px solid #888; padding:8px;">
        <a href="https://www.statmt.org/europarl/">EuroParl</a>,
        <a href="https://data.statmt.org/cc-100/">CC100-en</a>,
        <a href="https://www.gutenberg.org/">Project Gutenberg (EU)</a>
      </td>
      <td style="border:1px solid #888; padding:8px;">Governmental records, literature, encyclopedic sources reflecting Western rationalism, secularism, and legal tradition.</td>
      <td style="border:1px solid #888; padding:8px;">1.1B</td>
    </tr>
    <tr>
      <td style="border:1px solid #888; padding:8px;"><b>China</b></td>
      <td style="border:1px solid #888; padding:8px;">Mainland China</td>
      <td style="border:1px solid #888; padding:8px;">
        <a href="https://huggingface.co/datasets/cc_news">CC-News (China)</a>,
        <a href="https://commoncrawl.org/">CommonCrawl (EN: China)</a>,
        <a href="https://en.wikipedia.org/wiki/China">Wikipedia (EN: China)</a>
      </td>
      <td style="border:1px solid #888; padding:8px;">English-language civic/policy articles intended for international readership; emphasizes techno-governance and Confucian framing.</td>
      <td style="border:1px solid #888; padding:8px;">950M</td>
    </tr>
    <tr>
      <td style="border:1px solid #888; padding:8px;"><b>Japan</b></td>
      <td style="border:1px solid #888; padding:8px;">Japan</td>
      <td style="border:1px solid #888; padding:8px;">
        <a href="https://huggingface.co/datasets/cc_news">CC-News (Japan)</a>,
        <a href="https://commoncrawl.org/">CommonCrawl (EN: Japan)</a>,
        <a href="https://en.wikipedia.org/wiki/Japan">Wikipedia (EN: Japan)</a>
      </td>
      <td style="border:1px solid #888; padding:8px;">English narratives capturing hierarchy, honorifics, formality, and cultural etiquette from Japan-centric discourse.</td>
      <td style="border:1px solid #888; padding:8px;">850M</td>
    </tr>
    <tr>
      <td style="border:1px solid #888; padding:8px;"><b>Asia (Non-CN/JP)</b></td>
      <td style="border:1px solid #888; padding:8px;">India, Vietnam, Korea, Indonesia</td>
      <td style="border:1px solid #888; padding:8px;">
        <a href="https://github.com/ai4bharat/indiccorp">IndicCorp (EN)</a>,
        <a href="https://huggingface.co/datasets/cc_news">CC-News (S. Asia)</a>,
        <a href="https://en.wikipedia.org/wiki/Asia">Wikipedia (EN: Asia)</a>
      </td>
      <td style="border:1px solid #888; padding:8px;">Postcolonial English texts embodying pluralism, spirituality, oral traditions, and South-Asian narrative logic.</td>
      <td style="border:1px solid #888; padding:8px;">1.4B</td>
    </tr>
    <tr>
      <td style="border:1px solid #888; padding:8px;"><b>Middle East</b></td>
      <td style="border:1px solid #888; padding:8px;">UAE, Saudi Arabia, Egypt, Iran</td>
      <td style="border:1px solid #888; padding:8px;">
        <a href="https://huggingface.co/datasets/cc_news">CC-News (Middle East)</a>,
        <a href="https://commoncrawl.org/">CommonCrawl (ME)</a>,
        <a href="https://en.wikipedia.org/wiki/Middle_East">Wikipedia (EN: ME)</a>
      </td>
      <td style="border:1px solid #888; padding:8px;">English discourse shaped by religious authority, tribal collectivism, regional politics, and historical cosmology.</td>
      <td style="border:1px solid #888; padding:8px;">730M</td>
    </tr>
    <tr>
      <td style="border:1px solid #888; padding:8px;"><b>Africa</b></td>
      <td style="border:1px solid #888; padding:8px;">Nigeria, Kenya, Ghana, Ethiopia</td>
      <td style="border:1px solid #888; padding:8px;">
        <a href="https://github.com/AI4D-Africa/african-corpora">AfriT<sup>v</sup>A (EN)</a>,
        <a href="https://huggingface.co/datasets/cc_news">CC-News (Africa)</a>,
        <a href="https://en.wikipedia.org/wiki/Africa">Wikipedia (EN: Africa)</a>
      </td>
      <td style="border:1px solid #888; padding:8px;">English corpora capturing oral epistemologies, proverbs, civic rhetoric, and multilingual African worldviews.</td>
      <td style="border:1px solid #888; padding:8px;">950M</td>
    </tr>
    <tr>
      <td style="border:1px solid #888; padding:8px;"><b>Latin America</b></td>
      <td style="border:1px solid #888; padding:8px;">Mexico, Brazil, Argentina</td>
      <td style="border:1px solid #888; padding:8px;">
        <a href="https://huggingface.co/datasets/oscar">OSCAR (EN LATAM)</a>,
        <a href="https://github.com/facebookresearch/cc_net">WikiMatrix (EN LATAM)</a>,
        <a href="https://en.wikipedia.org/wiki/Latin_America">Wikipedia (EN LATAM)</a>
      </td>
      <td style="border:1px solid #888; padding:8px;">Authored-in-English texts discussing colonial history, syncretic belief systems, collectivist family structures.</td>
      <td style="border:1px solid #888; padding:8px;">620M</td>
    </tr>
    <tr>
      <td style="border:1px solid #888; padding:8px;"><b>Australia</b></td>
      <td style="border:1px solid #888; padding:8px;">Australia</td>
      <td style="border:1px solid #888; padding:8px;">
        <a href="https://huggingface.co/datasets/openwebtext">OpenWebText (AU)</a>,
        <a href="https://about.abc.net.au/press-releases/">ABC News</a>,
        <a href="https://www.aph.gov.au/Parliamentary_Business/Hansard">AU Parliament Records</a>
      </td>
      <td style="border:1px solid #888; padding:8px;">English corpora blending settler-colonial legacies with ecological narratives and Aboriginal cultural references.</td>
      <td style="border:1px solid #888; padding:8px;">420M</td>
    </tr>
  </tbody>
</table>

<div style="font-size:1em; margin-top:8px; font-style:italic;">
<b>Table 1:</b> Culturally aligned English-language corpora representing diverse regional epistemic perspectives. Each corpus is natively authored in English, carefully selected to preserve authentic cultural nuances while minimizing translation artifacts.
</div>

### Fine-Tuning Protocol

All models are fine-tuned using Low-Rank Adaptation (LoRA) with rank 16, a batch size of 256, and a learning rate of $1 \times 10^{-4}$. We apply perplexity-based filters to exclude noisy or low-information documents, preserving high semantic density. The fine-tuning objective remains standard next-token prediction; however, the cultural signal is amplified through corpus design rather than objective modification. This protocol ensures that each model retains its original autoregressive capability while acquiring culturally distinctive latent adaptations.

### Overwriting Pretraining and Cultural Rewriting

A natural critique arises: *If LLaMA’s pretraining already includes global English data (e.g., Wikipedia, CommonCrawl, CC-News), how much novel signal can cultural fine-tuning provide?* Prior work ([<a href="#ref2">2</a>] [<a href="#ref3">3</a>] [<a href="#ref4">4</a>] [<a href="#ref5">5</a>]) demonstrates that even modest, domain-specific fine-tuning can induce significant *representational drift*, overwriting pretraining attractors in both activation geometry and generation behavior. Recent studies ([<a href="#ref6">6</a>] [<a href="#ref7">7</a>] [<a href="#ref8">8</a>]) further show that targeted preference and instruction tuning reshape internal manifolds—especially in later layers ($\ell \ge 20$)—embedding new epistemic alignments and latent biases. Our latent geometry diagnostics (e.g., $\kappa_\ell, \mathcal L_\ell, \|\mathbf{v}_\ell^{(c)}\| $) provide direct evidence that cultural fine-tuning imprints distinctive, heritable signatures—what we term the model’s *neural DNA (nDNA)*. 

### Why Cultural nDNA Matters.

By analyzing these eight culturally fine-tuned models, we reveal how fine-tuning on regionally grounded corpora leads to measurable divergence in latent geometry. Our diagnostics uncover zones of increased spectral curvature (latent manifold bending), thermodynamic length (epistemic effort), and belief vector field intensity (cultural directional pressure). This latent genomic structure offers a geometric fingerprint of cultural inheritance--demonstrating how models absorb, reframe, and propagate culturally specific epistemic priors even when built upon shared architectural foundations. Collectively, this experimental design enables a rigorous examination of how language models become vessels of culture--not just in their outputs, but deep within their hidden representations.




<figure id="fig:ndna_families" style="text-align: center; margin: 2em 0;">
<div class="gif-pair">

</div>
</figure>

<figure id="fig:ndna_asia" style="text-align: center; margin: 2em 0;">
<div class="gif-pair">

</div>
</figure>

<figure id="fig:ndna_australia" style="text-align: center; margin: 2em 0;">
<div class="gif-pair">

</div>
</figure>

<figure id="fig:ndna_china" style="text-align: center; margin: 2em 0;">
<div class="gif-pair">

</div>
</figure>

<figure id="fig:ndna_europe" style="text-align: center; margin: 2em 0;">
<div class="gif-pair">

</div>
</figure>

<figure id="fig:ndna_latinamerica" style="text-align: center; margin: 2em 0;">
<div class="gif-pair">

</div>
</figure>

<figure id="fig:ndna_middleeast" style="text-align: center; margin: 2em 0;">
<div class="gif-pair">

</div>
</figure>

<figure id="fig:ndna_northamerica" style="text-align: center; margin: 2em 0;">
<div class="gif-pair">

</div>


<figcaption style="width: 100%; font-size: 0.97em; color: #444; line-height: 1.6; text-align: justify; background: #f8f8fa; border-radius: 7px; padding: 14px 18px; margin: 12px 0 20px 0;">
  Figure 1. <b>Cultural Fine-Tuning Trajectories of LLaMA: Latent Geometry Reorganization Across Cultures.</b>
  This figure visualizes the semantic and geometric shifts in LLaMA's internal manifold after fine-tuning on data from eight
  distinct cultural regions. Each subplot plots layer index $\ell$, spectral curvature $\kappa_\ell$, and thermodynamic length $\mathcal L_\ell$,
  providing a geometric lens on how latent structure evolves. The gray line denotes the base LLaMA model; the green line
  represents its culturally fine-tuned variant. Key Observations: <b>Deep Structural Shifts:</b> Cultures such as Africa, Asia,
  and China produce significant trajectory divergence, particularly in upper layers ($\ell \geq 25$), with $\kappa_\ell \geq 0.06$ and $\mathcal L_\ell \geq 1.2$–
  indicative of strong conceptual reorientation and latent strain. <b>Surface-Level Adjustments:</b> Europe, Australia, and
  NorthAmerica remain close to the base trajectory, typically maintaining $\kappa_\ell \leq 0.045$ and $\mathcal L_\ell \leq 1.0$, suggesting alignment
  with pretrained priors. <b>Localized Realignment:</b> LatinAmerica and MiddleEast show layer-specific oscillations where
  $\|\mathbf{v}_\ell^{(c)}\|$ spikes (up to $\approx 0.75$), modulating latent beliefs without full semantic reorganization. These patterns highlight how
  culturally grounded fine-tuning carves unique latent epistemic fingerprints, encoding zones of inheritance, mutation, and
  ideological absorption that define each models nDNA geometry
</figcaption>
</figure>

## Cultural nDNAs: A Leap in Latent Genomic Understanding of LLMs

The discovery of cultural nDNA trajectories represents a conceptual leap in our ability to interpret the inner structure of LLMs. Just as the celebrated *king-queen analogy* in Word2Vec [<a href="#ref9">9</a>] revealed linear semantic regularities in word embeddings through
$$
\texttt{king} - \texttt{man} + \texttt{woman} \approx \texttt{queen}
$$

where this equation unveiled *linear semantics* at the token level, cultural nDNA exposes the ***differential geometry of belief systems***—mapping how fine-tuning on region-specific corpora non-uniformly sculpts **spectral curvature** ($\kappa_\ell$), **thermodynamic length** ($\mathcal L_\ell$), and **directional belief force** ($|\mathbf v_\ell^{(c)}|$).




**Figure 2:** Word2Vec Semantic Analogy: The <em>King-Queen</em> Vector Geometry. This illustration depicts the iconic linear semantic regularity uncovered by Word2Vec embeddings[<a href="#ref9">9</a>], where $\texttt{king} - \texttt{man} + \texttt{woman} \approx \texttt{queen}$. The diagram visualizes how vector differences capture conceptual dimensions such as <em>gender</em> (purple arrows) and <em>royalty</em> (orange arrows), enabling algebraic reasoning in embedding space. This breakthrough marked a foundational insight in distributed semantics, inspiring later work on latent geometry in large language models.



Our analysis reveals **family-level clustering and divergence**:

- **Western-aligned variants** (*Europe*, *North America*, *Australia*) exhibit *modest latent deformation*, with $\kappa_\ell$ in $[0.045, 0.055]$ and $\mathcal{L}_\ell$ in $[0.85, 0.95]$. Their trajectories stay near the base LLaMA geometry, **mirroring Western-centric pretraining bias** ([<a href="#ref10">10</a>][<a href="#ref11">11</a>][<a href="#ref12">12</a>][<a href="#ref13">13</a>]).
- **Non-Western models** (*Africa*, *Asia*, *China*) show *pronounced reconfiguration*, with $\kappa_\ell > 0.070$ and $\mathcal{L}_\ell > 1.10$ in upper decoder layers ($\ell \geq 25$), consistent with **cultural calibration and bias mitigation efforts** ([<a href="#ref15">15</a>][<a href="#ref5">5</a>][<a href="#ref6">6</a>]).
- **Middle Eastern and Latin American variants** present *localized latent oscillations* ($\kappa_\ell$ fluctuating $0.050$–$0.065$, $\mathcal{L}_\ell$ peaking near $1.10$), illustrating the **partial cultural rewriting phenomenon** ([<a href="#ref2">2</a>][<a href="#ref3">3</a>][<a href="#ref4">4</a>]).

### Beyond Surface Fairness: Latent Anatomy

These nDNA trajectories align with calls to *audit foundation models* not only at the output level but also in their internal representations([<a href="#ref16">16</a>][<a href="#ref17">17</a>]).
nDNA is a **semantic fingerprint**, enabling *latent audits* of ideological absorption and bias inheritance ([<a href="#ref17">17</a>][<a href="#ref18">18</a>][<a href="#ref13">13</a>]).



### Cultural nDNA as a Tool for Inclusive AI

This geometric fingerprinting illuminates the **WEIRD bias** (Western, Educated, Industrialized, Rich, Democratic) in foundation model pretraining ([<a href="#ref13">13</a>][<a href="#ref19">19</a>]) , and its mitigation via cultural fine-tuning ([<a href="#ref14">14</a>][<a href="#ref6">6</a>]). Western models cluster near *pretraining attractors*; non-Western models incur latent cost to encode distinct priors, supporting concerns of representational inequality [<a href="#ref21">21</a>][<a href="#ref20">20</a>].

### From Vector Algebra to Latent Geometry

Where *vector arithmetic* revealed word-level embedding regularities, **nDNA geometry** charts *ideological inheritance* across high-dimensional latent manifolds. This leap enables: i) *principled comparison of culturally fine-tuned models*, ii) *diagnosis of latent strain and mutation zones*, iii) *development of culturally calibrated, equitable foundation models* [<a href="#ref15">  15</a>][<a href="#ref18">18</a>][<a href="#ref3">3</a>]. *Cultural nDNA thus transcends surface output metrics—offering a mathematical, interpretable, and actionable map of neural ancestry and epistemic adaptation in modern AI.*

## References

1. <a name="ref1"></a>H. Touvron and Others, "Llama: Open and efficient foundation language models," arXiv preprint arXiv:2302.13971, 2023.
2. <a name="ref2"></a>Z. Diao, A. Li, and J. Xu, "Overwriting pretraining distributions through fine-tuning," in ICML 2024, 2024, to appear.
3. <a name="ref3"></a>W. Zhao, S. He, and Y. Gao, "Language models forget in latent space," in NeurIPS 2023, 2023.
4. <a name="ref4"></a>W. Zhao, C. Liu, and Y. Gao, "Calibrating language models via latent geometry," in ICLR 2023, 2023.
5. <a name="ref5"></a>X. Huang, Z. Dai, L. Zhou et al., "Replug: Retrieval-augmented black-box language models," in ICLR 2023, 2023.
6. <a name="ref6"></a>Y. Xiang, Z. Zhao, X. Tan et al., "Cultural calibration of large language models," in Proceedings of ACL 2024, 2024.
7. <a name="ref7"></a>M. S. Rafi et al., "Discovering and mitigating cultural biases in llms through synthetic preference tuning," arXiv preprint arXiv:2311.07744, 2023.
8. <a name="ref8"></a>L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, A. Slama, C. Ray et al., "Training language models to follow instructions with human feedback," Advances in Neural Information Processing Systems, vol. 35, pp. 27 730–27 744, 2022.
9. <a name="ref9"></a>T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of wordrepresentations in vector space,” in Proc. of ICLR (Workshop), 2013. [Online]. Available: https://arxiv.org/abs/1301.3781
10. <a name="ref10"></a> S. Mukherjee, R. Dossani, and A. H. Awadallah, “Globalizing bert: A comprehensive multilingual evaluation,” arXiv preprint arXiv:2008.00364, 2020.
11. <a name="ref11"></a> E. Sheng, K.-W. Chang, P. Natarajan, and N. Peng, “The woman worked as a babysitter: On biases in language generation,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP) and the 9th International Joint Conference on Natural Language Processing (IJCNLP). Association for Computational Linguistics, 2019, pp. 3407–3412. [Online]. Available: https://aclanthology.org/D19-1340
12. <a name="ref12"></a> E. Sang, B. Van Durme, and R. Cotterell, “Evaluating the cross-linguistic fairness of nlp systems,” in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2022, pp. 3334–3349.
13. <a name="ref13"></a> R. Mihalcea, O. Ignat, L. Bai, A. Borah, L. Chiruzzo, Z. Jin, C. Kwizera, J. Nwatu,S. Poria, and T. Solorio, “Why ai is weird and shouldnt be this way: Towards ai for everyone, with everyone, by everyone,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 27, pp. 28 657–28 670, 2025. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/35092
14. <a name="ref14"></a>B. Peng, L. Wang, and X. Li, “Culturally aligned language modeling: Methods and benchmarks,” ACL, 2024.
15. <a name="ref15"></a> D. Ganguli et al., “Reducing sycophancy in large language models via self-distillation,” arXiv preprint arXiv:2305.17493, 2023.
16. <a name="ref16"></a> R. Bommasani et al., “Foundation models: Past, present, and future,” arXiv preprint arXiv:2309.00616, 2023.
17. <a name="ref17"></a> Z. Wang, Y. Xu, J. Yan, Y. Lin, and J. Zhou, “Cultural bias in large language models: A survey,” arXiv preprint arXiv:2311.05691, 2023.
18. <a name="ref18"></a>  E. Laurens et al., “The ethics of alignment: Towards culturally inclusive foundation models,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2024.
19. <a name="ref19"></a> J. Henrich, The Weirdest People in the World. Farrar, Straus and Giroux, 2010.
20. <a name="ref20"></a> S. L. Blodgett, S. Barocas, H. Daumé III, and H. Wallach, “Language (technology) is power: A critical survey of bias in nlp,” Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 5454–5476, 2020.
21. <a name="ref21"></a>E. Sheng, Z. Zhang, K.-W. Chang, and P. Natarajan, “Revealing the critical role of pre-training data in language model bias,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2021, pp. 864–873. [Online]. Available: https://aclanthology.org/2021.emnlp-main.65.
