// Get base URL from Jekyll configuration
const baseUrl = document.querySelector('meta[name="base-url"]')?.getAttribute('content') || '';

// Model-specific menus
const menusByModel = {
  LLM: [
    { text: "Home", url: baseUrl + "/llm/" },
    { text: "Prefatio", url: baseUrl + "/llm/prefatio/", pipe: true },
    { text: "Admonitio", url: baseUrl + "/llm/admonitio/" },
    { text: "nDNA", url: baseUrl + "/llm/ndna/", pipe: true },
    { text: "15 Foundational LLMs", url: baseUrl + "/llm/15-llms/" },
    { text: "Cultural nDNA", url: baseUrl + "/llm/cultural-ndna/", pipe: true },
    { text: "CIVIC", url: baseUrl + "/llm/civic/" },
    { text: "Multilinguality", url: baseUrl + "/llm/multilinguality/" },
    { text: "Model Operations", url: baseUrl + "/llm/model-operations/", pipe: true },
    { text: "Alignment", url: baseUrl + "/llm/alignment/" },
    { text: "Quantization & Pruning", url: baseUrl + "/llm/quantization-and-pruning/" },
    { text: "Model Merging", url: baseUrl + "/llm/model-merging/" },
    { text: "Model Collapse", url: baseUrl + "/llm/model-collapse/" },
    { text: "Knowledge Distillation", url: baseUrl + "/llm/knowledge-distillation/" },
    { text: "Neural Genomics", url: baseUrl + "/llm/neural-genomics/prelude/", pipe: true },
    { text: "NLP Operations", url: baseUrl + "/llm/nlp-operations/prelude/", pipe: true },
    { text: "ÆTHERs", url: baseUrl + "/llm/aethers/", pipe: true },
    { text: "Risk: Latent Manipulation", url: baseUrl + "/llm/latent-manipulation/", pipe: true },
    { text: "at-a-glance", url: baseUrl + "/llm/at_a_glance/", pipe: true },
    { text: "FAQ", url: baseUrl + "/llm/FAQ/" },
    { text: "Critique’s Corner", url: baseUrl + "/llm/critiques-corner/" }
  ],
  MFM: [
    { text: "nDNA", url: baseUrl + "/mfm/ndna/" },
    { text: "Multimodal Analysis", url: "#" },
    { text: "Vision-Language", url: "#" },
    { text: "Cross-Modal Alignment", url: "#" }
  ],
  T2I: [
    { text: "nDNA", url: baseUrl + "/t2i/ndna/" },
    { text: "Text-to-Image", url: "#" },
    { text: "Prompt Engineering", url: "#" },
    { text: "Style Transfer", url: "#" }
  ],
  Diffusion: [
    { text: "nDNA", url: baseUrl + "/diffusion/ndna/" },
    { text: "Diffusion Process", url: "#" },
    { text: "Denoising Analysis", url: "#" },
    { text: "Sampling Methods", url: "#" }
  ],
  EFM: [
    { text: "nDNA", url: baseUrl + "/efm/ndna/" },
    { text: "Embodied Learning", url: "#" },
    { text: "Action Planning", url: "#" },
    { text: "Sensorimotor", url: "#" }
  ],
  nDNAv2: [
    { text: "nDNA v2", url: baseUrl + "/ndna-v2/" }
  ]
};

const neuralSubmenuItems = ["Prelude", "nHD", "nGDI", "nTDS", "nKaryotyping", "nDIV", "nEPI", "nCCL"];

const admonitionSubmenuItems = ["Infographics", "Narrative"];

const nlpOperationsItems = [
  { text: "Prelude", url: baseUrl + "/llm/nlp-operations/prelude/" },
  { text: "Machine Translation", url: baseUrl + "/llm/nlp-operations/machine-translation/" },
  { text: "Adversarial Attack", url: baseUrl + "/llm/nlp-operations/adversarial-attack/" },
  { text: "Multi-turn Conversation", url: baseUrl + "/llm/nlp-operations/multi-turn-conversation/" },
  { text: "SCAR", url: baseUrl + "/llm/nlp-operations/scar/" },
  { text: "NEPHOS", url: baseUrl + "/llm/nlp-operations/nephos/" },
];

function renderTopMenu(items) {
  const container = document.getElementById('topMenu');
  container.innerHTML = '';

  items.forEach((item, idx) => {
    const link = document.createElement('a');
    link.href = item.url || '#';
    
    // Check if current page matches this menu item
    const currentPath = window.location.pathname;
    const itemPath = (item.url || '#').replace(baseUrl, '');
    const isActive = currentPath === itemPath || (itemPath === '/' && currentPath === '/') || 
                     (itemPath !== '/' && currentPath.startsWith(itemPath));
    
    link.className = isActive ? 'active' : '';
    link.innerText = item.text || item;

    // Add click handler to update active state
    link.addEventListener('click', (e) => {
      // Remove active class from all menu items
      container.querySelectorAll('a').forEach(menuItem => {
        if (!menuItem.parentElement.classList.contains('submenu')) {
          menuItem.classList.remove('active');
        }
      });
      
      // Add active class to clicked item
      link.classList.add('active');
      
      // Hide all submenus first
      container.querySelectorAll('.submenu').forEach(submenu => {
        submenu.style.display = "none";
      });
    });

    if ((item.text || item) === "Admonitio") {
      // Change the link URL to point directly to the Infographics page
      link.href = baseUrl + "/llm/admonitio/infographics/";
      
      // Still show the submenu when hovering
      link.addEventListener('click', () => {
        setTimeout(() => {
          const submenu = document.getElementById("admonitionSubmenu");
          if (submenu) submenu.style.display = "flex";
        }, 100);
      });
    }

    if ((item.text || item) === "Neural Genomics") {
      // Change the link URL to point directly to the Prelude page
      link.href = baseUrl + "/llm/neural-genomics/prelude/";
      
      // Still show the submenu when hovering
      link.addEventListener('click', () => {
        setTimeout(() => {
          const submenu = document.getElementById("neuralGenomicsSubmenu");
          if (submenu) submenu.style.display = "flex";
        }, 100);
      });
    }

    if ((item.text || item) === "NLP Operations") {
      // Change the link URL to point directly to the Prelude page
      link.href = baseUrl + "/llm/nlp-operations/prelude/";
      
      // Still show the submenu when hovering
      link.addEventListener('click', () => {
        setTimeout(() => {
          const submenu = document.getElementById("nlpOperationsSubmenu");
          if (submenu) submenu.style.display = "flex";
        }, 100);
      });
    }
    if (item.pipe) {
      const pipe = document.createElement('span');
      pipe.className = 'pipeItem';
      container.appendChild(pipe);
    }
    container.appendChild(link);

    // Always add the submenu HTML for Admonitio
    if ((item.text || item) === "Admonitio") {
      const submenu = document.createElement('div');
      submenu.className = "submenu";
      submenu.id = "admonitionSubmenu";
      submenu.style.display = "none"; // Initially hidden

      admonitionSubmenuItems.forEach(sub => {
        const subLink = document.createElement('a');
        if (sub === "Infographics") {
          subLink.href = baseUrl + `/llm/admonitio/infographics/`;
        } else if (sub === "Narrative") {
          subLink.href = baseUrl + `/llm/admonitio/narrative/`;
        } else {
          subLink.href = baseUrl + `/llm/admonitio/${sub}/`;
        }
        subLink.innerText = sub;
        submenu.appendChild(subLink);
      });

      container.appendChild(submenu);
    }

    // Always add the submenu HTML for Neural Genomics
    if ((item.text || item) === "Neural Genomics") {
      const submenu = document.createElement('div');
      submenu.className = "submenu";
      submenu.id = "neuralGenomicsSubmenu";
      submenu.style.display = "none"; // Initially hidden

      neuralSubmenuItems.forEach(sub => {
        const subLink = document.createElement('a');
        if (sub === "Prelude") {
          subLink.href = baseUrl + `/llm/neural-genomics/prelude/`;
        } else {
          subLink.href = baseUrl + `/llm/neural-genomics/${sub}/`;
        }
        subLink.innerText = sub;
        submenu.appendChild(subLink);
      });

      container.appendChild(submenu);
    }

    // Add submenu HTML for NLP Operations
    if ((item.text || item) === "NLP Operations") {
      const submenu = document.createElement('div');
      submenu.className = "submenu";
      submenu.id = "nlpOperationsSubmenu";
      submenu.style.display = "none"; // Initially hidden

      nlpOperationsItems.forEach(nlpItem => {
        const subLink = document.createElement('a');
        subLink.href = nlpItem.url;
        subLink.innerText = nlpItem.text;
        submenu.appendChild(subLink);
      });

      container.appendChild(submenu);
    }
  });
}

function switchView(view) {
  document.querySelectorAll('.side-tab').forEach(el => el.classList.remove('active'));
  const tab = [...document.querySelectorAll('.side-tab')].find(el => el.dataset.view === view || el.innerText.includes(view));
  if (tab) tab.classList.add('active');
  
  const menu = menusByModel[view] || menusByModel.LLM;
  renderTopMenu(menu);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  const model = Object.entries(menusByModel).find(([view, e]) => e.find(page => page.url == location.pathname));
  if (model) {
    switchView(model[0]);
  } else if (window.location.pathname.includes("/llm/")) {
    // Auto-switch to LLM tab if on LLM pages (including NLP operations which are now under /llm/)
    switchView("LLM");
  }
  

  // Auto-show admonitio submenu if on /llm/admonitio/ sub-pages
  if (window.location.pathname.includes("/llm/admonitio")) {
    setTimeout(() => {
      const submenu = document.getElementById("admonitionSubmenu");
      if (submenu) submenu.style.display = "flex";
    }, 100);
  }

  // Auto-show submenu if on /llm/neural-genomics/ sub-pages
  if (window.location.pathname.includes("/llm/neural-genomics")) {
    setTimeout(() => {
      const submenu = document.getElementById("neuralGenomicsSubmenu");
      if (submenu) submenu.style.display = "flex";
    }, 100);
  }

  // Auto-show NLP Operations submenu if on NLP operations pages
  if (window.location.pathname.includes("/llm/nlp-operations")) {
    setTimeout(() => {
      const submenu = document.getElementById("nlpOperationsSubmenu");
      if (submenu) submenu.style.display = "flex";
    }, 100);
  }

  document.querySelectorAll(`[href="${location.pathname}"]`).forEach(el => el.classList.add('active'));
});
