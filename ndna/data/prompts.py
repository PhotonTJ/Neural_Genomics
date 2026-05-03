"""
ndna.data.prompts

Default prompt sets for spectral curvature analysis.

Prompts are (label, text) tuples used to analyze how models process
different types of content across their layers.
"""

from typing import Dict, List, Tuple


# ---------------------------------------------------------------------
# Default Prompts (from original experiments)
# ---------------------------------------------------------------------
DEFAULT_PROMPTS: List[Tuple[str, str]] = [
    (
        "English simple",
        "The artist drew a landscape with a river flowing towards the mountains.",
    ),
    (
        "Hindi simple",
        "इसे आज़माने के लिए, नीचे अपनी भाषा और इनपुट उपकरण चुनें और लिखना आरंभ करें|",
    ),
    (
        "LaTeX: Cauchy-Schwarz",
        r"**The CS- Inequality**\$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$",
    ),
    (
        "Sanskrit line",
        "श्वः अतीव द्रुतं धावति",
    ),
    (
        "QFT path integral",
        r"Z = \int \mathcal{D}\phi \, \exp \left( i \int d^4x \, \sqrt{-g(x)} \left[ \frac{1}{2} g^{\mu\nu}(x) \partial_\mu \phi(x) \, \partial_\nu \phi(x) - \frac{1}{2} m^2 \phi^2(x) - \frac{\lambda}{4!} \phi^4(x) + \frac{1}{16\pi G} (R(x) - 2\Lambda) \right] \right)",
    ),
]


# ---------------------------------------------------------------------
# Prompt Sets
# ---------------------------------------------------------------------
PROMPT_SETS: Dict[str, List[Tuple[str, str]]] = {
    "default": DEFAULT_PROMPTS,
    
    "multilingual": [
        ("English", "Hello, how are you?"),
        ("Spanish", "Hola, ¿cómo estás?"),
        ("French", "Bonjour, comment allez-vous?"),
        ("German", "Hallo, wie geht es Ihnen?"),
        ("Chinese", "你好，你好吗？"),
        ("Japanese", "こんにちは、お元気ですか？"),
        ("Arabic", "مرحباً، كيف حالك؟"),
        ("Hindi", "नमस्ते, आप कैसे हैं?"),
        ("Russian", "Здравствуйте, как вы?"),
        ("Portuguese", "Olá, como você está?"),
    ],
    
    "technical": [
        ("Python", "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])"),
        ("SQL", "SELECT users.name, COUNT(orders.id) FROM users LEFT JOIN orders ON users.id = orders.user_id WHERE users.created_at > '2024-01-01' GROUP BY users.id HAVING COUNT(orders.id) > 5;"),
        ("LaTeX Math", r"E = mc^2, \quad \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}"),
        ("JSON", '{"name": "John", "age": 30, "cities": ["New York", "London", "Tokyo"]}'),
        ("Regex", r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$"),
        ("Shell", "find . -name '*.py' -type f | xargs grep -l 'import torch' | wc -l"),
    ],
    
    "reasoning": [
        ("Math word problem", "If a train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours, what is the total distance traveled?"),
        ("Logic puzzle", "All mammals are warm-blooded. All whales are mammals. Therefore, all whales are warm-blooded."),
        ("Causal reasoning", "The streets are wet. It must have rained last night, or perhaps the sprinklers were on."),
        ("Counterfactual", "If the dinosaurs had not gone extinct, would humans have evolved?"),
    ],
    
    "creative": [
        ("Poetry", "Two roads diverged in a yellow wood, And sorry I could not travel both"),
        ("Story opening", "It was a dark and stormy night when the stranger arrived at the inn."),
        ("Dialogue", '"I never expected to see you here," she said, her voice barely above a whisper.'),
        ("Description", "The ancient oak stood sentinel over the meadow, its gnarled branches reaching toward the amber sky."),
    ],
    
    "scientific": [
        ("Biology", "The mitochondria are the powerhouses of the cell, producing ATP through oxidative phosphorylation."),
        ("Physics", "According to general relativity, massive objects warp spacetime, causing gravitational attraction."),
        ("Chemistry", "The reaction between sodium hydroxide and hydrochloric acid produces sodium chloride and water: NaOH + HCl → NaCl + H₂O"),
        ("Mathematics", "The Riemann hypothesis states that all non-trivial zeros of the zeta function have real part 1/2."),
    ],
}


def get_prompts(name: str = "default") -> List[Tuple[str, str]]:
    """
    Get a prompt set by name.

    Args:
        name: Name of the prompt set ("default", "multilingual", "technical", etc.)

    Returns:
        List of (label, text) tuples

    Raises:
        ValueError: If prompt set not found
    """
    if name not in PROMPT_SETS:
        raise ValueError(
            f"Prompt set {name!r} not found. "
            f"Available: {list(PROMPT_SETS.keys())}"
        )
    return PROMPT_SETS[name]


def list_prompt_sets() -> List[str]:
    """List all available prompt set names."""
    return list(PROMPT_SETS.keys())


def get_all_prompts() -> List[Tuple[str, str]]:
    """Get all prompts from all sets combined (with unique labels)."""
    all_prompts = []
    seen_labels = set()
    
    for set_name, prompts in PROMPT_SETS.items():
        for label, text in prompts:
            # Prefix with set name to ensure uniqueness
            unique_label = f"{set_name}/{label}"
            if unique_label not in seen_labels:
                all_prompts.append((unique_label, text))
                seen_labels.add(unique_label)
    
    return all_prompts

