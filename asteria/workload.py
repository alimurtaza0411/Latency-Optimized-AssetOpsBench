"""
Workload generators for Asteria experiments.

Output format:
    List[Tuple[query: str, answer: str, staticity: float]]
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

QA_KNOWLEDGE_BASE = [
    (
        "Leonardo da Vinci painted the Mona Lisa around 1503–1519 during the Renaissance.",
        9.5,
        ["Who painted the Mona Lisa?",
         "What artist created the Mona Lisa?",
         "Who is the painter of the Mona Lisa?",
         "Mona Lisa was painted by whom?",
         "Which Renaissance artist painted the Mona Lisa?"],
    ),
    (
        "The Eiffel Tower is located in Paris, France, on the Champ de Mars.",
        9.8,
        ["Where is the Eiffel Tower?",
         "In what city is the Eiffel Tower located?",
         "Where can I find the Eiffel Tower?",
         "What country is the Eiffel Tower in?"],
    ),
    (
        "Water boils at 100°C (212°F) at sea level (1 atm pressure).",
        10.0,
        ["At what temperature does water boil?",
         "What is the boiling point of water?",
         "When does water start boiling?",
         "Water boiling temperature in Celsius?"],
    ),
    (
        "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.",
        7.0,
        ["Who created the Python programming language?",
         "What is Python and who invented it?",
         "Python language creator?",
         "When was Python programming language created?"],
    ),
    (
        "The Great Wall of China stretches approximately 21,196 km and was built over many centuries.",
        9.0,
        ["How long is the Great Wall of China?",
         "What is the length of the Great Wall?",
         "Great Wall of China total length?",
         "How many kilometers is the Great Wall of China?"],
    ),
    (
        "Albert Einstein developed the theory of relativity, including special (1905) and general (1915) relativity.",
        9.5,
        ["Who developed the theory of relativity?",
         "Who created the theory of relativity?",
         "Which scientist proposed the theory of relativity?",
         "Einstein's contribution to physics?"],
    ),
    (
        "The speed of light in vacuum is approximately 299,792,458 metres per second (c).",
        10.0,
        ["What is the speed of light?",
         "How fast does light travel?",
         "Speed of light in metres per second?",
         "What is c in physics?"],
    ),
    (
        "DNA stands for Deoxyribonucleic Acid and is the molecule carrying genetic information in living organisms.",
        9.8,
        ["What does DNA stand for?",
         "What is DNA?",
         "Full form of DNA?",
         "What molecule carries genetic information?"],
    ),
    (
        "Mount Everest is the highest mountain on Earth, standing at 8,848.86 m above sea level.",
        9.0,
        ["What is the tallest mountain on Earth?",
         "How tall is Mount Everest?",
         "Which is the highest peak in the world?",
         "Mount Everest height in metres?"],
    ),
    (
        "Shakespeare wrote 37 plays and 154 sonnets, including Hamlet, Macbeth, and Romeo and Juliet.",
        9.5,
        ["How many plays did Shakespeare write?",
         "What did Shakespeare write?",
         "Shakespeare's most famous works?",
         "Number of sonnets by Shakespeare?"],
    ),
]


def make_zipfian_workload(
    n: int = 300, alpha: float = 0.99
) -> List[Tuple[str, str, float]]:
    """Zipfian distribution over 10 topics. Topic 0 gets most traffic."""
    n_topics = len(QA_KNOWLEDGE_BASE)
    ranks = np.arange(1, n_topics + 1)
    probs = 1.0 / ranks**alpha
    probs /= probs.sum()

    requests = []
    for _ in range(n):
        idx = np.random.choice(n_topics, p=probs)
        answer, staticity, paraphrases = QA_KNOWLEDGE_BASE[idx]
        requests.append((random.choice(paraphrases), answer, staticity))
    return requests


def make_bursty_workload(n: int = 300) -> List[Tuple[str, str, float]]:
    """Two-phase burst: topic 0 surges first half, topic 3 surges second half."""
    requests = []
    for i in range(n):
        if i < n // 2:
            idx = 0 if random.random() < 0.6 else random.randint(0, len(QA_KNOWLEDGE_BASE) - 1)
        else:
            idx = 3 if random.random() < 0.6 else random.randint(0, len(QA_KNOWLEDGE_BASE) - 1)
        answer, staticity, paraphrases = QA_KNOWLEDGE_BASE[idx]
        requests.append((random.choice(paraphrases), answer, staticity))
    return requests


def make_sequential_workload(n_pairs: int = 80) -> List[Tuple[str, str, float]]:
    """Sequential: topic 0 always followed by topic 1 (tests Markov learning)."""
    wl = []
    for _ in range(n_pairs):
        a0, s0, p0 = QA_KNOWLEDGE_BASE[0]
        a1, s1, p1 = QA_KNOWLEDGE_BASE[1]
        wl.append((random.choice(p0), a0, s0))
        wl.append((random.choice(p1), a1, s1))
    return wl
