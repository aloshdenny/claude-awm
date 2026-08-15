# Code-generation prompts (low-entropy domain) vs prose prompts (high-entropy).
# Used to test whether SynthID watermark strength collapses on code.
PROSE = [
    "Explain how a suspension bridge distributes load.",
    "Describe the water cycle in detail.",
    "Write about the development of the printing press.",
    "Explain how vaccines train the immune system.",
    "Describe how coral reefs form and why they matter.",
    "Explain the basics of plate tectonics.",
    "Write about the history of standardized timekeeping.",
    "Describe how a refrigerator moves heat.",
]
CODE = [
    "Write a Python function to merge two sorted lists. Include type hints and a docstring.",
    "Implement binary search in Python with full error handling.",
    "Write a Python class for a thread-safe LRU cache.",
    "Implement quicksort in Python with comments explaining each step.",
    "Write a Python decorator that retries a function on exception with backoff.",
    "Implement a linked list in Python with insert, delete, and reverse methods.",
    "Write a Python function to parse a CSV file into a list of dicts, handling quoted fields.",
    "Implement Dijkstra's shortest-path algorithm in Python using a heap.",
]
