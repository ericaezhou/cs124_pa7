"""
Extra Credit 

In this assignment, you will create your own AI agent using DSPy and any tools/APIs you choose.

Your agent should:
1. Use DSPy's framework (signatures, modules, etc.)
2. Implement some interesting functionality beyond basic chat
3. Demonstrate your implementation with a working example

You have complete freedom to:
- Choose what your agent does (data analysis, creative generation, etc.)
- Select which APIs or libraries to integrate
- Design the interface and interaction pattern

Some ideas to get you started:
- A research assistant that uses web search
- A creative writing collaborator with style adaptation
- A data analysis agent that processes files
- A multi-step reasoning agent for complex problems
- Something completely different! Be creative!

Complete the TODOs below to build your agent.
"""

import dspy
import os
from api_keys import TOGETHER_API_KEY

# Configure environment
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

import numpy as np
import difflib
from agent import similarity, titles, ratings_matrix, MemoryTools, memory_config
from mem0 import Memory

def extract_titles_from_text(text: str):
    text_lower = text.lower()
    found = []

    for title in titles:
        title_lower = title.lower()

        if title_lower in text_lower:
            found.append(title)
            continue

        simple = title_lower
        if "(" in simple:
            simple = simple[:simple.rfind("(")].strip()

        if simple and simple in text_lower:
            found.append(title)
            continue

        if "," in simple:
            parts = [p.strip() for p in simple.split(",")]
            if len(parts) == 2 and parts[1] in {"the", "a", "an"}:
                reordered = f"{parts[1]} {parts[0]}"
                if reordered in text_lower:
                    found.append(title)
                    continue

    deduped = []
    seen = set()
    for t in found:
        if t not in seen:
            deduped.append(t)
            seen.add(t)

    return deduped


def recommend_from_favorites(liked_movies_str: str, k: int = 3):
    liked_movies = [m.strip() for m in liked_movies_str.split("||") if m.strip()]

    resolved_movies = []

    for chunk in liked_movies:
        resolved = resolve_movie_title(chunk)
        if resolved:
            resolved_movies.append(resolved)
            continue

        extracted = extract_titles_from_text(chunk)
        resolved_movies.extend(extracted)

    #Remove duplicates
    resolved_movies = list(dict.fromkeys(resolved_movies))

    if len(resolved_movies) == 0:
        return "I couldn't identify any valid movies from your past preferences."

    user_ratings = np.zeros(len(titles))
    for movie in resolved_movies:
        idx = titles.index(movie)
        user_ratings[idx] = 1.0

    rated_movies = np.where(user_ratings != 0)[0]
    unrated_movies = np.where(user_ratings == 0)[0]

    scores = []
    for movie_idx in unrated_movies:
        score = 0.0
        for rated_idx in rated_movies:
            score += similarity(ratings_matrix[movie_idx], ratings_matrix[rated_idx])
        scores.append((movie_idx, score))

    scores.sort(key=lambda x: (-x[1], x[0]))
    recommendations = [titles[idx] for idx, _ in scores[:k]]

    return {
        "based_on": resolved_movies,
        "recommendations": recommendations
    }

class ExtraCreditSignature(dspy.Signature):
    """
    You are a personalized movie recommendation assistant.

    Rules:
    - If the user says they liked or loved a movie, store that preference in memory.
    - If the user asks for recommendations based on what they liked before:
      1. search memory for previously liked movies
      2. pass the relevant memory text into recommend_from_favorites
      3. return a short friendly recommendation
    - If the user asks what movies you remember they liked, use search_memories or get_all_memories.
    - Do not use general chat if the answer should come from memory.
    - Prefer using user_id="default_user" consistently.

    Examples:
    - "My name is Peter and I loved The Matrix" -> store memory
    - "Recommend movies based on what I liked before" -> search memory, then recommend_from_favorites
    """
    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(desc="Final personalized recommendation.")


class ExtraCreditAgent(dspy.Module):
    def __init__(self):
        super().__init__()

        # Initialize memory system
        self.memory = Memory.from_config(memory_config)
        self.memory_tools = MemoryTools(self.memory)

        self.tools = [
            recommend_from_favorites,
            self.memory_tools.store_memory,
            self.memory_tools.search_memories,
            self.memory_tools.get_all_memories
        ]

        self.react = dspy.ReAct(
            ExtraCreditSignature,
            tools=self.tools,
            max_iters=6
        )

    def forward(self, user_request: str):
        return self.react(user_request=user_request)



def run_demo():
    # Configure DSPy
    lm = dspy.LM(model='together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1')
    dspy.configure(lm=lm)

    agent = ExtraCreditAgent()

    print("🤖 Agent Demo")
    print("=" * 50)

    test_inputs = [
        "My name is Peter and I loved The Matrix",
        "What movies do you remember I liked? My name is Peter",
        "Recommend 3 movies for Peter based on what he liked before"
    ]

    for user_input in test_inputs:
        print(f"\n📝 User: {user_input}")
        response = agent(user_request=user_input)
        print(f"🤖 Agent: {response.process_result}")


if __name__ == "__main__":
    run_demo()
