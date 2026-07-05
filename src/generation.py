from __future__ import annotations
import re
from typing import List, Dict


GENERATION_INSTRUCTIONS = """Answer the question using only the information in the passages below. \
Write in your own words — do not copy passages verbatim, and do not include passage numbers, filenames, or labels in your answer. \
If the answer naturally has multiple steps, components, or items, present them as a numbered list with a short bold-style label for each item followed by a colon and a one-sentence explanation. \
Otherwise, write 1-3 direct sentences. \
If the passages don't contain the answer, respond exactly: "I don't have enough information in the provided documents to answer that."

Passages:
{context}

Question: {question}
Answer:"""

SUMMARY_INSTRUCTIONS = """Using only the information in the passages below, write a clear, well-organized \
summary of the document in your own words. Do not copy passages verbatim, and do not include passage \
numbers, filenames, or labels in your answer. \
Structure the summary as multiple short paragraphs: start with a 1-2 sentence overview of what the \
document covers, then use following paragraphs to cover the main topics, categories, or steps in more \
detail. If the passages don't contain enough information to summarize, respond exactly: \
"I don't have enough information in the provided documents to answer that."

Passages:
{context}

Question: {question}
Summary:"""


def build_generation_prompt(question: str, retrieved_chunks: List[Dict], is_summary: bool = False) -> str:
    context = "\n\n".join(
        f"Passage {i + 1}: {c['text']}" for i, c in enumerate(retrieved_chunks)
    )
    template = SUMMARY_INSTRUCTIONS if is_summary else GENERATION_INSTRUCTIONS
    return template.format(context=context, question=question)


def build_citation_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    context = "\n\n".join(
        f"[{c['source']} - chunk {c['chunk_id']}]\n{c['text']}" for c in retrieved_chunks
    )
    return GENERATION_INSTRUCTIONS.format(context=context, question=question)

build_prompt = build_citation_prompt


_LEAK_PATTERNS = [
    re.compile(r"\[.*?-\s*chunk\s*\d+\]", re.IGNORECASE),
    re.compile(r"Passage\s*\d+\s*:", re.IGNORECASE),
]


def _clean_generated_text(text: str) -> str:
    for pattern in _LEAK_PATTERNS:
        text = pattern.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class FlanT5Generator:
    def __init__(self, model_name: str = "google/flan-t5-base", max_input_tokens: int = 480):
        from transformers import pipeline  
        self.model_name = model_name
        self.pipe = pipeline("text2text-generation", model=model_name)
        self.tokenizer = self.pipe.tokenizer
        self.max_input_tokens = max_input_tokens

    def _fit_context_to_budget(self, question: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        chunks = list(retrieved_chunks)
        while chunks:
            prompt = build_generation_prompt(question, chunks)
            n_tokens = len(self.tokenizer.encode(prompt))
            if n_tokens <= self.max_input_tokens:
                return chunks
            chunks.pop()  
        return chunks

    def generate(self, question: str, retrieved_chunks: List[Dict], is_summary: bool = False) -> str:
        if not retrieved_chunks:
            return "I don't have enough information in the provided documents to answer that."

        fitted_chunks = self._fit_context_to_budget(question, retrieved_chunks)
        prompt = build_generation_prompt(question, fitted_chunks, is_summary=is_summary)
        max_new_tokens = 350 if is_summary else 200
        output = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        raw_answer = output[0]["generated_text"].strip()
        return _clean_generated_text(raw_answer)


class ExtractiveGenerator:
    model_name = "extractive-keyword-overlap (offline fallback, no real LLM)"

    def generate(self, question: str, retrieved_chunks: List[Dict], is_summary: bool = False) -> str:
        if not retrieved_chunks:
            return "I don't have enough information in the provided documents to answer that."

        question_words = set(question.lower().split())
        best_sentence = ""
        best_score = -1
        for chunk in retrieved_chunks:
            for sentence in chunk["text"].split(". "):
                overlap = len(question_words & set(sentence.lower().split()))
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence.strip()

        return best_sentence or retrieved_chunks[0]["text"][:300]


class CohereGenerator:

    def __init__(self, api_key: str | None = None, model_name: str = "command-r-08-2024"):
        import os
        import cohere 

        resolved_key = api_key or os.environ.get("COHERE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Cohere API key found. Pass api_key=... to CohereGenerator/get_generator, "
                "or set the COHERE_API_KEY environment variable."
            )
        self.client = cohere.ClientV2(api_key=resolved_key)
        self.model_name = model_name

    def generate(self, question: str, retrieved_chunks: List[Dict], is_summary: bool = False) -> str:
        if not retrieved_chunks:
            return "I don't have enough information in the provided documents to answer that."

        prompt = build_generation_prompt(question, retrieved_chunks, is_summary=is_summary)
        max_tokens = 700 if is_summary else 300

        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        raw_answer = response.message.content[0].text.strip()
        return _clean_generated_text(raw_answer)


def get_generator(
    offline_mode: bool = False,
    model_name: str = "google/flan-t5-base",
    backend: str = "local",
    cohere_api_key: str | None = None,
):
    if offline_mode:
        return ExtractiveGenerator()
    if backend == "cohere":
        cohere_model = model_name if "command" in model_name else "command-r-08-2024"
        return CohereGenerator(api_key=cohere_api_key, model_name=cohere_model)
    return FlanT5Generator(model_name=model_name)