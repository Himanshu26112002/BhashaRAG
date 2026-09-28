"""Prompt templates for grounded, bilingual answering."""

from __future__ import annotations

LANGUAGE_INSTRUCTIONS = {
    "hi": "The user wrote in Hindi. Write your entire answer in Hindi (Devanagari script).",
    "hinglish": (
        "The user wrote in Hinglish (Hindi in Roman script). Answer in the same casual Hinglish, "
        "in Roman script."
    ),
    "mixed": "The user mixed Hindi and English. Answer in the same mix they used.",
    "en": "The user wrote in English. Answer in English.",
}

ANSWER_SYSTEM_PROMPT = """You are BhashaRAG, an assistant that answers questions using only the user's own documents.

The documents may be in Hindi, English or both, and the question may be in a different language from the documents. Understand the sources in whatever language they are in, and translate faithfully when needed.

How to answer:
- Base every factual claim on the numbered context passages. After each claim, cite its source in square brackets, e.g. [1] or [2][3].
- If the passages do not contain the answer, say so plainly (in the user's language) and suggest what document might help. Do not fill gaps from general knowledge.
- Keep numbers, dates, names and amounts exactly as they appear in the sources.
- Be concise: lead with the direct answer, then supporting detail. Use short bullet lists when listing steps, criteria or documents.

{language_instruction}"""

CONTEXT_TEMPLATE = """Context passages from the knowledge base:

{passages}

Question: {question}"""

PASSAGE_TEMPLATE = "[{n}] (source: {filename}, page {page})\n{text}"

REWRITE_SYSTEM_PROMPT = """Rewrite the user's latest message as a single standalone search query, resolving pronouns and references using the conversation so far. Keep the user's original language and script. Output only the rewritten query, nothing else."""


def format_passages(hits: list[dict]) -> str:
    return "\n\n".join(
        PASSAGE_TEMPLATE.format(n=i, filename=h["filename"], page=h["page"], text=h["text"])
        for i, h in enumerate(hits, start=1)
    )


def no_llm_answer(hits: list[dict], language: str) -> str:
    if not hits:
        return (
            "मुझे आपके दस्तावेज़ों में इससे संबंधित कुछ नहीं मिला।"
            if language == "hi"
            else "I couldn't find anything relevant in your documents."
        )
    header = (
        "कोई LLM कॉन्फ़िगर नहीं है, इसलिए यहाँ सबसे प्रासंगिक अंश दिए गए हैं:"
        if language == "hi"
        else "No LLM is configured (LLM_PROVIDER=none), so here are the most relevant passages:"
    )
    lines = [header, ""]
    for i, h in enumerate(hits[:3], start=1):
        snippet = h["text"] if len(h["text"]) <= 600 else h["text"][:600].rsplit(" ", 1)[0] + " …"
        lines.append(f"**[{i}] {h['filename']}, p.{h['page']}**\n{snippet}\n")
    return "\n".join(lines)
