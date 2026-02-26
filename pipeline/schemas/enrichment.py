"""
pipeline/schemas/enrichment.py

Pydantic schema for structured LLM output via instructor.
All Literal types match FEATURES.md allowed enums + extended values.
instructor guarantees the LLM always returns valid, schema-conformant data.
"""

from typing import Literal

from pydantic import BaseModel, Field

# ── Allowed Enum Types ─────────────────────────────────────────────────────────

UseCase = Literal[
    # Conversation & Assistance
    "Chat Assistant",
    "Role Play",
    "Creative Writing",
    # Coding
    "Code Generation",
    "Code Review",
    "Code Explanation",
    # Information & Reasoning
    "Question Answering",
    "Reasoning",
    "Math",
    "Text Summarization",
    "Translation",
    # Retrieval & Embeddings
    "RAG / Retrieval",
    "Text Embedding",
    # Multimodal
    "Image Understanding",
    "Video Understanding",
    "Audio Processing",
    # Agentic
    "Function Calling",
    "Agent / Automation",
    # Data & Analysis
    "Data Analysis",
    "Document Processing",
]

Domain = Literal[
    "General",
    "Code",
    "Vision",
    "Multimodal",     # vision + text + audio
    "Embedding",
    "Reasoning",
    "Math",
    "Medical",
    "Science",
    "Language",       # translation / multilingual
    "Audio",
    "Finance",
    "Legal",
    "Education",
]

Language = Literal[
    # Wide coverage
    "English",
    "Multilingual",
    # Asian
    "Chinese",
    "Japanese",
    "Korean",
    "Vietnamese",
    "Thai",
    "Indonesian",
    "Hindi",
    # Middle East & Central Asia
    "Arabic",
    "Persian",
    "Turkish",
    "Urdu",
    # European
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Italian",
    "Russian",
    "Polish",
    "Dutch",
    "Swedish",
    "Romanian",
    "Czech",
    "Ukrainian",
]

Complexity = Literal["beginner", "intermediate", "advanced"]

Audience = Literal[
    "Developers",
    "Beginners",
    "Researchers",
    "Data Scientists",
    "DevOps",
    "Students",
    "Content Creators",
    "Educators",
    "Medical Professionals",
    "Legal Professionals",
    "Business Analysts",
]

ModelFamily = Literal[
    # Meta Llama lineage
    "Llama",
    # Mistral / Mixtral
    "Mistral",
    # Google
    "Gemma",
    # Microsoft
    "Phi",
    # Qwen (Alibaba)
    "Qwen",
    # DeepSeek
    "DeepSeek",
    # Yi (01.AI)
    "Yi",
    # Command (Cohere)
    "Command",
    # Falcon (TII)
    "Falcon",
    # Granite (IBM)
    "Granite",
    # Orca / Dolphin community  fine-tunes
    "Orca",
    "Dolphin",
    # BLOOM / multilingual
    "BLOOM",
    # Embedding / BERT-style
    "BERT",
    "Nomic",
    "mxbai",
    # Vision/Multimodal
    "LLaVA",
    "Moondream",
    "BakLLaVA",
    # Other notable
    "Vicuna",
    "WizardLM",
    "CodeLlama",
    "StarCoder",
    "SQLCoder",
    "OpenHermes",
    "Other",
]


# ── Enrichment Output Schema ───────────────────────────────────────────────────

class EnrichmentOutput(BaseModel):
    """
    Structured output schema sent to instructor + Groq.
    instructor enforces this schema — LLM retries automatically if output is invalid.
    """

    # ── Core Classification ────────────────────────────────────────────────────
    use_cases: list[UseCase] = Field(
        min_length=1,
        max_length=5,
        description="Primary use cases for this model (1-5 items).",
    )
    domain: Domain = Field(
        description="Primary domain this model specialises in.",
    )
    languages: list[Language] = Field(
        min_length=1,
        max_length=8,
        description="Languages the model supports (1-8 items). Include 'Multilingual' if it supports many.",
    )
    complexity: Complexity = Field(
        description="beginner=<8GB RAM required, intermediate=8-32GB, advanced=>32GB.",
    )

    # ── Model Identity ─────────────────────────────────────────────────────────
    model_family: ModelFamily = Field(
        description=(
            "Model architecture family (e.g. Llama, Mistral, Qwen, Phi, Gemma). "
            "Infer from the model name, description, and README. Use 'Other' if unknown."
        ),
    )
    base_model: str | None = Field(
        default=None,
        description="Base model identifier if this is a fine-tune (e.g. 'llama3.1'). null if base model.",
    )
    is_fine_tuned: bool = Field(
        description="True if fine-tuned or distilled from a base model. False if it IS the base model.",
    )
    is_uncensored: bool = Field(
        description=(
            "True if explicitly uncensored or abliterated. "
            "Check name, description, README for: 'uncensored', 'abliterated', 'no restrictions', 'DAN'. "
            "False otherwise."
        ),
    )

    # ── Description ────────────────────────────────────────────────────────────
    best_for: str = Field(
        min_length=10,
        max_length=300,
        description="Single concise sentence describing the ideal user or task for this model.",
    )
    license: str | None = Field(
        default=None,
        description=(
            "License identifier. Common values: 'MIT', 'Apache 2.0', 'Llama 3 Community License', "
            "'Llama 2 Community License', 'Gemma Terms of Use', 'DeepSeek License', "
            "'Qwen License', 'CC BY 4.0', 'Proprietary'. null ONLY if truly not found."
        ),
    )

    # ── Quality ────────────────────────────────────────────────────────────────
    strengths: list[str] = Field(
        min_length=1,
        max_length=5,
        description="2-5 short strings highlighting what this model excels at.",
    )
    limitations: list[str] = Field(
        min_length=1,
        max_length=5,
        description="1-5 short strings about known weaknesses or constraints.",
    )
    target_audience: list[Audience] = Field(
        min_length=1,
        max_length=4,
        description="Target audience groups (1-4 items).",
    )
