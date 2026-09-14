"""
LLM client for the Ask Filmory assistant.

Wraps Google Gemini for two tasks:
1. Intent extraction — structured output from user's natural language.
2. Explanation generation — grounded response using supplied evidence only.

Handles timeouts, validation failures, and fallback to template responses.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from google import genai
from google.genai import types as genai_types

from app.config import settings
from app.schemas.assistant import (
    ExplanationItem,
    IntentDelta,
    MovieEvidence,
    MovieIntent,
    StructuredExplanation,
)

logger = logging.getLogger("filmory.llm")

# ---------------------------------------------------------------------------
# Lazy client singleton
# ---------------------------------------------------------------------------
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to backend/.env to use the assistant."
            )
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
INTENT_SYSTEM_PROMPT = """\
You are the intent parser for Filmory, a movie recommendation assistant.
Given a user message (and optionally a conversation history), extract their
movie preferences into a structured JSON update delta.

Rules:
- Classify the user's intent into one of three request_mode values:
  1. "catalog_lookup": The user wants to see, find, or list specific titles, a movie franchise, or a film collection (e.g., "Avengers all movies", "The Avengers 1998", "list Batman movies", "Toy Story 2").
     * Extract the clean franchise or movie title into reference_titles AND semantic_query.
     * Do NOT invent or add unrelated genres.
  2. "similar_movies": The user asks for movies LIKE, similar to, or inspired by a reference movie (e.g., "movies like Avengers", "movies similar to Inception").
     * Put the reference movie title(s) in reference_titles.
  3. "personalized_recommendation": The user asks for discovery, mood, or constraint-based recommendations (e.g., "recommend a comedy", "good sci-fi under 2 hours without horror").
- Extract ONLY what the user explicitly states or clearly implies.
- Do NOT invent preferences the user did not express.
- If the user asks for "all movies" or "show movies" without any title, franchise, or genre,
  set needs_clarification=true and write a clarification_question asking what kind of movies or franchise they would like to explore.
- For genres, use standard movie genres: Action, Adventure, Animation, Comedy,
  Crime, Documentary, Drama, Fantasy, Horror, Mystery, Romance, Sci-Fi,
  Thriller, War, Western, Musical, Film-Noir, Children, IMAX.
- If the user references a specific movie by name, put it in reference_titles.
- If the request is too vague to produce useful results (e.g., "suggest something"),
  set needs_clarification=true and write a brief clarification_question.
- For follow-up messages:
  * If the user wants to remove/clear a constraint (e.g. "forget the runtime limit", "remove horror filter", "clear all restrictions"),
    add the field name to clear_fields: ["max_runtime_minutes", "excluded_genres", "preferred_genres", "min_year", "max_year", "min_rating", "mood", "all"].
  * If the user refers to an ordinal item from the previous response to exclude (e.g. "exclude the first movie", "don't show the second one", "skip the 3rd"),
    set target_ordinal_exclusion to the 1-indexed integer (1, 2, 3, etc).
  * If the user updates or tightens a restriction (e.g. "make it shorter", "under 90 mins"), set it in set_fields AND in the corresponding schema field.
- max_runtime_minutes should be in minutes (convert from hours if needed).
- rating is on a 0-10 scale.

Respond with ONLY a valid JSON object matching this schema:
{
  "request_mode": "catalog_lookup" | "personalized_recommendation" | "similar_movies",
  "set_fields": {},
  "clear_fields": [],
  "add_excluded_movie_ids": [],
  "target_ordinal_exclusion": null,
  "semantic_query": "",
  "preferred_genres": [],
  "excluded_genres": [],
  "min_year": null,
  "max_year": null,
  "max_runtime_minutes": null,
  "min_rating": null,
  "reference_titles": [],
  "mood": "",
  "needs_clarification": false,
  "clarification_question": ""
}
"""

EXPLANATION_SYSTEM_PROMPT = """\
You are the explanation writer for Filmory, a movie recommendation assistant.
You receive a list of recommended movies with evidence IDs and facts about WHY each was chosen.

Rules:
- Respond in strictly valid JSON matching this schema:
{
  "summary": "Conversational 1-2 sentence overview of why these picks suit the user request.",
  "items": [
    {
      "movie_id": 123,
      "explanation": "Clear factual explanation of why this movie matches the user's constraints and taste.",
      "evidence_ids": ["movie:123:genres", "movie:123:runtime"]
    }
  ]
}
- In "items", include an explanation for each recommended movie.
- For "evidence_ids", ONLY use evidence keys provided in the evidence package for that specific movie.
- Do NOT invent facts, directors, awards, or details not present in the evidence.
- Do NOT mention raw math algorithms or numerical scores in user-facing text.
"""


# ---------------------------------------------------------------------------
# Intent extraction
# ---------------------------------------------------------------------------
def fallback_extract_intent(user_message: str) -> IntentDelta:
    """
    Heuristic intent extractor used when Gemini is unavailable or rate limited.
    Accurately parses:
    - Bare queries like 'all movies' -> needs_clarification
    - Similar movies like 'movies like Avengers' -> request_mode='similar_movies', reference_titles=['Avengers']
    - Franchise/title queries like 'Avengers all movie', 'The Avengers 1998' -> request_mode='catalog_lookup', reference_titles=['Avengers']
    - Genre queries -> request_mode='personalized_recommendation'
    """
    msg_trimmed = user_message.strip()
    msg_lower = msg_trimmed.lower()

    # 1. Bare "all movies" / "show movies" without criteria
    bare_queries = {"all movie", "all movies", "show all movies", "show movies", "list movies", "movies", "films"}
    if msg_lower in bare_queries:
        return IntentDelta(
            request_mode="catalog_lookup",
            needs_clarification=True,
            clarification_question="What kind of movies are you looking for? You can ask for a specific genre, decade, or franchise like 'Avengers'.",
        )

    # 2. Similar movies check: "movies like X", "similar to X", "like X"
    similar_match = re.search(r"^(?:show\s+(?:me\s+)?)?(?:movies?\s+like|films?\s+like|similar\s+to|like)\s+(.+)$", msg_lower)
    if similar_match:
        target_ref = similar_match.group(1).strip()
        return IntentDelta(
            request_mode="similar_movies",
            reference_titles=[target_ref],
            semantic_query=target_ref,
        )

    # 3. Franchise / Catalog lookup: e.g. "avengers all movie", "all avengers movies", "the avengers 1998"
    suffix_match = re.search(r"^(.+?)\s+(?:all\s+movies?|all\s+films?|movies?|films?|series|franchise)$", msg_lower)
    if suffix_match:
        title_candidate = suffix_match.group(1).strip()
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", title_candidate)
        extracted_year = int(year_match.group(1)) if year_match else None
        cleaned_title = re.sub(r"\b(19\d\d|20\d\d)\b", "", title_candidate).strip() if year_match else title_candidate
        return IntentDelta(
            request_mode="catalog_lookup",
            reference_titles=[cleaned_title],
            semantic_query=cleaned_title,
            min_year=extracted_year,
            max_year=extracted_year,
        )

    prefix_match = re.search(r"^(?:all\s+(?:the\s+)?|list\s+all\s+(?:the\s+)?|show\s+all\s+(?:the\s+)?)(.+?)(?:\s+movies?|\s+films?)?$", msg_lower)
    if prefix_match:
        title_candidate = prefix_match.group(1).strip()
        title_candidate = re.sub(r"\s+(?:movies?|films?|series|franchise)$", "", title_candidate).strip()
        if title_candidate and title_candidate not in bare_queries:
            return IntentDelta(
                request_mode="catalog_lookup",
                reference_titles=[title_candidate],
                semantic_query=title_candidate,
            )

    year_match = re.search(r"^(.+?)\s+\b(19\d\d|20\d\d)\b$", msg_lower)
    if year_match:
        title_candidate = year_match.group(1).strip()
        yr = int(year_match.group(2))
        return IntentDelta(
            request_mode="catalog_lookup",
            reference_titles=[title_candidate],
            semantic_query=title_candidate,
            min_year=yr,
            max_year=yr,
        )

    # 4. Genre extraction
    known_genres = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
        "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western",
    ]
    preferred = [g for g in known_genres if re.search(rf"\b{g.lower()}\b", msg_lower)]
    if preferred:
        return IntentDelta(
            request_mode="personalized_recommendation",
            preferred_genres=preferred,
            semantic_query=msg_trimmed,
        )

    return IntentDelta(
        request_mode="catalog_lookup",
        semantic_query=msg_trimmed,
        reference_titles=[msg_trimmed],
    )


def extract_intent(
    user_message: str,
    conversation_history: list[dict] | None = None,
) -> IntentDelta:
    """
    Call Gemini to parse a user's natural-language movie request into
    a structured IntentDelta. Falls back to heuristic IntentDelta on failure.
    """
    client = _get_client()

    messages: list[genai_types.Content] = []

    # Add conversation history for context (if follow-up)
    if conversation_history:
        for msg in conversation_history[-settings.ASSISTANT_MAX_HISTORY :]:
            role = "user" if msg["role"] == "user" else "model"
            messages.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=msg["content"])],
                )
            )

    # Current user message
    messages.append(
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        )
    )

    try:
        response = client.models.generate_content(
            model=settings.GEMINI_CHAT_MODEL,
            contents=messages,
            config=genai_types.GenerateContentConfig(
                system_instruction=INTENT_SYSTEM_PROMPT,
                temperature=0.1,  # Low temperature for structured extraction
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )

        raw_text = response.text.strip()
        logger.debug("LLM intent response: %s", raw_text[:500])

        # Parse and validate through Pydantic
        data = json.loads(raw_text)
        delta = IntentDelta.model_validate(data)
        return delta

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM intent JSON: %s", e)
        return IntentDelta(
            semantic_query=user_message,
            needs_clarification=True,
            clarification_question="I had trouble understanding that. Could you describe what kind of movie you're looking for?",
        )
    except Exception as e:
        logger.error("LLM intent extraction failed: %s", e, exc_info=True)
        # Graceful fallback to heuristic intent parsing
        return fallback_extract_intent(user_message)


# ---------------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------------
def generate_explanation(
    evidence_list: list[MovieEvidence],
    user_request: str,
    conversation_context: str = "",
) -> tuple[str, StructuredExplanation]:
    """
    Generate a grounded natural-language response explaining why these
    movies were recommended. Uses ONLY the evidence provided and produces
    auditable StructuredExplanation linked to verifiable evidence_ids.

    Falls back to a deterministic template response if LLM fails or hallucinates.
    """
    if not evidence_list:
        summary = "I couldn't find movies matching those specific criteria. Try broadening your genres or runtime constraints."
        return summary, StructuredExplanation(summary=summary, items=[])

    valid_movie_ids = {ev.movie_id for ev in evidence_list}
    valid_evidence_keys: dict[int, set[str]] = {
        ev.movie_id: set(ev.evidence_keys) for ev in evidence_list
    }

    # Build evidence context for the LLM
    evidence_text = "RECOMMENDED MOVIES WITH EVIDENCE:\n\n"
    for i, ev in enumerate(evidence_list, 1):
        evidence_text += f"Movie {i} [ID: {ev.movie_id}]: {ev.title} ({ev.year or 'Unknown'})\n"
        evidence_text += f"  Genres: {', '.join(ev.genres)}\n"
        evidence_text += f"  Runtime: {ev.runtime or 'Unknown'} minutes\n"
        evidence_text += f"  Rating: {ev.rating or 'Unrated'}/10\n"
        if ev.description_snippet:
            evidence_text += f"  Synopsis: {ev.description_snippet}\n"
        if ev.matched_constraints:
            evidence_text += f"  Matches user request: {', '.join(ev.matched_constraints)}\n"
        if ev.personalization_signals:
            evidence_text += f"  Personalization: {', '.join(ev.personalization_signals)}\n"
        evidence_text += f"  Available Evidence IDs: {', '.join(ev.evidence_keys)}\n\n"

    prompt = f"User's request: {user_request}\n\n{evidence_text}"
    if conversation_context:
        prompt = f"Conversation context: {conversation_context}\n\n{prompt}"

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=settings.GEMINI_CHAT_MODEL,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)],
                )
            ],
            config=genai_types.GenerateContentConfig(
                system_instruction=EXPLANATION_SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text.strip()
        data = json.loads(raw_text)
        candidate_explanation = StructuredExplanation.model_validate(data)

        # Audit validation: verify all movie_ids and evidence_ids belong to the evidence package
        validated_items = []
        for item in candidate_explanation.items:
            if item.movie_id not in valid_movie_ids:
                logger.warning("LLM hallucinated movie_id %s not in evidence package", item.movie_id)
                continue
            # Filter evidence_ids to strictly known keys
            allowed_keys = valid_evidence_keys.get(item.movie_id, set())
            item.evidence_ids = [eid for eid in item.evidence_ids if eid in allowed_keys]
            validated_items.append(item)

        if validated_items:
            candidate_explanation.items = validated_items
            summary = candidate_explanation.summary or "Here are recommendations matching your preferences:"
            return summary, candidate_explanation

    except Exception as e:
        logger.error("LLM explanation generation failed or invalid: %s", e, exc_info=True)

    # Template fallback: 100% deterministic and factually grounded
    items = []
    for ev in evidence_list[:5]:
        matched_str = ", ".join(ev.matched_constraints) if ev.matched_constraints else "your preferences"
        exp_text = f"{ev.title} fits {matched_str}."
        items.append(
            ExplanationItem(
                movie_id=ev.movie_id,
                explanation=exp_text,
                evidence_ids=[k for k in ev.evidence_keys if "constraint" in k or "genre" in k],
            )
        )

    summary = f"Based on your request, here are {len(items)} films that match your criteria."
    return summary, StructuredExplanation(summary=summary, items=items)

