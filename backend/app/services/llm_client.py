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


# Last extraction path taken by extract_intent (for evaluation auditing).
# One of: "llm" | "llm_unparseable" | "llm_error_fallback" | "fallback".
# Read via last_extraction_source(). Never raises.
_last_source: str = "fallback"
# Wall-clock ms spent inside the most recent generate_content call (intent).
# 0.0 when the last extraction did not reach the API. Read via
# last_call_latency_ms(). Lets evaluators separate inference latency from
# quota-pacing sleeps.
_last_latency_ms: float = 0.0


def last_extraction_source() -> str:
    """Return how the most recent extract_intent call produced its delta."""
    return _last_source


def last_call_latency_ms() -> float:
    """Return API time (ms) of the most recent intent call, 0.0 if none."""
    return _last_latency_ms


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

    # 1b. Ordinal exclusion follow-ups ("exclude the second movie", "skip the 3rd one").
    # Checked early so a follow-up is never misread as a title search. The mode
    # is left unset so _merge_intents preserves the session's request_mode.
    _ordinals = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }
    _ord_match = re.search(
        r"(?:exclude|remove|skip|hide|drop|don'?t show|not)\s+(?:the\s+)?"
        r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)?)"
        r"\s+(?:movie|film|one|result|option|pick|choice|recommendation)",
        msg_lower,
    )
    if _ord_match:
        _raw_ord = _ord_match.group(1)
        _num = _ordinals.get(_raw_ord)
        if _num is None:
            _num = int(re.sub(r"(st|nd|rd|th)$", "", _raw_ord))
        return IntentDelta(target_ordinal_exclusion=_num)

    # 2. Similar movies check: "movies like X", "similar to X", "like X",
    #    "movies similar to X", "films similar to X"
    similar_match = re.search(
        r"^(?:show\s+(?:me\s+)?)?(?:movies?\s+like|films?\s+like|movies?\s+similar\s+to|films?\s+similar\s+to|similar\s+to|like)\s+(.+)$",
        msg_lower,
    )
    if similar_match:
        target_ref = similar_match.group(1).strip()
        return IntentDelta(
            request_mode="similar_movies",
            reference_titles=[target_ref],
            semantic_query=target_ref,
        )

    # 3. Franchise / Catalog lookup: e.g. "avengers all movie", "all avengers movies", "the avengers 1998"
    # A leading request verb ("recommend action movies") signals discovery, not a
    # franchise name — skip the suffix heuristic then and let genre parsing handle it.
    _looks_like_request = bool(
        re.match(
            r"^(?:recommend|show(?:\s+me)?|give(?:\s+me)?|find|suggest|please|get\s+me|i\s+want|i\s+would\s+like|can\s+you|could\s+you)\b",
            msg_lower,
        )
    )
    suffix_match = None
    if not _looks_like_request:
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

    # 3b. Year-range expressions ("after 2000", "before 1990", "from the 90s").
    # These describe discovery filters — not an exact-title year — so they route
    # to personalized_recommendation and skip the exact-year title branch below.
    _range_min: Optional[int] = None
    _range_max: Optional[int] = None
    _after_m = re.search(r"(?:after|post|since|newer\s+than)\s+(19\d\d|20\d\d)\b", msg_lower)
    if _after_m:
        _range_min = int(_after_m.group(1))
    _before_m = re.search(r"(?:before|pre|older\s+than|up\s+to)\s+(19\d\d|20\d\d)\b", msg_lower)
    if _before_m:
        _range_max = int(_before_m.group(1))
    _dec_m = re.search(r"\b(?:from|of|in|during)?\s*(?:the\s+)?((?:19|20)?\d0)s\b", msg_lower)
    if _dec_m and _range_min is None and _range_max is None:
        _dec_raw = _dec_m.group(1)
        _dec_full = int(_dec_raw) if len(_dec_raw) == 4 else int(f"19{_dec_raw}")
        # Two-digit decades ("90s") refer to the 1900s; "2000s" is explicit.
        if len(_dec_raw) == 2 and _dec_raw != "00":
            _dec_full = 1900 + int(_dec_raw)
        elif _dec_raw == "00":
            _dec_full = 2000
        _range_min, _range_max = _dec_full, _dec_full + 9

    year_match = re.search(r"^(.+?)\s+\b(19\d\d|20\d\d)\b$", msg_lower)
    if year_match and _range_min is None and _range_max is None:
        title_candidate = year_match.group(1).strip()
        yr = int(year_match.group(2))
        return IntentDelta(
            request_mode="catalog_lookup",
            reference_titles=[title_candidate],
            semantic_query=title_candidate,
            min_year=yr,
            max_year=yr,
        )

    # 4. Genre extraction (also picks up explicit runtime caps like "under 90 minutes"
    #    and year ranges like "after 2000" / "from the 90s")
    # Canonical genre names always count. Looser colloquial aliases ("funny",
    # "animated", "scary"...) only count alongside other discovery signals, so
    # bare short titles ("Funny Games", "Scary Movie") still route to catalog lookup.
    _genre_aliases: dict[str, list[str]] = {
        "Action": ["action"],
        "Adventure": ["adventure"],
        "Animation": ["animation", "animated", "anime", "cartoon"],
        "Comedy": ["comedy", "comedies", "funny", "humor", "humour"],
        "Crime": ["crime"],
        "Documentary": ["documentary", "documentaries", "docuseries"],
        "Drama": ["drama"],
        "Fantasy": ["fantasy"],
        "Horror": ["horror", "scary", "creepy", "horror movie"],
        "Mystery": ["mystery"],
        "Romance": ["romance", "romantic"],
        "Sci-Fi": ["sci-fi", "scifi", "sci fi", "science fiction"],
        "Thriller": ["thriller"],
        "War": ["war"],
        "Western": ["western"],
    }
    _words = re.findall(r"[a-z]+", msg_lower)
    _has_discovery = (
        _looks_like_request
        or len(_words) > 4
        or bool(
            re.search(
                r"\b(without|excluding|except|under|over|rated|rating|after|before|since|minutes?|hours?|decade|\d0s)\b",
                msg_lower,
            )
        )
    )
    preferred = []
    for g, aliases in _genre_aliases.items():
        strict_hit = re.search(rf"\b{re.escape(aliases[0])}\b", msg_lower)
        loose_hit = any(
            re.search(rf"\b{re.escape(a)}\b", msg_lower) for a in aliases[1:]
        )
        if strict_hit or (loose_hit and _has_discovery):
            preferred.append(g)
    runtime_cap: Optional[int] = None
    _rt_match = re.search(
        r"(?:under|less than|below|max|within|up to|shorter than)\s+(\d+)\s*(hours?|hrs?|minutes?|mins?)",
        msg_lower,
    )
    if _rt_match:
        _amount = int(_rt_match.group(1))
        _unit = _rt_match.group(2)
        runtime_cap = _amount * 60 if _unit.startswith("hour") or _unit.startswith("hr") else _amount
    if preferred or runtime_cap is not None or _range_min is not None or _range_max is not None:
        _set_fields: dict = {}
        if runtime_cap is not None:
            _set_fields["max_runtime_minutes"] = runtime_cap
        if _range_min is not None:
            _set_fields["min_year"] = _range_min
        if _range_max is not None:
            _set_fields["max_year"] = _range_max
        return IntentDelta(
            request_mode="personalized_recommendation",
            preferred_genres=preferred,
            semantic_query=msg_trimmed,
            max_runtime_minutes=runtime_cap,
            min_year=_range_min,
            max_year=_range_max,
            set_fields=_set_fields,
        )

    # 5. Vague / content-free requests ("suggest something", "anything good?").
    # The message carries no title, genre, year, or reference — ask for clarification
    # instead of misreading filler as a catalog title.
    _request_verbs = {
        "suggest", "show", "find", "give", "get", "recommend", "recommendation",
        "recommendations", "watch", "see", "explore", "browse", "play",
        "please", "me", "something", "anything", "whatever", "stuff",
        "good", "best", "nice", "great", "cool", "fun", "interesting",
        "movie", "movies", "film", "films", "a", "an", "the", "some", "to",
        "for", "me", "i", "want", "like", "need", "tonight", "now",
    }
    _tokens = re.findall(r"[a-z]+", msg_lower)
    if _tokens and all(t in _request_verbs for t in _tokens):
        return IntentDelta(
            request_mode="catalog_lookup",
            needs_clarification=True,
            clarification_question="What kind of movies are you looking for? You can ask for a specific genre, decade, or franchise like 'Avengers'.",
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
    The path taken is recorded in last_extraction_source().
    """
    global _last_source
    global _last_latency_ms
    _last_latency_ms = 0.0
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
        import time as _time

        _t0 = _time.monotonic()
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
        _last_latency_ms = (_time.monotonic() - _t0) * 1000.0

        raw_text = response.text.strip()
        logger.debug("LLM intent response: %s", raw_text[:500])

        # Parse and validate through Pydantic
        data = json.loads(raw_text)
        delta = IntentDelta.model_validate(data)
        _last_source = "llm"
        return delta

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM intent JSON: %s", e)
        _last_source = "llm_unparseable"
        return IntentDelta(
            semantic_query=user_message,
            needs_clarification=True,
            clarification_question="I had trouble understanding that. Could you describe what kind of movie you're looking for?",
        )
    except Exception as e:
        logger.error("LLM intent extraction failed: %s", e, exc_info=True)
        # Graceful fallback to heuristic intent parsing
        _last_source = "llm_error_fallback"
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

