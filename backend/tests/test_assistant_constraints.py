import pytest
from app.services.llm_client import fallback_extract_intent

@pytest.mark.parametrize('text,preferred,excluded,rating', [
    ('funny comedy movies without horror under 120 minutes', ['Comedy'], ['Horror'], None),
    ('drama movies excluding romance made after 2000', ['Drama'], ['Romance'], None),
    ('westerns under 130 minutes', ['Western'], [], None),
    ('animated movies rated at least 8 out of 10', ['Animation'], [], 8.0),
    ('documentaries rated at least 7.5 out of 10', ['Documentary'], [], 7.5),
])
def test_explicit_constraints_survive_offline_extraction(text,preferred,excluded,rating):
    intent = fallback_extract_intent(text)
    assert intent.preferred_genres == preferred
    assert intent.excluded_genres == excluded
    assert intent.min_rating == rating

@pytest.mark.parametrize('text', ['comedy movies','all comedy movies','show all comedy movies'])
def test_simple_genre_requests_are_not_catalogue_titles(text):
    intent=fallback_extract_intent(text)
    assert intent.request_mode == 'personalized_recommendation'
    assert intent.preferred_genres == ['Comedy']
    assert not intent.reference_titles

def test_multiple_exclusions_and_fractional_runtime():
    intent=fallback_extract_intent('drama without horror or romance under 1.5 hours')
    assert intent.preferred_genres == ['Drama']
    assert set(intent.excluded_genres) == {'Horror','Romance'}
    assert intent.max_runtime_minutes == 90
def test_offline_followup_can_clear_limits_without_changing_mode():
    from app.services.llm_client import fallback_extract_intent
    result = fallback_extract_intent("forget the runtime limit")
    assert "max_runtime_minutes" in result.clear_fields
    assert result.request_mode is None
    assert fallback_extract_intent("clear all restrictions").clear_fields == ["all"]


def test_local_mode_never_constructs_a_cloud_client_even_with_a_key(monkeypatch):
    from app.config import settings
    from app.services import llm_client
    from app.schemas.assistant import MovieEvidence
    monkeypatch.setattr(settings, 'ASSISTANT_LOCAL_ONLY', True)
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'unused-test-key')
    def forbidden():
        raise AssertionError('A cloud client was constructed in local mode')
    monkeypatch.setattr(llm_client, '_get_client', forbidden)
    intent = llm_client.extract_intent('comedy without horror under 120 minutes')
    assert intent.preferred_genres == ['Comedy'] and intent.excluded_genres == ['Horror']
    assert llm_client.last_call_latency_ms() == 0
    evidence = MovieEvidence(movie_id=1, title='Test Comedy', genres=['Comedy'],
                             matched_constraints=['Comedy'], evidence_keys=['movie:1:genres'])
    _, explanation = llm_client.generate_explanation([evidence], 'comedy')
    assert [item.movie_id for item in explanation.items] == [1]
    assert explanation.items[0].evidence_ids == ['movie:1:genres']


@pytest.mark.parametrize('phrase,minimum,maximum', [
    ('after 2000', 2001, None),
    ('since 2000', 2000, None),
    ('before 1990', None, 1989),
    ('up to 1990', None, 1990),
    ('newer than 2000', 2001, None),
    ('older than 1990', None, 1989),
])
def test_year_boundaries_distinguish_strict_and_inclusive_language(phrase, minimum, maximum):
    intent = fallback_extract_intent('show comedy movies ' + phrase)
    assert (intent.min_year, intent.max_year) == (minimum, maximum)
