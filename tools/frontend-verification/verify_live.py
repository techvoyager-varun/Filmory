"""Read-only checks against the running local Filmory API, using the QA account."""
import argparse
import datetime
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE = 'http://127.0.0.1:8100'
RESULTS = []

def request(path, token=None, body=None, expected=200):
    headers = {'Content-Type': 'application/json'}
    if token: headers['Authorization'] = 'Bearer ' + token
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urlopen(Request(BASE+path, data=data, headers=headers), timeout=60) as response:
            status, raw = response.status, response.read()
    except HTTPError as error:
        status, raw = error.code, error.read()
    assert status == expected, f'{path}: HTTP {status}, expected {expected}'
    return json.loads(raw) if raw else None

def check(name, condition, detail):
    assert condition, f'{name}: {detail}'
    RESULTS.append({'check': name, 'status': 'pass', 'detail': detail})

def ranked(name, movies, excluded=()):
    ids = [m['movieId'] for m in movies]
    check(name+' nonempty / unique / finite', bool(ids) and len(ids)==len(set(ids)) and all(math.isfinite(m['score']) for m in movies), {'returned':len(ids),'variants':sorted({m.get('variant') or 'fallback' for m in movies})})
    check(name+' excludes consumed IDs', not set(ids).intersection(excluded), {'excluded_count':len(set(excluded))})
    return ids

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--email',required=True)
    parser.add_argument('--phase',choices=['before','after'],default='before')
    args=parser.parse_args()
    if not args.email.startswith('filmory.frontend.qa.') or not args.email.endswith('@example.com'):
        raise SystemExit('Use only the dedicated frontend QA account created for this check.')
    health=request('/health')
    check('Backend health',health['status']=='healthy' and health['models_loaded'],{'device':health['device'],'models_loaded':health['models_loaded']})
    auth=request('/api/auth/login',body={'email':args.email,'password':'FilmoryLocalQa123!'})
    token=auth['accessToken']
    user=request('/api/auth/me',token)
    check('QA session restoration',user['email']==args.email and user['name']=='Filmory Frontend QA',{'onboarding_completed':user['onboardingCompleted']})
    check('Persisted onboarding',set(user['favoriteGenres'])=={'Sci-Fi','Comedy','Thriller'} and set(user['favoriteMovieIds'])=={2571,589},'Selected genres and two favourite films persist through /me.')
    page=request('/api/movies?limit=48')
    check('Catalog pagination',page['total']>=27000 and len(page['movies'])==48,{'catalogue_total':page['total']})
    genre=request('/api/movies?genre=Sci-Fi&limit=48')
    check('Genre filter',all('Sci-Fi' in m['genres'] for m in genre['movies']),{'returned':len(genre['movies'])})
    titles=request('/api/movies?sort=title&limit=48')['movies']
    repeated=request('/api/movies?sort=title&limit=48')['movies']
    check('Title-sort stability', [m['movieId'] for m in titles]==[m['movieId'] for m in repeated] and [m['movieId'] for m in titles]!=[m['movieId'] for m in page['movies']], 'Title sort is repeatable and differs from popularity. Ordering follows PostgreSQL collation, not Python Unicode ordering.')
    ratings=request('/api/movies?sort=rating&limit=48')['movies']
    check('Rating sort',[m['rating'] for m in ratings]==sorted((m['rating'] for m in ratings),reverse=True),'Descending rating order verified.')
    years=request('/api/movies?sort=year&limit=48')['movies']
    check('Year sort',[m['year'] for m in years]==sorted((m['year'] for m in years),reverse=True),'Descending year order verified.')
    matches=request('/api/search?query=Matrix')
    check('Title search',bool(matches) and all('matrix' in m['title'].lower() for m in matches),{'returned':len(matches)})
    check('Empty search',request('/api/search?query=zzzzFilmoryQaNoSuchTitle')==[],'No-result search returns an empty list.')
    request('/api/movies/999999999',expected=404)
    RESULTS.append({'check':'Unknown movie ID','status':'pass','detail':'404 rather than a fabricated movie.'})
    history=request('/api/users/me/history',token)
    likes=request('/api/users/me/likes',token)
    saved=request('/api/users/me/my-list',token)
    seen={e['movie']['movieId'] for e in history}|set(likes)|{m['movieId'] for m in saved}
    ranked('QA recommendations',request('/api/recommendations/me?top_k=12&variant=auto',token),seen|set(user['favoriteMovieIds']) if args.phase=='before' else seen)
    ranked('Guest feed',request('/api/recommendations/guest?top_k=12&variant=auto'))
    ranked('Similar movies',request('/api/similar/2571?top_k=14'),[2571])
    demo=request('/api/auth/demo',body={})
    ranked('Mapped demo feed',request('/api/recommendations/me?top_k=12&variant=auto',demo['accessToken']))
    metrics=request('/api/metrics')
    check('Model metrics', 'Transformer' in metrics['results'] and bool(metrics['full_catalogue']), 'Sampled and full-catalogue results are available separately.')
    taste=request('/api/taste-state',token)
    check('Taste state',bool(taste['genres']) and len(taste['vectors']['long'])==len(taste['genres']),'Diagnostic taste vectors have matching dimensions.')
    if args.phase=='after':
        check('Recorded play history',{2571,589}.issubset({e['movie']['movieId'] for e in history}),'Two distinct mapped films persist in history.')
        recommendations=request('/api/recommendations/me?top_k=12&variant=auto',token)
        check('New-account sequence ranking',bool(recommendations) and all(m.get('variant')=='sequence_only' for m in recommendations),{'returned':len(recommendations),'weights':recommendations[0]['expertWeights']})
        sessions=request('/api/assistant/sessions',token)
        check('Assistant session saved',bool(sessions),'QA chat is persisted and visible in session listing.')
        detail=request('/api/assistant/sessions/'+sessions[0]['session_id'],token)
        reply=next(m for m in reversed(detail['messages']) if m['role']=='assistant')
        movies=reply['movies']
        check('Assistant hard constraints', bool(movies) and all('Comedy' in m['genres'] and 'Horror' not in m['genres'] and m['year']>2000 and 0<m['runtime']<=120 for m in movies),{'returned':len(movies),'scope':'One live QA query, not a general dialogue-accuracy score.'})
    report={'checked_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat(),'phase':args.phase,'scope':'Live local API functional checks; no model retraining or recommendation-accuracy claim.','checks':RESULTS}
    target=Path(__file__).resolve().parent/f'live-checks-{args.phase}.json'
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'{len(RESULTS)} live checks passed; report: {target}')

if __name__=='__main__': main()
