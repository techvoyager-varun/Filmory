"""Compare browser-observed genre cards with live catalogue and account state."""
import argparse
import json
import math
from pathlib import Path
from urllib.request import Request,urlopen


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--email',required=True)
    parser.add_argument('--ui-file',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if not (args.email.startswith('filmory.frontend.qa.') and args.email.endswith('@example.com')):
        raise ValueError('Only the dedicated QA fixture is permitted')
    token=None
    def request(path,body=None):
        headers={'Content-Type':'application/json'}
        if token: headers['Authorization']='Bearer '+token
        req=Request('http://127.0.0.1:8100'+path,headers=headers,
                    data=json.dumps(body).encode() if body is not None else None)
        with urlopen(req,timeout=30) as response: return json.load(response)
    auth=request('/api/auth/login',{'email':args.email,'password':'FilmoryLocalQa123!'})
    token=auth['accessToken']
    case=json.loads(args.ui_file.read_text(encoding='utf-8'))[-1]
    ids=[int(card['href'].rsplit('/',1)[-1]) for card in case['cards']]
    assert ids and len(ids)==len(set(ids)), 'Empty or duplicate frontend cards'
    movies=[request('/api/movies/'+str(mid)) for mid in ids]
    expected=set(case['genres'])
    matches=[bool(expected & set(movie['genres'])) for movie in movies]
    extra={}
    if case['kind'] in ('home','after_play'):
        if case['kind']=='home':
            assert all(matches), 'Some displayed movies do not match the requested genre(s)'
        account=request('/api/auth/me')
        assert set(account['favoriteGenres'])==expected, 'Preferences did not persist'
        live=request('/api/recommendations/me?top_k=12&variant=auto')
        assert ids==[movie['movieId'] for movie in live], 'Frontend recommendation cache is stale'
        if case['kind']=='after_play':
            history=request('/api/users/me/history')
            played={row['movie']['movieId'] for row in history}
            assert set(case['played_ids']) <= played, 'A frontend Play was not persisted'
            assert not played & set(ids), 'Already played titles remain in personalized recommendations'
            assert all(math.isfinite(movie['score']) and 0<=movie['score']<=1 for movie in live)
            if len(played)>=2:
                assert all(movie['variant']=='sequence_only' and movie['ncfScore'] is None for movie in live)
            extra={'played_ids':sorted(played),'variants':sorted({movie['variant'] for movie in live}),
                   'watched_excluded':True,'scores_finite_and_bounded':True}
    elif case['kind']=='assistant':
        assert all(matches), 'Some displayed movies do not match the requested genre(s)'
        assert all('Horror' not in movie['genres'] and 0<movie['runtime']<=150 for movie in movies), 'Assistant violated exclusions/runtime'
        sessions=request('/api/assistant/sessions')
        detail=request('/api/assistant/sessions/'+sessions[0]['session_id'])
        reply=next(message for message in reversed(detail['messages']) if message['role']=='assistant')
        assert ids==[movie['movieId'] for movie in reply['movies']], 'Displayed chat differs from persisted results'
    else:
        raise ValueError('Unknown verification case')
    entry={'kind':case['kind'],'requested_genres':case['genres'],'displayed':len(ids),'matching':sum(matches),
           'unique':True,'passed':True,'represented_requested_genres':sorted(expected & {g for movie in movies for g in movie['genres']}),
           'movies':[{'movieId':movie['movieId'],'title':movie['title'],'genres':movie['genres'],'runtime':movie['runtime']} for movie in movies],**extra}
    report=json.loads(args.report.read_text()) if args.report.exists() else {
        'scope':'Live frontend functional genre checks; dedicated QA account; not a user satisfaction or accuracy benchmark','cases':[]}
    report['cases'].append(entry)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"{case['kind']} {case['genres']}: {sum(matches)}/{len(ids)} matching; UI and server agree")


if __name__=='__main__': main()
