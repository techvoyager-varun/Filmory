"""Read-only checks of the dedicated local browser QA fixture (except login)."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', required=True)
    parser.add_argument('--phase', choices=['hidden', 'restored'], required=True)
    parser.add_argument('--movie', type=int, required=True)
    args = parser.parse_args()
    if not (args.email.startswith('filmory.frontend.qa.') and args.email.endswith('@example.com')):
        raise SystemExit('Only the dedicated QA account is permitted.')
    token = None
    def request(path, body=None):
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['Authorization'] = 'Bearer ' + token
        req = Request('http://127.0.0.1:8100' + path,
                      data=json.dumps(body).encode() if body is not None else None,
                      headers=headers)
        with urlopen(req, timeout=30) as response:
            return json.load(response)
    checks = []
    def check(name, condition, detail=None):
        checks.append({'check': name, 'passed': bool(condition), 'detail': detail})
        if not condition:
            raise AssertionError(name)
    health = request('/health')
    check('Loaded local recommendation models', health['status'] == 'healthy' and health['models_loaded'])
    auth = request('/api/auth/login', {'email': args.email, 'password': 'FilmoryLocalQa123!'})
    token = auth['accessToken']
    check('Dedicated unmapped fixture', auth['user']['name'] == 'Filmory Frontend QA')
    hidden = set(request('/api/users/me/hidden-movies'))
    check('Persisted feedback state', (args.movie in hidden) == (args.phase == 'hidden'))
    history = request('/api/users/me/history')
    consumed = {entry['movie']['movieId'] for entry in history}
    for variant in ['auto', 'static', 'normalized_fixed', 'damr', 'mmr']:
        movies = request('/api/recommendations/me?top_k=50&variant=' + variant)
        ids = [movie['movieId'] for movie in movies]
        check(variant + ' exclusions and uniqueness', not ((hidden | consumed) & set(ids)) and len(ids) == len(set(ids)), {'returned': len(ids)})
        check(variant + ' bounded finite scores', all(isinstance(movie.get('score'), (int, float)) and 0 <= movie['score'] <= 1 for movie in movies))
    sessions = request('/api/assistant/sessions')
    check('Browser chat persisted', bool(sessions))
    detail = request('/api/assistant/sessions/' + sessions[0]['session_id'])
    reply = next(message for message in reversed(detail['messages']) if message['role'] == 'assistant')
    movies = reply['movies']
    check('Live local assistant constraints', bool(movies) and all(
        'Comedy' in movie['genres'] and 'Horror' not in movie['genres']
        and movie['year'] > 2000 and 0 < movie['runtime'] <= 120 for movie in movies), {'returned': len(movies)})
    if args.phase == 'hidden':
        check('Assistant respects rejected movie', args.movie not in {movie['movieId'] for movie in movies})
    policy = json.loads((ROOT/'backend/ml/serving_policy.json').read_text())
    active = json.loads((ROOT/'backend/ml/active_bundle.json').read_text())
    bundle = Path(active['path'])
    if not bundle.is_absolute():
        bundle = ROOT/'backend/ml'/bundle
    hashes = {name: hashlib.sha256((bundle/name).read_bytes()).hexdigest() for name in policy['checkpoint_hashes']}
    check('Existing checkpoints unchanged', hashes == policy['checkpoint_hashes'])
    target = Path(__file__).resolve().parent / ('live-' + args.phase + '.json')
    target.write_text(json.dumps({'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Live local functional feedback checks; not a new relevance accuracy score.',
        'checks': checks}, indent=2) + '\n')
    print(f'{len(checks)} checks passed: {target}')


if __name__ == '__main__':
    main()
