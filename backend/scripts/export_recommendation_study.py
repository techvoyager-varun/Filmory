"""Export opt-in pilot outcomes locally, without names/emails or raw user IDs.

QA fixtures are excluded. Ratings reflect judged interest, not streaming
satisfaction. Never substitutes fabricated observations for human participation.
"""
import argparse
import hashlib
import hmac
import json
import sys
from pathlib import Path
from collections import Counter

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BACKEND))
from app.config import settings
from app.database import SessionLocal
from app.models.db_models import RecommendationStudy, User


def build_report(studies, users, secret):
    rows = []
    for study in studies:
        user = users[study.user_id]
        if user.name == 'Filmory Frontend QA' and user.email.startswith('filmory.frontend.qa.') and user.email.endswith('@example.com'):
            continue
        pseudonym = hmac.new(secret.encode(),str(study.user_id).encode(),hashlib.sha256).hexdigest()[:16]
        ranked_ids = [movie['movieId'] for movie in study.movies]
        answers = {key:value for key,value in study.answers.items() if int(key) in ranked_ids}
        rows.append({'participant':pseudonym,'slate_id':study.id,'movie_ids':ranked_ids,
                     'answers':answers,'policy_snapshot':study.policy_snapshot,
                     'completed':len(answers)==len(ranked_ids)})
    counts = Counter(choice for row in rows for choice in row['answers'].values())
    judged = counts['interesting']+counts['not_for_me']
    return {'scope':'Opt-in local judged-interest pilot; QA fixtures excluded; not actual watch satisfaction or controlled A/B evidence',
        'participants':len({row['participant'] for row in rows}), 'slates':len(rows),
        'completed_slates':sum(row['completed'] for row in rows), 'answer_counts':dict(counts),
        'interesting_share_among_decided':counts['interesting']/judged if judged else None,
        'limitations':['Self-selected participants','Repeated slates from one participant are not independent samples',
                       'Unknown/unrated movies are not automatically negative','No accuracy or significance claim from a small pilot'],
        'rows':rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output file to preserve prior study exports')
    with SessionLocal() as db:
        studies=db.query(RecommendationStudy).order_by(RecommendationStudy.created_at).all()
        users={user.id:user for user in db.query(User).filter(User.id.in_({study.user_id for study in studies}))}
        report=build_report(studies,users,settings.SECRET_KEY)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key not in ['rows','limitations']},indent=2))


if __name__=='__main__': main()
