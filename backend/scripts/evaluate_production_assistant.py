"""Exercise the production assistant handler without ground-truth filter injection.

Copies catalogue metadata into an isolated SQLite database; never alters the
application's users or sessions. Provider calls are disabled for this regression
run. Results do not certify live LLM performance or claim-level factuality.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

BACKEND=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BACKEND))
from app.database import Base, SessionLocal
from app.models.db_models import Movie,User
from app.ml.model_service import model_service
from app.schemas.assistant import MovieIntent
from app.services.assistant_orchestrator import handle_message,_matches_hard_constraints
from app.services.llm_client import fallback_extract_intent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    model_service.load_all()
    engine=create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    fields=["movie_id","title","genres","year","rating","rating_count","runtime","description","poster_url","backdrop_url"]
    with SessionLocal() as source:
        catalogue=[{name:getattr(movie,name) for name in fields} for movie in source.query(Movie).all()]
    results=[]
    with Session(engine) as db:
        db.bulk_insert_mappings(Movie,catalogue)
        user=User(id=1,name="Isolated evaluation",email="evaluation@example.invalid",password_hash="unused",model_user_id=None)
        db.add(user);db.commit()
        with patch("app.services.assistant_orchestrator.extract_intent",side_effect=lambda text,history: fallback_extract_intent(text)), \
             patch("app.services.llm_client._get_client",side_effect=RuntimeError("Offline regression; provider disabled")), \
             patch("app.ml.recommender.ensure_movie_posters",lambda *args:None):
            for setname,filename in [("development","assistant_queries.jsonl"),("hidden","assistant_queries_hidden.jsonl")]:
                cases=[json.loads(line) for line in (BACKEND/"evaluation"/filename).read_text(encoding="utf-8").splitlines() if line.strip()]
                for case in cases:
                    began=time.perf_counter()
                    try:
                        response=handle_message(user,case["query"],None,db)
                        ids=[m.movieId for m in response.movies]
                        constraints=MovieIntent(**case.get("constraints",{}))
                        # Truth is used only for AFTER-THE-FACT scoring, never retrieval.
                        violations=[mid for mid in ids if not _matches_hard_constraints(db.get(Movie,mid),constraints)]
                        allowed={ev.movie_id:set(ev.evidence_keys) for ev in response.evidence}
                        exp=response.structured_explanation
                        ref_invalid=[]
                        if exp:
                            for item in exp.items:
                                if item.movie_id not in ids or not item.evidence_ids or not set(item.evidence_ids)<=allowed.get(item.movie_id,set()):
                                    ref_invalid.append(item.movie_id)
                        row={"id":case["id"],"set":setname,"query":case["query"],"source":"offline production handler",
                             "routing_correct":response.intent.request_mode==case["expected_mode"],
                             "predicted_mode":response.intent.request_mode,"expected_mode":case["expected_mode"],
                             "returned_movie_ids":ids,"constraint_violations":violations,
                             "expected_movie_ids":case.get("expected_movie_ids",[]),
                             "expected_clarification":case.get("expect_clarification",False),
                             "clarification":bool(response.clarification),"expected_empty":case.get("expect_empty",False),
                             "empty":not ids,"invalid_evidence_reference_items":ref_invalid,
                             "intent":response.intent.model_dump(),"latency_ms":(time.perf_counter()-began)*1000,
                             "followup":[]}
                        follow=case.get("followups",[])
                        if case.get("followup"): follow=[case["followup"]]
                        for turn in follow:
                            query=turn if isinstance(turn,str) else turn.get("query","")
                            if not query:continue
                            response=handle_message(user,query,response.session_id,db)
                            row["followup"].append({"query":query,"intent":response.intent.model_dump(),"movie_ids":[m.movieId for m in response.movies]})
                        results.append(row)
                    except Exception as error:
                        db.rollback()
                        results.append({"id":case["id"],"set":setname,"error":type(error).__name__+": "+str(error)})
    summary={}
    for name in ["development","hidden"]:
        rows=[r for r in results if r["set"]==name];ok=[r for r in rows if "error" not in r]
        summary[name]={"queries":len(rows),"errors":len(rows)-len(ok),"routing_correct":sum(r["routing_correct"] for r in ok),
                       "returned_movies":sum(len(r["returned_movie_ids"]) for r in ok),
                       "constraint_violations":sum(len(r["constraint_violations"]) for r in ok),
                       "expected_clarification_correct":sum(r["clarification"] for r in ok if r["expected_clarification"]),
                       "expected_clarification_cases":sum(r["expected_clarification"] for r in ok),
                       "expected_empty_correct":sum(r["empty"] for r in ok if r["expected_empty"]),
                       "expected_empty_cases":sum(r["expected_empty"] for r in ok),
                       "invalid_evidence_reference_items":sum(len(r["invalid_evidence_reference_items"]) for r in ok),
                       "p50_ms":float(np.percentile([r["latency_ms"] for r in ok],50)) if ok else None,
                       "p95_ms":float(np.percentile([r["latency_ms"] for r in ok],95)) if ok else None}
    summary["limitations"]=["Offline heuristic extraction and template fallback; live provider not measured",
                             "Development and hidden cases have been inspected; neither is a new independently blinded set",
                             "No oracle filters supplied to production handler; evidence references do not establish semantic claim support",
                             "Catalogue copied from local PostgreSQL into isolated SQLite; timings include local inference, not browser/network"]
    (args.output/"per_query.jsonl").write_text("".join(json.dumps(r)+"\n" for r in results),encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
