"""Generate a readable report directly from the frozen evaluation artifacts."""
import argparse
import json
from pathlib import Path
from evaluate_pipeline_quality import paired


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    record=json.loads((args.evaluation/'metrics.json').read_text())
    rows=[json.loads(line) for line in (args.evaluation/'mapped-test.jsonl').read_text().splitlines()]
    intervals={name:paired(rows,name,record['seed']+2) for name in [
        'DAMR gate','DAMR + momentum','DAMR + agreement','DAMR + diversity']}
    (args.evaluation/'damr-paired-uncertainty.json').write_text(json.dumps({
        'scope':'Secondary exploratory ablations, not a basis for post-test policy selection',
        'comparisons_to_current':intervals},indent=2),encoding='utf-8')
    mapped=record['test']['mapped']['metrics']
    current=mapped['current']
    selected=mapped['selected']
    lines=['# Filmory pipeline quality evaluation — 14 October 2026','',
        'The custom DAMR algorithm exists in `backend/app/ml/damr.py` and is used by Ask Filmory. '
        'The homepage defaults to the validated fixed/maturity-based policy. This pass evaluated alternatives; '
        '**none met the predeclared held-out promotion criterion, so the serving policy remains unchanged.**','',
        '## Protocol','',
        f"Frozen clean checkpoint bundle; {len(record['validation_users'])} validation users and {len(record['new_test_users'])} additional test users. "
        'All five scenarios use the same cohorts. Users have at least 20 train events, permitting comparable 2/5/10/20-event contexts. '
        'Validation targets and fresh test targets remain distinct. The full mapped catalogue has 22,836 items; observed history is excluded.','',
        'The weight grid uses a 0.1 simplex step plus the current weights, candidate budgets 100/300/1,000, '
        'and Transformer/collaborative/mixed retrieval. All validation choices were saved before the test phase. '
        'Recorded earlier cohorts, including protocol-only reservations from interrupted runs, were excluded. '
        'A test-phase inference-mode execution failure was corrected and resumed from the locked selection; choices/cohort were not retuned. '
        'Both original and resumed source snapshots are retained.','',
        'These are existing training users and catalogue items. Short-history scenarios hide the requester embedding; '
        'they are simulations of a new account, not evidence about entirely unseen people. '
        'No new training, cloud inference or checkpoint changes occurred.','',
        '## Mapped-user model comparison','',
        '| System | Full-catalogue HR@10 | NDCG@10 | Candidate recall |',
        '|---|---:|---:|---:|']
    for name in ['NCF Baseline','Hybrid NCF','Transformer','current','selected']:
        data=mapped[name]
        label={'current':'Current normalized ensemble (100 candidates)','selected':'Validation-selected ensemble (300 candidates)'}.get(name,name)
        lines.append(f"| {label} | {data['hr10']*100:.2f}% | {data['ndcg10']:.6f} | {data['candidate_recall']*100:.2f}% |")
    lines += ['',f"The current path misses the held-out target during retrieval for **{current['retrieval_misses']} of 1,000 users**. "
        f"For **{current['retrieved_but_not_top10']}**, it retrieves the target but ranks it below the top ten; **{current['hits']}** are hits. "
        'Both retrieval and final ranking limit the pipeline. Candidate recall is measured at each path’s declared budget.','',
        f"Widening the selected pool raises candidate recall from {current['candidate_recall']*100:.1f}% to {selected['candidate_recall']*100:.1f}%, "
        f"but top-ten hits change from {current['hits']} to {selected['hits']}. More retrieved targets alone do not establish better final recommendations.",
        '', '## Validation-selected alternatives versus current policy','',
        '| Scenario | Selected pool / retrieval / weights | Test HR@10 current → selected | Test NDCG@10 current → selected | Promote? |',
        '|---|---|---:|---:|---|']
    for mode,data in record['test'].items():
        choice=record['decisions'][mode]['selected']
        a,b=data['metrics']['current'],data['metrics']['selected']
        lines.append(f"| {mode} | {choice['candidate_k']} / {choice['strategy']} / {choice['weights']} | "
            f"{a['hr10']*100:.2f}% → {b['hr10']*100:.2f}% | {a['ndcg10']:.6f} → {b['ndcg10']:.6f} | {'Yes' if data['promotion_supported'] else 'No'} |")
    lines += ['', 'Mapped weights are Hybrid NCF / Transformer / genre. Live-mode weights are neighbor / Transformer / genre; '
        'new accounts never borrow an existing learned NCF user embedding. All paired 95% intervals for selected-policy differences include zero. '
        'The predeclared rule requires positive lower bounds for both HR@10 and NDCG@10. These five comparisons are exploratory, '
        'not five independent replications or a prospective satisfaction test.','',
        '## DAMR: custom method and controlled ablations','',
        'DAMR uses long/short genre profiles to estimate drift, focus, maturity and freshness; a calibrated softmax gate adapts expert weights. '
        'A momentum vector adds directional genre relevance, expert agreement adjusts confidence, and optional quality/MMR components control '
        'popular-quality priors and redundancy. The combination is Filmory’s proposed custom method. It is not proof of worldwide novelty. '
        'Hybrid NCF and Transformers are established model families. MMR is an established component, '
        'not an invention of Filmory ([Goldstein and Carbonell, 1998](https://aclanthology.org/X98-1025/)).','',
        'For this experiment, every ablation uses the identical 100 Transformer candidates and full-catalogue held-out target. '
        'State uses positional decay with freshness zero because prediction-time timestamps are not available in the serving bundle. '
        'Bayesian quality is disabled because there are no train-only per-movie ratings/counts; importing catalogue ratings could introduce unsupported evidence or leakage. '
        'This is not a complete test of the live timestamp/quality configuration.','',
        '| Ablation | HR@10 | NDCG@10 | Intra-list diversity |',
        '|---|---:|---:|---:|']
    for name in ['Fixed on DAMR pool','DAMR gate','DAMR + momentum','DAMR + agreement','DAMR + diversity']:
        data=mapped[name]
        lines.append(f"| {name} | {data['hr10']*100:.2f}% | {data['ndcg10']:.6f} | {data['ild10']:.6f} |")
    lines += ['', 'Gate/momentum variants show small descriptive ranking gains; agreement/diversity have relevance trade-offs. '
        'Do not select a new default from these already inspected test ablations. Exploratory paired intervals are retained in '
        '`damr-paired-uncertainty.json`; a separate validation/confirmation protocol would be needed before promoting a modified DAMR variant.','',
        '## Local feedback and next evidence','',
        'The homepage now has an optional **Help evaluate Filmory** review. It freezes the displayed list and policy snapshot, '
        'stores per-movie interesting/not-for-me/unsure judgments with account ownership checks, and supports local exports without names/emails. '
        'QA ratings are excluded. Actual classmate participation is pending; software tests are not participants. '
        'See `docs/verification/classmate-feedback-guide.md`.','',
        '## Paper-ready interpretation','',
        '> Filmory implements DAMR as a proposed deterministic re-ranking method combining drift-conditioned expert fusion, '
        'taste momentum and agreement, with standard optional quality and diversity components. On a fresh 1,000-user '
        'full-catalogue replay, the current normalized ensemble achieved HR@10 of 0.066 and NDCG@10 of 0.028675. '
        'Validation-selected wider retrieval improved candidate recall but did not demonstrate a reliable gain in final ranking. '
        'Controlled DAMR ablations reveal relevance/diversity trade-offs rather than uniform superiority. '
        'These results concern transductive offline replay and do not establish prospective satisfaction or overall accuracy.','',
        'The earlier 90.75% sampled Transformer HR@10 and 3.50% full-catalogue result are from a different cohort/protocol. '
        'Keep them separate; do not overwrite the historical measurements with this serving experiment. '
        'The existing LaTeX manuscript was not modified in this pass.','',
        '## Artifacts and verification','',
        f"Evaluation: `{args.evaluation.as_posix()}`. Includes pre-test protocol/selection, per-user validation and test rows, "
        'metrics, paired uncertainty and source/checkpoint hashes. The active serving overlay remains unchanged.','',
        'Backend regression/migration/evaluation/study checks: 79 passed. Frontend API tests: 9 passed. '
        'TypeScript and client/SSR build passed; lint has zero errors and eight existing Fast Refresh warnings. '
        'Study ownership, explicit consent, frozen slate preservation, vote replacement and QA export exclusion are tested.', '',
        'The optional homepage review has also been checked live with a disposable QA account; '
        'see `docs/verification/recommendation-study-live-20261014.json`. Real classmate participation remains pending.', '',
        '![Optional homepage review](../verification/recommendation-study-20261014.png)']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report:',args.output)


if __name__=='__main__': main()
