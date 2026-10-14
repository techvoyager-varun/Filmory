import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_pipeline_quality import metric, summarize, paired
from export_recommendation_study import build_report


def test_pipeline_diagnostics_separate_retrieval_and_ranking_failures():
    result=summarize([metric(0,False),metric(0,True),metric(2,True),metric(1,True)])
    assert result['retrieval_misses']==1 and result['retrieved_but_not_top10']==1
    assert result['hits']==2 and result['candidate_recall']==.75 and result['hr10']==.5
    assert result['conditional_hr10_given_retrieved']==pytest.approx(2/3)
    assert result['ndcg10']==pytest.approx((1+1/1.584962500721156)/4)


def test_equal_policy_cannot_pass_a_positive_improvement_gate():
    rows=[{'current':metric(i%3,True),'selected':metric(i%3,True)} for i in range(30)]
    comparison=paired(rows,'selected',42)
    for name in ['hr10','ndcg10']:
        assert comparison[name]['mean_difference']==0
        assert comparison[name]['95_percent_interval']==[0.,0.]


def test_local_pilot_export_excludes_qa_and_does_not_count_unsure_as_negative():
    users={1:SimpleNamespace(id=1,name='Filmory Frontend QA',email='filmory.frontend.qa.1@example.com'),
           2:SimpleNamespace(id=2,name='Classmate Private Name',email='private@example.com')}
    studies=[SimpleNamespace(id='qa',user_id=1,movies=[{'movieId':1}],answers={'1':'interesting'},policy_snapshot={}),
             SimpleNamespace(id='real',user_id=2,movies=[{'movieId':1},{'movieId':2},{'movieId':3}],
                             answers={'1':'interesting','2':'unsure','3':'not_for_me'},policy_snapshot={'variant':'auto'})]
    report=build_report(studies,users,'test-secret')
    assert report['participants']==1 and report['slates']==1 and report['completed_slates']==1
    assert report['interesting_share_among_decided']==.5
    assert report['answer_counts']['unsure']==1
    assert 'private@example.com' not in str(report) and 'Classmate Private Name' not in str(report)
    assert report['rows'][0]['participant']!='2' and 'user_id' not in report['rows'][0]
