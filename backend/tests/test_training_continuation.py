import json
import pickle
import torch
from scripts.train_models import main

def test_continuation_keeps_clean_split_and_parent_until_validation_improves(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    config = dict(num_users=6,num_items=20,num_genres=2,embedding_dim_ncf=4,
        transformer_embedding_dim=8,transformer_heads=2,transformer_layers=1,transformer_max_len=4)
    (source/'model_config.json').write_text(json.dumps(config))
    (source/'movies_metadata.csv').write_text('movieId,title,genres\n100,Toy,Action\n')
    maps = {'user2idx':{100+u:u for u in range(6)},'idx2user':{u:100+u for u in range(6)},
        'movie2idx':{100+i:i for i in range(20)},'idx2movie':{i:100+i for i in range(20)},
        'genre2idx':{'Action':0,'Comedy':1},'idx2genre':{0:'Action',1:'Comedy'},
        'user_sequences':{u:[u,u+1,u+2,u+3,u+4] for u in range(6)}}
    for name,data in maps.items():
        with (source/(name+'.pkl')).open('wb') as f:pickle.dump(data,f)
    torch.save(torch.tensor([[1.,0.],[0.,1.]]*10), source/'movie_genre_matrix.pt')
    common = ['--source',str(source),'--epochs','1','--sequence-epochs','1',
        '--positives-per-epoch','18','--sequence-samples','12','--batch-size','4',
        '--embedding-dim','4','--eval-users','2','--catalogue-users','2','--threads','1']
    parent = tmp_path/'parent'; main(common+['--output',str(parent)])
    child = tmp_path/'child'; result=main(common+['--output',str(child),'--continue-from',str(parent), '--learning-rate','0.0002'])
    provenance=json.loads((child/'provenance.json').read_text())
    assert provenance['fresh_initialization'] is False
    assert len(provenance['parent_checkpoint_sha256']) == 3
    assert json.loads((child/'split.json').read_text()) == json.loads((parent/'split.json').read_text())
    parent_result=json.loads((parent/'metrics.json').read_text())
    for model, selected in result['selected_checkpoints'].items():
        parent_best=parent_result['selected_checkpoints'][model]['validation']['ndcg10']
        assert selected['validation']['ndcg10'] >= parent_best
