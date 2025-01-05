# SimHGCL
Simple Yet Effective Heterogeneous Graph Contrastive Learning for Recommendation

We will organize the complete code and upload it after the paper is accepted for publication.
### Enviroments
- python==3.10
- pytorch==2.0
- cuda==118
- dgl==2.0
## How to Run the code
### Environment Installation
```
pip install torch==2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```
pip install  dgl==2.0 -f https://data.dgl.ai/wheels/cu118/repo.html
```
### Running on Amazon, Douban-Movie, Yelp, and Movielens Datasets
```
python main.py --dataset=amazon --device='cuda:0' --lr=0.005 --ssl_lambda=0.002  --reg_lambda=0.0001  --temperature=0.2
--dim=64  --epochs=80  --batch_size=4096 --test_batch_size=300  --verbose=1  --GCN_layer=2  --col_D_inv=-0.3 
```
```
python main.py --dataset=douban-movie --device='cuda:0' --lr=0.005 --ssl_lambda=0.04  --reg_lambda=0.0001  --temperature=0.3
--dim=64  --epochs=100  --batch_size=4096 --test_batch_size=300  --verbose=1  --GCN_layer=2  --col_D_inv=-0.9
```
``` 
python main.py --dataset=yelp --device='cuda:0' --lr=0.001 --ssl_lambda=0.02  --reg_lambda=0.0001  --temperature=0.2
--dim=64  --epochs=120  --batch_size=4096 --test_batch_size=300  --verbose=1  --GCN_layer=2  --col_D_inv=-0.1
```
``` 
python main.py --dataset=movielens-1m --device='cuda:0' --lr=0.005 --ssl_lambda=0.06  --reg_lambda=0.0001  --temperature=0.2
--dim=64  --epochs=200  --batch_size=4096 --test_batch_size=300  --verbose=1  --GCN_layer=2  --col_D_inv=-0.5
``` 
