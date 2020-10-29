# dacon-landmark


### Training
- Quick start

`` python main.py --batch_size=64 --test_batch_size=64 --save_itr=500 --test_itr=500 
``

- Background

`` nohup python3 -u main.py --batch_size=64 --test_batch_size=64 --save_itr=500 --test_itr=500 > 201029.log & 
``

- Resume

`` python main.py --batch_size=64 --test_batch_size=64 --save_itr=500 --test_itr=500 --resume./snapshots/2020-10-28_22_28_35/last.pt
``

### Test
`` python main.py --test --test_batch_size=256 --snapshot=./2020-10-28_22_28_35/last.pt``

### Note
- In `main.py` set `base` to be path of your dataset directory including `test.h5` and etc, e.g. `/data/micmic123/tmp/`.

- In `main.py`, set `os.environ['CUDA_VISIBLE_DEVICES']` as you want.