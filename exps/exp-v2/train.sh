FLAGS_cudnn_deterministic=True python tools/train.py --config "exps/exp-v2/config.yaml"
FLAGS_cudnn_deterministic=True python tools/test.py --config "exps/exp-v2/config.yaml"