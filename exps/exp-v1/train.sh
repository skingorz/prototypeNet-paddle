FLAGS_cudnn_deterministic=True python tools/train.py --config "exps/exp-v1/config.yaml"
FLAGS_cudnn_deterministic=True python tools/test.py --config "exps/exp-v1/config.yaml" 