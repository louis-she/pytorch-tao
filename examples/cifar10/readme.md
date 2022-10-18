# Cifar example using Tao

This is a image classification task using Tao to solve. The task is easy but the example covers almost everything Tao has. Before start you should install the dependencies from `requirements.txt`.

## Steps to run

1. copy the content of `main.py` to any empty folder or just clone this repo then switch into this folder.
2. `tao init`
3. `tao run main.py`

All the log will be saved in the `./log` folder.

## Steps to enable wandb

1. Search `Uncomment to use wandb` in the `main.py`, and just uncomment them, there should be 4 lines of them.
2. open `.tao/cfg.py` with any editor, configure the `wandb_project` and `wandb_api_key`, note that the config file is also a Python script.
3. `git add -A; git commit -m "add wandb tracker"`
4. `tao run main.py`

Now, open your wandb dashboard, should get some charts there.

## Steps to auto tune paramters

1. open `.tao/cfg.py`, uncomment `study_storage` and `tune_direction`
2. `git add -A; git commit -m "add wandb tracker"`
3. `tao tune main.py --max_epochs 2`
