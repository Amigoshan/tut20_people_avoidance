# tut20_people_avoidance

### Dependencies
- Python package
  ```pip install msgpack-rpc-python numpy opencv-python```

- PyTorch Version 0.5.0+, version 0.5.0, 1.3.0 are tested.

- Simulation environment 

  Download the binary file from: [CLUSTER]/data/datasets/wenshanw/unreal/Blocks_tut20. 


### Test existing DQN model

- Open the simulation environment
  ```sh Blocks.sh -windowed```

- Run the model:
  ``` python dqn_people_avoidance.py --use-int-plotter  --omni-dir --multi-frame 3 --test --load-qnet --qnet-model 4_3_dqn_2000000.pkl```

### Train a new model

- Open the simulation environment

- Run the command:
  ```python dqn_people_avoidance.py --exp-prefix "1_7_" --use-int-plotter --train-step 2000000 --train-interval 100 --batch-size 1000 --lr 0.001 --omni-dir --multi-frame 3 ``` 