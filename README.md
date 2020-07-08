# tut20_people_avoidance

### Dependencies
```pip install msgpack-rpc-python numpy opencv-python```

### Test existing DQN model

Open the simulation environment

``` python dqn_people_avoidance.py --use-int-plotter  --omni-dir --multi-frame 3 --test --load-qnet --qnet-model 4_3_dqn_2000000.pkl```