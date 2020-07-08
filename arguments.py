import argparse

def get_args():
    parser = argparse.ArgumentParser(description='dqn')

    parser.add_argument('--working-dir', default='./',
                        help='working directory')

    parser.add_argument('--exp-prefix', default='debug_',
                        help='exp prefix')

    parser.add_argument('--env-name', default='PeopleAvoidDiscreteEnv',
                        help='environment to train on (default: PeopleAvoidDiscreteEnv)')

    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')

    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')

    parser.add_argument('--epsilon', default='0.9_0.05',
                        help='exploration rate (default: 0.9_0.05)')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size (default: 64)')

    parser.add_argument('--memory-size', type=int, default=100000,
                        help='size for replay memory (default: 100000)')

    parser.add_argument('--speedup', type=float, default=1.0,
                        help='speed up the simulation (default: 1.0)')

    parser.add_argument('--train-interval', type=int, default=10,
                        help='number of interactions in each training (default: 10)')

    parser.add_argument('--target-update-interval', type=int, default=10000,
                        help='update target network (default: 10000)')

    parser.add_argument('--episode-len', type=int, default=1000,
                        help='maximun length of an episode (default: 1000)')

    parser.add_argument('--snapshot', type=int, default=100000,
                        help='snapshot for models and figures (default: 100000)')

    parser.add_argument('--train-step', type=int, default=1000000,
                        help='number of interactions in total (default: 1000000)')

    parser.add_argument('--load-qnet', action='store_true', default=False,
                        help='Read and use an existing policy (default: False)')

    parser.add_argument('--load-memory', action='store_true', default=False,
                        help='Read and use an existing policy (default: False)')

    parser.add_argument('--memory-prefix', default='debug_',
                        help='memory file prefix')

    parser.add_argument('--test', action='store_true', default=False,
                        help='test (default: False)')

    parser.add_argument('--test-num', type=int, default=10,
                        help='test (default: False)')
    
    parser.add_argument('--qnet-model', type=str, default='',
                        help='policy model, format: xxx.pkl')

    parser.add_argument('--use-int-plotter', action='store_true', default=False,
                        help='Enable cluster mode.')

    parser.add_argument('--plot-interval', type=int, default=100,
                        help='The plot interval when running on cluster. Use only after --cluster option.')

    parser.add_argument('--omni-dir', action='store_true', default=False,
                        help='Use sensory input from four directions.')

    parser.add_argument('--strict-safe', action='store_true', default=False,
                        help='terminate the episode after collision.')

    parser.add_argument('--multi-frame', type=int, default=1,
                        help='stack historic frames on the input (default: 1)')

    args = parser.parse_args()

    return args
