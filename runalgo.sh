#!/bin/bash

python3 src/main.py --config=cromac --env-config=join1 with env_args.key="join1:hallway 4-5-6" env_args.n_agents=3  env_args.state_numbers="[4,5,6]" noise_epsilon=0.05 kappa=5 weight_clamp=0.1
python3 src/main.py --config=cromac --env-config=join1 with env_args.key="join1:hallway 3-3-4-4"  env_args.n_agents=4  env_args.state_numbers="[3,3,4,4]" noise_epsilon=0.05 kappa=10 weight_clamp=0.2 
python3 src/main.py --config=cromac --env-config=traffic with env_args.key="traffic:easy_slow" noise_epsilon=0.0005 weight_clamp=0.3 
python3 src/main.py --config=cromac --env-config=traffic with env_args.key="traffic:easy_fast" noise_epsilon=0.001 weight_clamp=0.6 
python3 src/main.py --config=cromac --env-config=lbf with env_args.time_limit=25 env_args.key="lbforaging:Foraging-1s-8x8-3p-1f-coop-v2" noise_epsilon=0.03 kappa=5 
python3 src/main.py --config=cromac --env-config=lbf with env_args.time_limit=15 env_args.key="lbforaging:Foraging-1s-8x8-4p-1f-coop-v2" noise_epsilon=0.05 kappa=10
python3 src/main.py --config=cromac --env-config=sc2  with env_args.map_name="1o_10b_vs_1r" noise_epsilon=0.0075 robust_weight=0.3 weight_clamp=0.3 
python3 src/main.py --config=cromac --env-config=sc2  with env_args.map_name="1o_2r_vs_4r" noise_epsilon=0.015 robust_weight=0.1 weight_clamp=0.2 

