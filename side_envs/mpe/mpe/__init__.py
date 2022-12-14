from gym.envs.registration import register
import mpe.scenarios as scenarios
# Multiagent envs
# ----------------------------------------

_particles = {
    "multi_speaker_listener": "MultiSpeakerListener-v0",
    "simple_adversary": "SimpleAdversary-v0",
    "simple_crypto": "SimpleCrypto-v0",
    "simple_push": "SimplePush-v0",
    "simple_reference": "SimpleReference-v0",
    "simple_speaker_listener": "SimpleSpeakerListener-v0",
    #"simple_spread": "SimpleSpread-v0",
    #"simple_tag": "SimpleTag-v0",
    "simple_world_comm": "SimpleWorldComm-v0",
    "climbing_spread": "ClimbingSpread-v0",
}


for scenario_name, gymkey in _particles.items():
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()

    # Registers multi-agent particle environments:
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )

# Registers the custom simple tag environment:

simple_tag_pairs = [
    (1, 3), # default
    (1, 4),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5)
]

for num_good, num_adv in simple_tag_pairs:
    scenario_name = "simple_tag"
    gymkey = f"SimpleTag-{num_good}good-{num_adv}adv-v0"
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(num_good_agents=num_good, num_adversaries=num_adv)
    
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )


simple_spread_n = [
    3,
    4,
    5,
    6
]
for n in simple_spread_n:
    scenario_name = "simple_spread"
    gymkey = f"SimpleSpread-{n}-v0"
    scenario = scenarios.load(scenario_name+'.py').Scenario()
    world = scenario.make_world(n)
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )
# Registers the custom double spread environment:

for N in range(2, 11, 2):
    scenario_name = "simple_doublespread"
    gymkey = f"DoubleSpread-{N}ag-v0"
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(N)

    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )