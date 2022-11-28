from gym.envs.registration import register
import mpe.scenarios as scenarios
# Multiagent envs
# ----------------------------------------

def _register(scenario_name, gymkey):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,

            "done_callback": scenario.done,
        },
    )

scenario_name = "simple_tag"
gymkey = "SimpleTag-heuristic-v0"
_register(scenario_name, gymkey)

scenario_name = "simple_tag_fix"
gymkey = "SimpleTag-fix-frozen-v0"
_register(scenario_name, gymkey)

scenario_name = "simple_tag_bull"
gymkey = "SimpleTag-bull-bull-v0"
_register(scenario_name, gymkey)

scenario_name = "simple_tag_flash"
gymkey = "SimpleTag-flash-random-v0"
_register(scenario_name, gymkey)

# ----------

scenario_name = "simple_tag_rabbit_hole"
gymkey = "SimpleTag-rabbit-hole-v0"
_register(scenario_name, gymkey)

for accel in [2.4]:
    for speed in [0.5]:
        scenario_name = f"simple_tag_accel_{accel}_speed_{speed}"
        gymkey = f"SimpleTag-accel-{accel}-speed-{speed}-v0"
        _register(scenario_name, gymkey)