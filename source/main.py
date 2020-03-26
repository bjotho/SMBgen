from source import constants as c
from source.states import level_state
import os

if not c.HUMAN_PLAYER:
    from threading import Thread
    from source.mario_gym.mario_env import MarioEnv
    from source.mario_gym.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
    from ray import init as ray_init
    from ray import shutdown as ray_shutdown
    from ray.memory_monitor import RayOutOfMemoryError
    from ray.tune.registry import register_env
    from ray.rllib.agents import dqn

dir_path = os.path.dirname(os.path.realpath(__file__))
_ray_error = False


def main():
    if c.HUMAN_PLAYER:
        run_game_main_and_exit()

    checkpoint_dir = os.path.join(dir_path, "checkpoints")
    checkpoint_all = os.path.join(dir_path, "checkpoints", "all")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_all, exist_ok=True)

    def test(trainer):
        global _ray_error

        config = dict(
            actions=COMPLEX_MOVEMENT,
            window=False,
            fps=60_000
        )
        env = MarioEnv(config)

        while not _ray_error:
            current_state = env.reset()
            done = False

            while not done and not _ray_error:
                action = trainer.compute_action(current_state)
                new_state, reward, done, info = env.step(action)
                env.render()
                current_state = new_state

        model_num = env.game.state_dict[c.LEVEL].training_sessions
        env.game.state_dict[c.LEVEL].generator.save_model(num=model_num)
        env.game.state_dict[c.LEVEL].generator.save_replay_memory(num=model_num)

    def restart_ray():
        global _ray_error
        _ray_error = False

        largest = level_state.find_latest_checkpoint(checkpoint_all)
        latest_checkpoint = None
        if largest > -1:
            latest_checkpoint = os.path.join(checkpoint_all, f"checkpoint_{str(largest)}/checkpoint-{str(largest)}")
            print("Resuming from ", latest_checkpoint)

        register_env(c.ENV_NAME, lambda config: MarioEnv(config))

        ray_init()

        trainer = dqn.ApexTrainer(env=c.ENV_NAME, config={
            "num_gpus": 1,
            "num_workers": 1,
            "eager": False,
            "model": {
                "conv_filters": [[c.OBS_FRAMES, c.OBS_SIZE, c.OBS_SIZE]]
            }
            # "train_batch_size": 2048
        })
        if latest_checkpoint and c.LOAD_CHECKPOINT:
            trainer.restore(latest_checkpoint)

        save_interval = 10
        save_counter = 0

        eval_thread = Thread(target=test, args=(trainer, ))
        eval_thread.daemon = True
        eval_thread.start()
        try:
            while True:
                trainer.train()
                if save_counter % save_interval == 1:
                    checkpoint = trainer.save(checkpoint_all)
                    print("Saved checkpoint:", checkpoint)

                save_counter += 1
        except RayOutOfMemoryError:
            print("Ray out of memory!")
        finally:
            print("Restarting ray...")
            _ray_error = True
            eval_thread.join()
            ray_shutdown()
            restart_ray()

    restart_ray()


def run_game_main_and_exit():
    import sys
    from source import tools
    from source.states import main_menu, load_screen
    if c.GENERATE_MAP:
        from source.states import level_gen as level
    else:
        from source.states import level

    game = tools.Control()
    game.fps = 60
    state_dict = {c.MAIN_MENU: main_menu.Menu(),
                  c.LOAD_SCREEN: load_screen.LoadScreen(),
                  c.LEVEL: level.Level(),
                  c.GAME_OVER: load_screen.GameOver(),
                  c.TIME_OUT: load_screen.TimeOut()}
    game.setup_states(state_dict, c.MAIN_MENU)
    if c.SKIP_MENU:
        game.flip_state(force=c.LEVEL)
    game.main()
    sys.exit(0)
