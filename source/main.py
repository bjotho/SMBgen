from source.mario_gym.mario_env import MarioEnv
from source.mario_gym.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from source import constants as c

from ray import init as ray_init
from ray.tune.registry import register_env
from ray.rllib.agents import impala

from threading import Thread
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    if c.HUMAN_PLAYER:
        from source import tools
        from source.states import main_menu, load_screen, level
        import sys
        game = tools.Control()
        game.fps = 60
        state_dict = {c.MAIN_MENU: main_menu.Menu(),
                      c.LOAD_SCREEN: load_screen.LoadScreen(),
                      c.LEVEL: level.Level(),
                      c.GAME_OVER: load_screen.GameOver(),
                      c.TIME_OUT: load_screen.TimeOut()}
        game.setup_states(state_dict, c.MAIN_MENU)
        game.main()
        sys.exit(0)

    checkpoint_dir = os.path.join(dir_path, "checkpoints")
    checkpoint_all = os.path.join(dir_path, "checkpoints", "all")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_all, exist_ok=True)

    register_env(c.ENV_NAME, lambda config: MarioEnv(config))

    def find_latest_checkpoint():
        largest = -1
        for chkpath in os.listdir(checkpoint_all):
            checkpoint_id = int(chkpath.split("_")[1])

            if checkpoint_id > largest:
                largest = checkpoint_id

        if largest == -1:
            return None
        else:
            ret_path = os.path.join(checkpoint_all, f"checkpoint_{str(largest)}" + f"/checkpoint-{str(largest)}")
            return ret_path

    def test(trainer):
        config = dict(
            actions=COMPLEX_MOVEMENT,
            window=False,
            fps=60_000
        )
        env = MarioEnv(config)

        while True:
            current_state = env.reset()
            done = False

            while not done:
                action = trainer.compute_action(current_state)
                new_state, reward, done, info = env.step(action)
                env.render()
                current_state = new_state

    ray_init()

    trainer = impala.ImpalaTrainer(env=c.ENV_NAME, config={
        "num_gpus": 2,
        "num_workers": 8
        # "model": {
        #     "conv_filters": [[c.OBS_FRAMES, c.OBS_SIZE, c.OBS_SIZE]]
        # }
        # "train_batch_size": 2048
    })
    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint:
        trainer.restore(latest_checkpoint)

    save_interval = 10
    save_counter = 0

    eval_thread = Thread(target=test, args=(trainer, ))
    eval_thread.daemon = True
    eval_thread.start()
    while True:
        trainer.train()
        if save_counter % save_interval == 1:
            checkpoint = trainer.save(checkpoint_all)
            print(checkpoint)
            """_chkp_path = os.path.dirname(os.path.abspath(checkpoint))
            try:
                os.remove(checkpoint_latest)
            except FileNotFoundError:
                pass
            os.symlink(_chkp_path, checkpoint_latest)
            #shutil.copy(checkpoint, checkpoint_latest)
            print("checkpoint saved at", checkpoint)"""
        save_counter += 1
