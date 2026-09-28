"""Real two-GPU loader and SPGroup lifecycle smoke test."""
import argparse

from bootstrap import setup

setup()
from minimax_sp import MiniMaxH3SPUNETLoader, sp_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet", required=True)
    parser.add_argument("--weight-dtype", default="default")
    parser.add_argument("--devices", default="auto")
    args = parser.parse_args()

    try:
        MiniMaxH3SPUNETLoader.execute(args.unet, args.weight_dtype, 2, args.devices)
        group = sp_group.active_group()
        if group is None or group.world != 2:
            raise RuntimeError("loader did not create the expected two-rank SPGroup")
        group.check_alive()
        print("PASS: loader created a live two-rank SPGroup", flush=True)
    finally:
        group = sp_group.active_group()
        if group is not None:
            group.shutdown()

    if sp_group.active_group() is not None:
        raise RuntimeError("SPGroup remained cached after shutdown")
    print("PASS: SPGroup shut down and left no cached group", flush=True)


if __name__ == "__main__":
    main()
