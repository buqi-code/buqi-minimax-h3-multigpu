"""Use the same CLI as run_baseline.py, with world_size fixed to 2."""
import asyncio
from run_baseline import main

if __name__ == "__main__":
    asyncio.run(main(world=2))
