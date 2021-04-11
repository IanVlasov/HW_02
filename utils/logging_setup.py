import logging
import sys


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
        # handlers=[logging.StreamHandler()]
    )
    logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":
    setup_logging()
    logging.getLogger(__name__).info('logging set up')