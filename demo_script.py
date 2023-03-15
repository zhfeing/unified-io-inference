import argparse

from PIL import Image

import uio.runner as runner
from uio.configs import CONFIGS
import numpy as np

from absl import logging
import warnings

# flax kicks up a lot of future warnings at the moment, ignore them
warnings.simplefilter(action='ignore', category=FutureWarning)

# To see INFO messages from `ModelRunner`
logging.set_verbosity(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", choices=list(CONFIGS))
    parser.add_argument("--model_weights")
    args = parser.parse_args()

    example_fig = "fig/dbg_img.png"

    model = runner.ModelRunner(args.model_size, args.model_weights)
    with Image.open(example_fig) as img:
        image = np.array(img.convert('RGB'))
    output = model.vqa(image, "What color is the sofa?")
    print(output["text"])  # Should print `green`


if __name__ == "__main__":
    main()
