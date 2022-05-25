import argparse
import os
from config.default import init_cfg
from utils.utils import create_logger, set_seed
import json
from model.trainer import Trainer
from model.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = init_cfg(args)
    cfg.freeze()
    # seed
    set_seed(cfg.seed)

    logger = create_logger(cfg.save_log_path)
    logger.info('======================= args =======================\n' + json.dumps(vars(args), indent=4))
    logger.info('======================= cfg =======================\n' + cfg.dump(indent=4))

    trainer = Trainer(cfg)
    trainer.train()

    #evaluator = Evaluator(cfg, trainer)
    #evaluator.get_attentions('01-25-17-49')
    #evaluator.evaluate()
    #evaluator.get_pred(17031)

if __name__ == "__main__":
    main()
