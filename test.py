import json
import logging
import os
import hydra
import wandb

from torch.utils.data import DataLoader


from src.models import modelFactory


logger = logging.getLogger(__name__)

WANDB_PROJECT_NAME = 'Bandwidth Extension'
WANDB_ENTITY = 'huji-dl-audio-lab'

METRICS_KEY_PESQ = 'Average pesq'
METRICS_KEY_STOI = 'Average stoi'
METRICS_KEY_LSD = 'Average lsd'
METRICS_KEY_SISNR = 'Average sisnr'
METRICS_KEY_VISQOL = 'Average visqol'

def run(args):

    from src.data import PrHrSet
    from src.evaluate import evaluate
    from src.log_results import log_results
    from src.utils import bold

    model = args.experiment.model
    dataset_name = os.path.basename(args.dset.name)

    logger.info(bold(f'Testing model {model} on {dataset_name} in {args.samples_dir}'))

    data_set = PrHrSet(args.samples_dir)
    logger.info(f'dataset size: {len(data_set)}')
    dataloader = DataLoader(data_set, batch_size=1, shuffle=False)
    avg_pesq, avg_stoi, avg_lsd, avg_sisnr, avg_visqol = evaluate(args, dataloader, epoch=args.epochs)

    metrics = {METRICS_KEY_PESQ: avg_pesq, METRICS_KEY_STOI: avg_stoi, METRICS_KEY_LSD: avg_lsd,
                    METRICS_KEY_SISNR: avg_sisnr, METRICS_KEY_VISQOL: avg_visqol}
    wandb.log(metrics, step=args.epochs)
    results = [metrics]
    info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
    logger.info('-' * 70)
    logger.info(bold(f"Overall Summary | {model}, {dataset_name} | {info}"))

    json.dump(results, open(args.test_results_file, "w"), indent=2)

    log_results(args, dataloader, epoch=args.epochs)

def _get_wandb_config(args):
    included_keys = ['eval_every', 'optim', 'lr', 'loss', 'epochs']
    wandb_config = {k: args[k] for k in included_keys}
    wandb_config.update(**args.experiment)
    wandb_config.update({'train': args.dset.train, 'test': args.dset.test})
    return wandb_config

def _main(args):
    global __file__
    print(args)
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str):
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    wandb_mode = os.environ['WANDB_MODE'] if 'WANDB_MODE' in os.environ.keys() else args.wandb.mode
    wandb.init(mode=wandb_mode, project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config=_get_wandb_config(args),
               group=os.path.basename(args.dset.name), resume=(args.continue_from != ""), name=args.experiment.name)

    run(args)


@hydra.main(config_path="conf", config_name="main_config")  # for latest version of hydra=1.0
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()