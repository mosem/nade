import itertools
import logging
import os
import hydra
import wandb

from torch.utils.data import DataLoader

from src.executor import start_ddp_workers
from src.models import modelFactory

logger = logging.getLogger(__name__)

WANDB_PROJECT_NAME = 'Bandwidth Extension'
WANDB_ENTITY = 'huji-dl-audio-lab'


def run(args):
    import torch

    from src import distrib
    from src.data import LrHrSet
    from src.solver import Solver
    logger.info(f'calling distrib.init')
    distrib.init(args)

    # torch also initialize cuda seed if available
    torch.manual_seed(args.seed)

    models = modelFactory.get_model(args)
    wandb.watch(tuple(models.values()), log=args.wandb.log, log_freq=args.wandb.log_freq)

    if args.show:
        logger.info(models)
        mb = sum(p.numel() for p in models.parameters()) * 4 / 2 ** 20
        logger.info('Size: %.1f MB', mb)
        return

    assert args.experiment.batch_size % distrib.world_size == 0
    args.experiment.batch_size //= distrib.world_size

    # Building datasets and loaders
    tr_dataset = LrHrSet(args.dset.train, args.experiment.lr_sr, args.experiment.hr_sr,
                         args.experiment.stride, args.experiment.segment)
    tr_loader = distrib.loader(tr_dataset, batch_size=args.experiment.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    if args.dset.valid:
        cv_dataset = LrHrSet(args.dset.test, args.experiment.lr_sr, args.experiment.hr_sr,
                            stride=None, segment=None)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        cv_loader = None

    if args.dset.test:
        tt_dataset = LrHrSet(args.dset.test, args.experiment.lr_sr, args.experiment.hr_sr,
                             stride=None, segment=None, with_path=True)
        tt_loader = distrib.loader(tt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    if torch.cuda.is_available():
        for model in models.values():
            model.cuda()

    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(models['generator'].parameters(), lr=args.lr, betas=(0.9, args.beta2))
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    optimizers = {'optimizer': optimizer}

    if 'adversarial' in args.experiment and args.experiment.adversarial:
        if args.experiment.discriminator_model == 'melgan':
            disc_optimizer = torch.optim.Adam(models['melgan'].parameters(), lr=args.lr, betas=(0.9, args.beta2))
        elif args.experiment.discriminator_model == 'hifi':
            disc_optimizer = torch.optim.AdamW(itertools.chain(models['msd'].parameters(), models['mpd'].parameters()),
                                               args.lr, betas=(0.9, args.beta2))
        optimizers.update({'disc_optimizer': disc_optimizer})


    # Construct Solver
    solver = Solver(data, models, optimizers, args)
    solver.train()

    distrib.close()


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
        logging.getLogger("src").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    tags = args.wandb.tags
    wandb_mode = os.environ['WANDB_MODE'] if 'WANDB_MODE' in os.environ.keys() else args.wandb.mode
    wandb.init(mode=wandb_mode, project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config=_get_wandb_config(args),
               group=os.path.basename(args.dset.name), resume=(args.continue_from != ""), name=args.experiment.name,
               tags=tags)

    if args.ddp and args.rank is None:
        start_ddp_workers(args)
    else:
        run(args)

    wandb.finish()


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