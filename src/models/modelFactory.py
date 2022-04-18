from src.models.sinc import Sinc
from src.models.interponet import Interponet
from src.models.seanet import Seanet
from src.models.demucs import Demucs
from src.models.demucs_no_lstm import DemucsNoLSTM
from src.models.seanet_lstm import SeanetLSTM
from src.models.seanet_ft import SeanetFt
from src.models.ebrn import EBRN
from src.models.lapsrn import LapSrn
from src.models.interponet_2 import Interponet_2
from src.models.modules import Discriminator, MultiPeriodDiscriminator, MultiScaleDiscriminator


def get_model(args):
    if args.experiment.model == 'sinc':
        generator = Sinc(**args.experiment.sinc)
    elif args.experiment.model == 'interponet':
        generator = Interponet(args)
    elif args.experiment.model == 'interponet_2':
        generator = Interponet_2(args)
    elif args.experiment.model == 'seanet':
        generator = Seanet(**args.experiment.seanet)
    elif args.experiment.model == 'seanet_lstm':
        generator = SeanetLSTM(**args.experiment.seanet_lstm)
    elif args.experiment.model == 'seanet_ft':
        generator = SeanetFt(args.experiment.seanet_ft)
    elif args.experiment.model == 'demucs':
        generator = Demucs(**args.experiment.demucs)
    elif args.experiment.model == 'demucs_no_lstm':
        generator = DemucsNoLSTM(**args.experiment.demucs_no_lstm)
    elif args.experiment.model == 'ebrn':
        generator = EBRN(**args.experiment.ebrn)
    elif args.experiment.model == 'lapsrn':
        generator = LapSrn(**args.experiment.lapsrn)


    models = {'generator': generator}

    if 'adversarial' in args.experiment and args.experiment.adversarial:
        if args.experiment.discriminator_model == 'melgan':
            discriminator = Discriminator(**args.experiment.discriminator)
            models.update({'melgan':discriminator})
        elif args.experiment.discriminator_model == 'hifi':
            mpd = MultiPeriodDiscriminator()
            msd = MultiScaleDiscriminator()
            models.update({'mpd': mpd, 'msd': msd})
    return models