import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'video_base8_cos11_final':
        from .Video_base_model8_cos11_final import VideoBaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
