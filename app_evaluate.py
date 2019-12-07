import sys
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .net import GAN
from .util import load_config

def evaluate(spec_files=None):
    if spec_files is None:
        return

    config_name = 'example'
    model_name = 'pretrained'
    shutil.copy('models/%s/%s/scripts/config_%s.py' %
                (config_name, model_name, config_name), 'config_tmp.py')
    cfg = load_config('tmp')
    cfg.name = config_name + '/' + model_name
    net = GAN(cfg, restore=True)
    net.restore(20000)
    print('processing files {}', spec_files)
    net.eval(spec_files=spec_files,
             output_dir='../app/static/download',
             step_by_step=False,
             show_linear=False,
             show_input=False,
             show_debug=False)
