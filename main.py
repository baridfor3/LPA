
# -*- coding: utf-8 -*-
# code warrior: Barid
import argparse
import contextlib
import tensorflow as tf
import sys
import os
import core_model_initializer as init
cwd = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir)))
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'  # fp16 training
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.set_soft_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(0)
# import argparse


@contextlib.contextmanager
def config_options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


options = {
    "layout_optimizer": True,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
    "disable_model_pruning": True,
    "scoped_allocator_optimization": True,
    "pin_to_host_optimization": True,
    "implementation_selector": True,
    "auto_mixed_precision": True,
    "disable_meta_optimizer": True,
    "min_graph_nodes": True,
}
config_options(options)

cwd = os.getcwd()
sys.path.insert(0, cwd + "/corpus")
sys.path.insert(1, cwd)
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]

MODE = 'LIP'


def _load_weights_if_possible(self, model, init_weight_path=None):
    if init_weight_path:
        tf.compat.v1.logging.info("Load weights: {}".format(init_weight_path))
        model.load_weights(init_weight_path)
    else:
        tf.compat.v1.logging.info(
            "Weights not loaded from path:{}".format(init_weight_path))


def main():

    parser = argparse.ArgumentParser(description='Train Lip Reading system')
    parser.add_argument('-m',
                        '--mode',
                        nargs='?',
                        default='LIP',
                        choices=['LIP', 'GPT', 'VIS'])
    args = parser.parse_args()
    MODE = args.mode
    strategy = tf.distribute.MirroredStrategy()
    callbacks = init.get_callbacks()
    with strategy.scope():
        if MODE == 'GPT':
            train_dataset = init.train_Transformer_input()
        else:
            train_dataset = init.train_input()
        model = init.trainer(MODE)
        optimizer = init.get_optimizer()
        callbacks = init.get_callbacks()
        ##################
        model.compile(optimizer=optimizer)
        model.mode = MODE
        model.fit(train_dataset, epochs=100,
                  verbose=1, callbacks=callbacks)


if __name__ == "__main__":
    main()
