import sys
from argparse import ArgumentParser
import subprocess, os
from datetime import datetime




def parse_train_args():
    parser = ArgumentParser()
    
    # Run settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--mini_train", action='store_true')
    parser.add_argument("--validate", action='store_true')
    parser.add_argument("--subset_train_as_val", action='store_true')
    parser.add_argument("--validate_on_train", action='store_true')
    parser.add_argument("--validate_on_test", action='store_true')

    # Training
    parser.add_argument("--mse", action='store_true')
    parser.add_argument("--kldiv", action="store_true", help="use KL-divergence loss")
    parser.add_argument("--seq_len", type=int, default=504, help="input sequence length")
    parser.add_argument("--cls", action='store_true')
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--constant_val_len", type=int, default=None)
    parser.add_argument("--accumulate_grad", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.)
    parser.add_argument("--lr_multiplier", type=float, default=1.0)
    parser.add_argument("--check_grad", action="store_true")
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--checkpoint_layers", action="store_true")
    parser.add_argument("--max_steps", type=int, default=450000)
    parser.add_argument("--max_epochs", type=int, default=100000)

    # Promoter Design Training
    parser.add_argument("--lr", type=float, default=5e-4)

    # Validate
    parser.add_argument("--check_val_every_n_epoch", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--fid_early_stop", action="store_true")
    parser.add_argument("--val_loss_es", action="store_true", help='only for cls train')
    parser.add_argument("--val_check_interval", type=int, default=None)
    parser.add_argument("--ckpt_iterations", type=int, nargs='+', default=None)
    parser.add_argument("--random_sequences", action="store_true")
    parser.add_argument("--taskiran_seq_path", type=str, default=None)

    # Data
    parser.add_argument('--dataset_type', type=str, choices=['enhancer', 'toy_fixed','toy_sampled'], default='argmax')
    parser.add_argument("--mel_enhancer", action='store_true')
    parser.add_argument("--overfit", action='store_true')
    parser.add_argument("--promoter_dataset", action='store_true')
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--seqsize", type=int, default=550, help="sequences will be padded so their length is equal to seqsize")
    parser.add_argument("--use_single_channel", action="store_true", help="use an extra channel to encode singleton information")
    parser.add_argument("--singleton_definition", choices=["integer", "threshold1100"], default="integer", help="singleton mode")
    parser.add_argument("--blocks", default=[256, 256, 128, 128, 64, 64, 32, 32], nargs="+", type=int, help="number of channels for EffNet-like blocks")
    parser.add_argument("--ks", default=5, type=int, help="kernel size of convolutional layers")
    parser.add_argument("--resize_factor", default=4, type=int, help="number of channels in a middle/high-dimensional convolutional layer of an EffNet-like block")
    parser.add_argument("--se_reduction", default=4, type=float, help="reduction number used in SELayer")
    parser.add_argument("--final_ch", default=18, type=int, help="number of channels of the final convolutional layer")
    parser.add_argument("--bn_momentum", default=0.1, type=float)

    # Logging
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--run_name", type=str, default="default")
    
    args = parser.parse_args()
    timestamp = datetime.fromtimestamp(datetime.now().timestamp()).strftime("%Y-%m-%d_%H-%M-%S")
    os.environ["MODEL_DIR"] = os.path.join("workdir", (args.run_name + '_' + timestamp))
    os.environ["WANDB_LOGGING"] = str(int(args.swanlab))

    from utils.logging import Logger
    os.makedirs(os.environ['MODEL_DIR'], exist_ok=True)
    sys.stdout = Logger(logpath=os.path.join(os.environ['MODEL_DIR'], f'log.log'), syspart=sys.stdout)
    sys.stdout.encoding = None # for pytorch lightning because it is stupid
    sys.stderr = Logger(logpath=os.path.join(os.environ['MODEL_DIR'], f'log.log'), syspart=sys.stderr)
    #if args.wandb:
        #if subprocess.check_output(["git", "status", "-s"]):
            #print('There were uncommited changes. Not running that stuff.')
            #exit()
    #args.commit = (
        #subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    #)
    return args