import argparse


def getConfig():
    resnet_path = './pretrain_model/resnet50-19c8e357.pth'
    load_test = './results/run-5/best_model_mae.pth'
    ''
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', type=str, default='F:/datasets/SODdatasets/')
    
    # Hyper-parameters
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--save_map', type=bool, default=True, help='Save prediction map')
    parser.add_argument('--FG_branch', type=bool, default=True, help='Dual branch training')
    parser.add_argument('--BG_branch', type=bool, default=True, help='Dual branch training')
    parser.add_argument('--model_choose', type=str, default='efficientnet', help='resnet50 or efficientnet')
    parser.add_argument('--frequency_radius', type=float, default=16, help='Frequency radius r in FFT')
    parser.add_argument('--clipping', type=float, default=2, help='Gradient clipping')
    parser.add_argument('--dataset', type=str, default='DUTS', help='DUTS')
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)  # Learning rate resnet:5e-5
    parser.add_argument('--aug_ver', type=int, default=2, help='1=Normal, 2=Hard')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10, help="Scheduler ReduceLROnPlateau's parameter & Early "
                                                                 "Stopping(+5)")
    parser.add_argument('--scheduler', type=str, default='Reduce', help='Reduce or Step')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--loss_f', type=str, default='BDL')

    # Training settings
    parser.add_argument('--version', type=str, default='4', help='Backbone Architecture')
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--arch', type=str, default='resnet')  # resnet or vgg
    parser.add_argument('--resnet_train', type=str, default=resnet_path)
    parser.add_argument('--load_train', type=str, default='')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--save_folder', type=str, default='./results')

    # Testing settings
    parser.add_argument('--load_test', type=str, default=load_test)
    parser.add_argument('--test_fold', type=str, default='./mask')  # Test results saving folder

    return parser.parse_args()


if __name__ == '__main__':
    cfg = getConfig()
    cfg = vars(cfg)
    print(cfg)
