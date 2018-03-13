import argparse,os
def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden_dim')   
    

    parser.add_argument('--max_seq_len', type=int, default=200,
                    help='max_seq_len')
    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size')
    parser.add_argument('--embedding_dim', type=int, default=100,
                    help='embedding_dim')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='learning_rate')
    parser.add_argument('--grad_clip', type=float, default=1e-1,
                    help='grad_clip')

    parser.add_argument('--model', type=str, default="cnn",
                    help='model name')
    parser.add_argument('--dataset', type=str, default="imdb",
                    help='dataset')
    parser.add_argument('--position', type=bool, default=False,
                    help='gpu number')
    
    parser.add_argument('--keep_dropout', type=float, default=0.8,
                    help='keep_dropout')
    parser.add_argument('--max_epoch', type=int, default=20,
                    help='max_epoch')
    parser.add_argument('--embedding_file', type=str, default="glove.6b.300",
                    help='glove or w2v')
    #kim CNN
    parser.add_argument('--kernel_sizes', type=str, default="1,2,3,5",
                    help='kernel_sizes')
    parser.add_argument('--kernel_nums', type=str, default="256,256,256,256",
                    help='kernel_nums')
    parser.add_argument('--embedding_type', type=str, default="non-static",
                    help='embedding_type')
    parser.add_argument('--gpu', type=str, default="0",
                    help='gpu number')
    parser.add_argument('--proxy', type=str, default="null",
                    help='http://proxy.xx.com:8080')
    
    
    
#
    args = parser.parse_args()

    args.kernel_sizes = [int(i) for i in args.kernel_sizes.split(",")]
    args.kernel_nums = [int(i) for i in args.kernel_nums.split(",")]
#
#    # Check if args are valid
#    assert args.rnn_size > 0, "rnn_size should be greater than 0"

    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] =args.gpu
    
    if args.model=="transformer":
        args.position=True
    print("papameter parsing done")
        
    return args 
