import argparse
def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden_dim')   
    

    parser.add_argument('--max_seq_len', type=int, default=300,
                    help='max_seq_len')
    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size')
    parser.add_argument('--embedding_dim', type=int, default=300,
                    help='embedding_dim')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='learning_rate')
    parser.add_argument('--grad_clip', type=float, default=1e-1,
                    help='grad_clip')
    parser.add_argument('--model', type=str, default="kim_cnn",
                    help='model name')
    parser.add_argument('--dataset', type=str, default="imdb",
                    help='dataset')
    parser.add_argument('--keep_dropout', type=str, default=0.8,
                    help='keep_dropout')

    #kim CNN
    parser.add_argument('--kernel_sizes', type=str, default="1,2,3,5",
                    help='kernel_sizes')
    parser.add_argument('--kernel_nums', type=str, default="256,256,256,256",
                    help='kernel_nums')
    parser.add_argument('--embedding_type', type=str, default="non-static",
                    help='embedding_type')
    
#
    args = parser.parse_args()

    args.kernel_sizes = [int(i) for i in args.kernel_sizes.split(",")]
    args.kernel_nums = [int(i) for i in args.kernel_nums.split(",")]
#
#    # Check if args are valid
#    assert args.rnn_size > 0, "rnn_size should be greater than 0"


    return args 
