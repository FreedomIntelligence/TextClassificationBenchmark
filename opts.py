import argparse,os,re
import configparser

class Params(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Data input settings
        
        parser.add_argument('--config', type=str, default="no_file_exists",
                        help='gpu number')
            
            
        parser.add_argument('--hidden_dim', type=int, default=128,
                        help='hidden_dim')     
    
        parser.add_argument('--max_seq_len', type=int, default=200,
                        help='max_seq_len')
        parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
        parser.add_argument('--embedding_dim', type=int, default=-1,
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
        parser.add_argument('--embedding_training', type=str, default="false",
                        help='embedding_training')
        #kim CNN
        parser.add_argument('--kernel_sizes', type=str, default="1,2,3,5",
                        help='kernel_sizes')
        parser.add_argument('--kernel_nums', type=str, default="256,256,256,256",
                        help='kernel_nums')
        parser.add_argument('--embedding_type', type=str, default="non-static",
                        help='embedding_type')
        parser.add_argument('--lstm_mean', type=str, default="mean",# last
                        help='lstm_mean')
        parser.add_argument('--lstm_layers', type=int, default=1,# last
                        help='lstm_layers')
        parser.add_argument('--gpu', type=int, default=0,
                        help='gpu number')
        parser.add_argument('--proxy', type=str, default="null",
                        help='http://proxy.xx.com:8080')
        parser.add_argument('--debug', type=str, default="true",
                        help='gpu number')
    
        parser.add_argument('--embedding_dir', type=str, default=".glove/glove.6B.300d.txt",
                        help='embedding_dir')
        
        parser.add_argument('--bert_dir', type=str, default="D:/dataset/bert/uncased_L-12_H-768_A-12",
                        help='bert dir')
        parser.add_argument('--bert_trained', type=str, default="false",
                        help='fine tune the bert or not')
        
        parser.add_argument('--from_torchtext', type=str, default="false",
                        help='from torchtext or native data loader')
    #
        args = parser.parse_args()
        
        if args.config != "no_file_exists":
            if os.path.exists(args.config):
                config = configparser.ConfigParser()
                config_file_path=args.config
                config.read(config_file_path)
                config_common = config['COMMON']
                for key in config_common.keys():
                    args.__dict__[key]=config_common[key]
            else:
                print("config file named %s does not exist" % args.config)
    
#        args.kernel_sizes = [int(i) for i in args.kernel_sizes.split(",")]
#        args.kernel_nums = [int(i) for i in args.kernel_nums.split(",")]
    #
    #    # Check if args are valid
    #    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    
        if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
            os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)
        
        if args.model=="transformer":
            args.position=True
        else:
            args.position=False
            
        # process the type for bool and list    
        for arg in args.__dict__.keys():
            if type(args.__dict__[arg])==str:
                if args.__dict__[arg].lower()=="true":
                    args.__dict__[arg]=True
                elif args.__dict__[arg].lower()=="false":
                    args.__dict__[arg]=False
                elif "," in args.__dict__[arg]:
                    args.__dict__[arg]= [int(i) for i in args.__dict__[arg].split(",")]
                else:
                    pass
    
            
        if os.path.exists("proxy.config"):
            with open("proxy.config") as f:
    
                args.proxy = f.read()
                print(args.proxy)
        
        return args 
    
    def parse_config(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        config_common = config['COMMON']
        is_numberic = re.compile(r'^[-+]?[0-9.]+$')
        for key,value in config_common.items():
            result = is_numberic.match(value)
            if result:
                if type(eval(value)) == int:
                    value= int(value)
                else :
                    value= float(value)

            self.__dict__.__setitem__(key,value)            

    def export_to_config(self, config_file_path):
        config = configparser.ConfigParser()
        config['COMMON'] = {}
        config_common = config['COMMON']
        for k,v in self.__dict__.items():        
            if not k == 'lookup_table':    
                config_common[k] = str(v)

        with open(config_file_path, 'w') as configfile:
            config.write(configfile)

    def parseArgs(self):
        #required arguments:
        parser = argparse.ArgumentParser(description='running the complex embedding network')
        parser.add_argument('-config', action = 'store', dest = 'config_file_path', help = 'The configuration file path.')
        args = parser.parse_args()
        self.parse_config(args.config_file_path)
    
    def setup(self,parameters):
        for k, v in parameters:
            self.__dict__.__setitem__(k,v)
    def get_parameter_list(self):
        info=[]
        for k, v in self.__dict__.items():
            if k in ["validation_split","batch_size","dropout_rate","hidden_unit_num","hidden_unit_num_second","cell_type","contatenate","model"]:
                info.append("%s-%s"%(k,str(v)))
        return info
    
    def to_string(self):
        return "_".join(self.get_parameter_list())


def parse_opt():

    parser = argparse.ArgumentParser()
    # Data input settings
    
    parser.add_argument('--config', type=str, default="no_file_exists",
                    help='gpu number')
        
        
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden_dim')     

    parser.add_argument('--max_seq_len', type=int, default=200,
                    help='max_seq_len')
    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size')
    parser.add_argument('--embedding_dim', type=int, default=-1,
                    help='embedding_dim')
    
    
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='learning_rate')
    parser.add_argument('--lr_scheduler', type=str, default="none",
                    help='lr_scheduler')
    parser.add_argument('--optimizer', type=str, default="adam",
                    help='optimizer')
    parser.add_argument('--grad_clip', type=float, default=1e-1,
                    help='grad_clip')
            
    parser.add_argument('--model', type=str, default="bilstm",
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
    parser.add_argument('--embedding_training', type=str, default="false",
                    help='embedding_training')
    #kim CNN
    parser.add_argument('--kernel_sizes', type=str, default="1,2,3,5",
                    help='kernel_sizes')
    parser.add_argument('--kernel_nums', type=str, default="256,256,256,256",
                    help='kernel_nums')
    parser.add_argument('--embedding_type', type=str, default="non-static",
                    help='embedding_type')
    parser.add_argument('--lstm_mean', type=str, default="mean",# last
                    help='lstm_mean')
    parser.add_argument('--lstm_layers', type=int, default=1,# last
                    help='lstm_layers')
    parser.add_argument('--gpu', type=int, default=0,
                    help='gpu number')
    parser.add_argument('--gpu_num', type=int, default=1,
                    help='gpu number')
    parser.add_argument('--proxy', type=str, default="null",
                    help='http://proxy.xx.com:8080')
    parser.add_argument('--debug', type=str, default="true",
                    help='gpu number')
    parser.add_argument('--bidirectional', type=str, default="true",
                    help='bidirectional')
    
    parser.add_argument('--embedding_dir', type=str, default=".glove/glove.6B.300d.txt",
                    help='embedding_dir')
    
    parser.add_argument('--bert_dir', type=str, default="D:/dataset/bert/uncased_L-12_H-768_A-12",
                    help='bert dir')
    parser.add_argument('--bert_trained', type=str, default="false",
                    help='fine tune the bert or not')
    
    parser.add_argument('--from_torchtext', type=str, default="false",
                    help='from torchtext or native data loader')
#
    args = parser.parse_args()
    
    if args.config != "no_file_exists":
        if os.path.exists(args.config):
            config = configparser.ConfigParser()
            config_file_path=args.config
            config.read(config_file_path)
            config_common = config['COMMON']
            for key in config_common.keys():
                args.__dict__[key]=config_common[key]
        else:
            print("config file named %s does not exist" % args.config)

#        args.kernel_sizes = [int(i) for i in args.kernel_sizes.split(",")]
#        args.kernel_nums = [int(i) for i in args.kernel_nums.split(",")]
#
#    # Check if args are valid
#    assert args.rnn_size > 0, "rnn_size should be greater than 0"

    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)
    
    if args.model=="transformer":
        args.position=True
    else:
        args.position=False
        
    # process the type for bool and list    
    for arg in args.__dict__.keys():
        if type(args.__dict__[arg])==str:
            if args.__dict__[arg].lower()=="true":
                args.__dict__[arg]=True
            elif args.__dict__[arg].lower()=="false":
                args.__dict__[arg]=False
            elif "," in args.__dict__[arg]:
                args.__dict__[arg]= [int(i) for i in args.__dict__[arg].split(",")]
            else:
                pass

        
    if os.path.exists("proxy.config"):
        with open("proxy.config") as f:

            args.proxy = f.read()
            print(args.proxy)
    
    return args 