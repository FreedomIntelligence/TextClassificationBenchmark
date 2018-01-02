# -*- coding: utf-8 -*-
import os,urllib
class Dataset(object):
    def __init__(self,opt=None):
        if opt is not None:
            self.setup(opt) 
        self.root=".data_waby"
        self.urls=[]
    def setup(self,opt):
#        self.http_proxy='http://dev-proxy.oa.com:8080'
        self.name=opt.dataset
        self.dirname=opt.dataset
        
        
    def process(self):
        dirname=self.download()
        print("processing dirname: "+ dirname)
        
        return dirname
    def download_from_url(self,url, path, schedule=None,http_proxy= "http://dev-proxy.oa.com:8080"):
        if schedule is None:
            schedule=lambda a,b,c : print("%.1f"%(100.0 * a * b / c), end='\r',flush=True) if (int(a * b / c)*100)%10==0 else None
        if http_proxy is not None:
            proxy = urllib.request.ProxyHandler({'http': http_proxy})
    # construct a new opener using your proxy settings
            opener = urllib.request.build_opener(proxy)
    # install the openen on the module-level
            urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url,path,lambda a,b,c : print("%.1f"%(100.0 * a * b / c), end='\r',flush=True) if (int(a * b / c)*100)%10==0 else None )
        return path
    
    def download(self,  check=None):
        """Download and unzip an online archive (.zip, .gz, or .tgz).
    
        Arguments:
            root (str): Folder to download data to.
            check (str or None): Folder whose existence indicates
                that the dataset has already been downloaded, or
                None to check the existence of root/{cls.name}.
    
        Returns:
            dataset_path (str): Path to extracted dataset.
        """
        import zipfile,tarfile
    
        path = os.path.join(self.root, self.name)
        check = path if check is None else check
        if not os.path.isdir(check):
            for url in self.urls:
                if isinstance(url, tuple):
                    url, filename = url
                else:
                    filename = os.path.basename(url)
                zpath = os.path.join(path, filename)
                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print('downloading {}'.format(filename))
                    
                    self.download_from_url(url, zpath)
                ext = os.path.splitext(filename)[-1]
                if ext == '.zip':
                    with zipfile.ZipFile(zpath, 'r') as zfile:
                        print('extracting')
                        zfile.extractall(path)
                elif ext in ['.gz', '.tgz']:
                    with tarfile.open(zpath, 'r:gz') as tar:
                        dirs = [member for member in tar.getmembers()]
                        tar.extractall(path=path, members=dirs)
        return os.path.join(path, os.path.splitext(filename)[-2])
    


if __name__ =="__main__":
    import opts
    opt = opts.parse_opt()
    opt.max_seq_len=-1
    from dataloader import Dataset
    x=Dataset(opt)
     
    x.process()
#    datas=loadData(opt)


