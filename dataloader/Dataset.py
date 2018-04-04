# -*- coding: utf-8 -*-
import os,urllib
class Dataset(object):
    def __init__(self,opt=None):
        if opt is not None:
            self.setup(opt)
            self.http_proxy= opt.__dict__.get("proxy","null")

        else:
            self.name="demo"
            self.dirname="demo"
            self.http_proxy="null"
               
        self.urls=[]
        self.root=".data"
        self.saved_path= os.path.join(os.path.join(self.root,"clean"),self.name)
        self.formated_files=None

        
        
    def setup(self,opt):

        self.name=opt.dataset
        self.dirname=opt.dataset
        self.http_proxy= opt.__dict__.get("proxy","null")
        
        
    def process(self):
        dirname=self.download()
        print("processing dirname: "+ dirname)
        raise Exception("method in father class have been called in processing: {} dataset".format(opt.dataset))
        return dirname
    
    
    def getFormatedData(self):
        
        if self.formated_files is not None:
            return self.formated_files
        
        if os.path.exists(self.saved_path):
            return [os.path.join(self.saved_path,filename) for filename in os.listdir(self.saved_path)]
        self.formated_files = self.process()
        return self.formated_files
    
    def download_from_url(self,url, path, schedule=None):
        #if schedule is None:
        #    schedule=lambda a,b,c : print("%.1f"%(100.0 * a * b / c), end='\r',flush=True) if (int(a * b / c)*100)%10==0 else None
        if self.http_proxy != "null":
            proxy = urllib.request.ProxyHandler({'http': self.http_proxy,'https': self.http_proxy})
    # construct a new opener using your proxy settings
            opener = urllib.request.build_opener(proxy)
    # install the openen on the module-level
            urllib.request.install_opener(opener)
            print("proxy in %s" % self.http_proxy)
#        urllib.request.urlretrieve(url,path,lambda a,b,c : print("%.1f"%(100.0 * a * b / c), end='\r',flush=True) if (int(a * b / c)*1000)%100==0 else None )a
        try:
            urllib.request.urlretrieve(url,path )
        except:
            import urllib2
            urllib2.urlretrieve(url,path )
        return path
    
    def download(self,check=None):
        """Download and unzip an online archive (.zip, .gz, or .tgz).
    
        Arguments:
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
                elif ext in ['.gz', '.tgz',".bz2"]:
                    with tarfile.open(zpath, 'r:gz') as tar:
                        dirs = [member for member in tar.getmembers()]
                        tar.extractall(path=path, members=dirs)
        else:
            print("%s do not need to be downloaded" % path)
        return path
    



