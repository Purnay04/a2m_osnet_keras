import os
import glob
import re
from PIL import Image
import tensorflow as tf

class Market1501():
    dataset_dir = './Market-1501-v15.09.15'
    
    def __init__(self, root=dataset_dir, **kwargs):
        #self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')
        
        self._check_before_run()
        
        #print("train")
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        #print("query")
        
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        #print("gallery")
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
    
        print("=> Maket1501 Loaded")
        print("   Dataset Satatistics:")
        print("    ------------------------------") 
        print("    subset   | # ids | # images")
        print("    ------------------------------")
        print("    train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("    query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("    gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("    ------------------------------")
        print("    total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("    ------------------------------")
        
        self.train = train
        self.query = query
        self.gallery = gallery
    
        #print(train[0])
        #image = tf.keras.preprocessing.image.load_img(train[0][0])
        #print(tf.keras.preprocessing.image.img_to_array(image))
        #print(query[0:2])
        
        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
    
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        #print(pid2label)
        
        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            assert 0 <= pid <= 1501
            assert 1 <= camid <= 6
            camid -= 1
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        
        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
    
    
if __name__ == '__main__':
    obj = Market1501()