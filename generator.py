import os
from scipy.misc import imsave
import time

class Generator(object):
    
    def generate_and_save_images(self, nums, directory):
        t1= time.clock()
        imgs = self.sess.run(self.sample_out)
        t2= time.clock()
        print("generation time is ==== %f"%(t2-t1) )
        for k in range(imgs.shape[0]):
            imgs_folder = os.path.join(directory, 'imgs_street_tran')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)
            
            imsave(os.path.join(imgs_folder,'%d.png') % k,
                imgs[k].reshape(32,32,3))
