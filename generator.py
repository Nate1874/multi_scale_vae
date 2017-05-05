import os
from scipy.misc import imsave

class Generator(object):
    
    def generate_and_save_images(self, nums, directory):
        imgs = self.sess.run(self.sample_out)
        for k in range(imgs.shape[0]):
            imgs_folder = os.path.join(directory, 'imgs_celeba_3_64_reshape_U')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)
            
            imsave(os.path.join(imgs_folder,'%d.png') % k,
                imgs[k].reshape(64,64,3))
