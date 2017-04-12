import os
from scipy.misc import imsave

class Generator(object):
    
    def generate_and_save_images(self, directory):
        imgs = self.sess.run([self.test_images[0],self.test_images[1],
            self.test_images[2],self.test_images[3],self.test_images[4]])
        for k in range(len(imgs)):
            imgs_folder = os.path.join(directory, 'imgs')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)
            
            imsave(os.path.join(imgs_folder,'%d.png') % k,
                imgs[k].reshape(self.height,self.width))
        print("saving==============================")
