import tensorflow as tf
import rsa
import hashlib
import numpy as np
from base64 import b64encode, b64decode

class PRW():
    def __init__(self):
        self.o_pub, self.o_pri = rsa.newkeys(512)
    
    def get_keypair(self):
        return self.o_pub, self.o_pri
    
    def h1(self, msg):
        hexa = hashlib.sha224(msg).hexdigest()
        return int(hexa, base=16)
    
    def h2(self, msg):
        hexa = hashlib.sha256(msg).hexdigest()
        return int(hexa, base=16)
    
    def h3(self, msg):
        hexa = hashlib.sha384(msg).hexdigest()
        return int(hexa, base=16)
    
    def h4(self, msg):
        hexa = hashlib.sha512(msg).hexdigest()
        return int(hexa, base=16)
    
    def create_signature(self, v):
        return b64encode(rsa.sign(v.encode('UTF-8'), self.o_pri, 'SHA-256')).decode('UTF-8')
    
    # @param  str sig RSA sinature, can be created using create_signature().
    # @param  int h   Height of the images used for training.
    # @param  int w   Width of the images used for training.
    # @param  int y   Total number of classes used in the training dataset.
    # @param  int n   The embedding pattern will be of shape (n, n, channels) or (n, n) if channels is zero.
    # @param  int ch  Channels of the input images, embedding pattern will be of shape (n, n, channels) or (n, n) if channels is zero.
    # @return int     The class which will be used as a watermark.
    def transform(self, sig, h, w, y, n, ch):
        sig = b64decode(sig.encode('UTF-8'))
        p = np.ones((h, w)) * 0.5
        y_w = self.h1(sig) % y
        bits = '{0:b}'.format(self.h2(sig) % (2**(n**2))) 
        # left pad with zeros
        bits = (max((n**2 - len(bits)), 0) * '0') + bits
        pos = (self.h3(sig) % (h - n), self.h4(sig) % (w - n))
        for y_cur in range(n):
            for x_cur in range(n):
                p[y_cur + pos[0], x_cur + pos[1]] = bits[y_cur * n + x_cur]
        lmbda = 1000
        for i in range(ch):
            p = np.expand_dims(p, axis=-1)
        self.p = p
        self.y_w = y_w
        self.lmbda = lmbda
        return y_w
    
    def apply_null_embedding(self, example, label):
        lmbdas = tf.cast(tf.ones(self.p.shape) * self.lmbda, tf.float64)
        example = tf.where(self.p == 0, -lmbdas, example)
        example = tf.where(self.p == 1, lmbdas, example)
        return example, label
        
    def apply_true_embedding(self, example, label):
        lmbdas = tf.cast(tf.ones(self.p.shape) * self.lmbda, tf.float64)
        example = tf.where(self.p == 0, lmbdas, example)
        example = tf.where(self.p == 1, -lmbdas, example)
        return example, tf.cast(self.y_w, tf.int64)
    
    def verify(self, model, h, w, y, n, sig, v, dataset, t_watermark, verbose=False):
        try:
            rsa.verify(v.encode('UTF-8'), b64decode(sig.encode('UTF-8')), self.o_pub)
        except rsa.pkcs1.VerificationError:
            if verbose:
                print("Argument v is not a valid signature.")
            return False
        
        y_w = self.transform(sig, h, w, y, n, 1)
        
        null_embedded = dataset.map(self.apply_null_embedding)
        true_embedded = dataset.map(self.apply_true_embedding)
        
        null_embedded = null_embedded.batch(128)
        true_embedded = true_embedded.batch(128)
        
        # model.evaluate is not working
        
        acc_null = model.evaluate(null_embedded, return_dict=True)['sparse_categorical_accuracy']
        acc_true = model.evaluate(true_embedded, return_dict=True)['sparse_categorical_accuracy']
        
        if verbose:
            print(f"null embedding has an accuracy of {round(acc_null, 4)}")
            print(f"true embedding has an accuracy of {round(acc_true, 4)}")
        
        if min(acc_null, acc_true) > t_watermark:
            return True
        return False