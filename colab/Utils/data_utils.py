from scipy.ndimage.filters import gaussian_filter1d
from scipy.misc import imread, imresize
from six.moves import urllib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tarfile
import zipfile
import os
import sys

#Global variable for image nomalization
CIFAR10_MEAN = None 
CIFAR10_STD = None
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_images():
    ''' Load sample images '''
    kitten, puppy = imread('./Utils/kitten.jpg'), imread('./Utils/puppy.jpg')
    # kitten is wide, and puppy is already square
    d = kitten.shape[1] - kitten.shape[0]
    kitten_cropped = kitten[:, d//2:-d//2, :]

    img_size = 200   # Make this smaller if it runs too slow
    x = np.zeros((2, img_size, img_size, 3))
    x[0, :, :, :] = imresize(puppy, (img_size, img_size))
    x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size))
    return x

def _normalize_image(img):
    img_max, img_min = np.max(img), np.min(img)
    img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype('uint8')

def plot_conv_images(x, out):
    ''' Plot original and convolution output images '''
    plt.subplot(2, 3, 1)
    plt.imshow(_normalize_image(x[0]))
    plt.title('Original image')
    plt.axis('off')        
    
    plt.subplot(2, 3, 2)
    plt.imshow(_normalize_image(out[0,:,:,0]))
    plt.title('Grayscale')
    plt.axis('off')        
    
    plt.subplot(2, 3, 3)
    plt.imshow(_normalize_image(out[0,:,:,1]))
    plt.title('Edges')
    plt.axis('off')        
    
    plt.subplot(2, 3, 4)
    plt.imshow(_normalize_image(x[1]))
    plt.axis('off')        
    
    plt.subplot(2, 3, 5)
    plt.imshow(_normalize_image(out[1,:,:,0]))
    plt.axis('off')        
    
    plt.subplot(2, 3, 6)
    plt.imshow(_normalize_image(out[1,:,:,1]))
    plt.axis('off')        
    
def maybe_download_and_extract(DATA_URL):
    ''' Download and extract the data if it doesn't already exist. '''
    dest_directory = './Utils'
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        if '.tar.gz' in filepath:
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)
        if '.zip' in filepath:
            zipfile.ZipFile(filepath).extractall(dest_directory)  
        print("Successfully downloaded and unpacked")
    else:
        print("Data has already been downloaded and unpacked.")

def _unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

def _preprocess_image(img, MEAN=IMAGENET_MEAN, STD=IMAGENET_STD):
    """ Preprocess an image: subtracts the pixel mean and divides by the standard deviation.  """
    return (img.astype(np.float32)/255.0 - MEAN) / STD

def _deprocess_image(img, MEAN=IMAGENET_MEAN, STD=IMAGENET_STD):
    """ Undo preprocessing on an image and convert back to uint8. """
    return np.clip(255 * (img * STD + MEAN), 0.0, 255.0).astype(np.uint8)

def load_CIFAR10(val_batch=[5]):
    ''' Load CIFAR10 dataset '''
    #maybe_download_and_extract('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    X_train, Y_train, X_val, Y_val = [], [], [], []
    for i in range(1,6):
        data = _unpickle("/content/drive/colab/elice/data/data_batch_%d" % i)
        if i not in val_batch:
            X_train.append(data['data'])
            Y_train.append(data['labels'])
        else:
            X_val.append(data['data'])
            Y_val.append(data['labels']) 
    test = _unpickle('/content/drive/colab/elice/data/test_batch')
    
    X_train = np.concatenate(X_train, axis=0).reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])
    Y_train = np.concatenate(Y_train, axis=0) 
    X_val = np.concatenate(X_val, axis=0).reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])
    Y_val = np.concatenate(Y_val, axis=0) 
    X_test = test['data'].reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])
    Y_test = np.array(test['labels'])
    
    #Normalize input images
    global CIFAR10_MEAN, CIFAR10_STD
    CIFAR10_MEAN = np.mean(X_train.astype(np.float32)/255.0, axis=0)
    CIFAR10_STD = np.std(X_train.astype(np.float32)/255.0, axis=0)
    X_train = np.array([_preprocess_image(img, CIFAR10_MEAN, CIFAR10_STD) for img in X_train])
    X_val   = np.array([_preprocess_image(img, CIFAR10_MEAN, CIFAR10_STD) for img in X_val])
    X_test  = np.array([_preprocess_image(img, CIFAR10_MEAN, CIFAR10_STD) for img in X_test])
    
    # Load the class-names
    Class_names = _unpickle("/content/drive/colab/elice/data/batches.meta")['label_names']
        
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, Class_names

def load_ImageNet_val(num=None):
    ''' Load a handful of validation images from ImageNet '''
    maybe_download_and_extract('http://cs231n.stanford.edu/imagenet_val_25.npz')
    
    f = np.load('./Utils/imagenet_val_25.npz')
    X = f['X']
    Y= f['y']
    Class_names = f['label_map'].item()
    if num is not None:
        X = X[:num]
        Y = Y[:num]
      
    #Normalize input images
    global IMAGENET_MEAN, IMAGENET_STD
    X = np.array([_preprocess_image(img, IMAGENET_MEAN, IMAGENET_STD) for img in X])
    Class_names = {i:c.split(',')[0] for i, c in Class_names.items()}
    
    return X, Y, Class_names
    
def plot_images(X, Y, C, idx=0, Each_Category=False, SaliencyMaps=None, ClassRepresentatve=None, Adversarial=None, Target_y=None):
    ''' Plot images '''
    if Each_Category:
        Category = set(Y)
        for i in range(10):
            while(1):
                if Y[idx] in Category:
                    Category.remove(Y[idx])
                    break
                else:
                    idx += 1
            
            plt.subplot(2, 5, Y[idx]+1)
            plt.imshow(_deprocess_image(X[idx], CIFAR10_MEAN, CIFAR10_STD))
            plt.title(C[Y[idx]])
            plt.axis('off')
            
    elif SaliencyMaps is not None:    
        for i in range(5):
            plt.subplot(2, 5, i+1)
            plt.imshow(_deprocess_image(X[idx+i], IMAGENET_MEAN, IMAGENET_STD))
            plt.title(C[Y[idx+i]])
            plt.axis('off')
            
            plt.subplot(2, 5, i+6)
            plt.imshow(SaliencyMaps[idx+i], cmap=plt.cm.hot)
            plt.title(C[Y[idx+i]])
            plt.axis('off')
            
    elif ClassRepresentatve is not None:    
        Iter = int(X.shape[0] / 4)
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(_deprocess_image(X[Iter*(i+1)], IMAGENET_MEAN, IMAGENET_STD))
            plt.title('%s\nIteration %d' % (C[Y], Iter*(i+1)))
            plt.axis('off')
            plt.gcf().set_size_inches(8, 8)
            
    elif Adversarial is not None:
        plt.subplot(1, 4, 1)
        plt.imshow(_deprocess_image(X[0]))
        plt.title(C[Y[0]])
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(_deprocess_image(Adversarial[0]))
        plt.title(C[Target_y])
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.title('Difference')
        plt.imshow(_deprocess_image((X-Adversarial)[0]))
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title('Magnified difference (10x)')
        plt.imshow(_deprocess_image(10 * (X-Adversarial)[0]))
        plt.axis('off')          
        
    else:
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(_deprocess_image(X[idx+i], IMAGENET_MEAN, IMAGENET_STD))
            plt.title(C[Y[idx+i]])
            plt.axis('off')
    plt.show()
        
def random_noise_image(num_iterations=100):
    X = 255 * np.random.rand(num_iterations, 224, 224, 3)
    X = _preprocess_image(X)
    return X

def jitter_image(X, ox, oy):
    Xi = np.roll(np.roll(X, ox, 1), oy, 2)
    return Xi

def unjitter_image(X, ox, oy):
    Xi = np.roll(np.roll(X, -ox, 1), -oy, 2)
    Xi = np.clip(Xi, -IMAGENET_MEAN/IMAGENET_STD, (1.0 - IMAGENET_MEAN)/IMAGENET_STD)
    return Xi

def blur_image(X, sigma=1):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X