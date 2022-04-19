
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import applications 
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, load_model
from numpy import resize, expand_dims
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from keras import backend as K
import glob
from skimage import segmentation
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte
import tensorflow as tf
from skimage import io
import numpy as np
from skimage.util import img_as_float, img_as_ubyte
from PIL import Image
from numpy import resize, expand_dims
from IPython.display import HTML, display
from keras.preprocessing.image import ImageDataGenerator
import pylab
from tqdm import tqdm
import time
import csv
import os
import functools
import multiprocessing
from multiprocessing import Process, Manager
import threading
import tensorflow as tf
import matplotlib.image as mpimg
def synchronized(wrapped):
    lock = threading.Lock()
    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            return wrapped(*args, **kwargs)
    return _wrap
class Classfield_segmented(object):
       c = csv.writer(open("dados.csv", "w", newline=''))
       location_drive=''
       IMG_WIDTH, IMG_HEIGHT = 224,224
       dict_classes=['Carex_riparia','Cirsium_arvense','Dead_plants','Oenanthe_aquatica','Others','Phalaris_arundinacea','Phragmites_australis','Salix_alba','Salix_cinerea','Typha','Urtica_dioica']

       color_classes=[(0, 0, 0),(255, 255, 255),(0, 0, 255),(255, 0, 0),(0, 255, 0),(255, 255, 0),(255, 0, 255),(0, 255, 255),(255, 153, 153),(153, 153, 204),(102, 0, 0)]
       model=None  
       
       def getRAMinfo(self):
         p = os.popen('free')
         i = 0
         while 1:
            i = i + 1
            line = p.readline()
            if i==2:
              return(line.split()[1:4])
       @synchronized
       def predict(self,image):
          predict=self.model.predict(image)
          return np.argmax(predict, axis=1)
          
       def job_process(self,model_name,filename, preprocess_input, decode_predictions,list_iou):
            print(filename)
            im=io.imread(filename)
            #plt.imshow(im)
            
            #plt.show()
            
            img_load=im.copy()
            ##create a superpixels
            segments = segmentation.slic(img_load, n_segments=4000,compactness=10, sigma=5)
            #mark boundaries
            img_slic=mark_boundaries(img_load, segments)
            
            mpimg.imsave("result/"+model_name+"/slic/"+filename.split("image/")[1],img_slic)
            #Classificando os superpixels
            #print(segments)
            #superpixels=[]
            pred_list=[]
            qtd=len(np.unique(segments))
            print("Processando os superpixels da imagem")
            img_paint=im.copy()    
            for (i, seg) in tqdm(enumerate( np.unique(segments )), desc = 'tqdm() Progress Bar Segments'):
              # Create a mask, painting black all pixels outside of segment and white the pixels inside.
              mask_segment = np.zeros(  img_load.copy().shape[:2], dtype="uint8")
              mask_segment[segments == seg] = 255

              size_segment = mask_segment[segments == seg].size
              segment =  img_load.copy() 
              segment = cv2.bitwise_and(segment, segment, mask=mask_segment)
              # Get the countours around the segment
              contours, _  = cv2.findContours(mask_segment,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
         
              m = -1
              max_contour = None
              for cnt in contours:
                       if (len(cnt) > m):
                          m = len(cnt)
                          max_contour = cnt

              # Get the rectangle that encompasses the countour
              x,y,w,h = cv2.boundingRect(max_contour)
              segment = segment[y:y+h, x:x+w]

              #superpixels.append(segment)
         
              # pre-process the image for classification
              image = cv2.resize(segment, (self.IMG_HEIGHT, self.IMG_WIDTH))
              image = image.astype("float") / 255.0
              image = img_to_array(image)
              image = np.expand_dims(image, axis=0)
           
              predict = self.predict(image)
              
              color=None

                 
              for cont, name_classs in enumerate(self.dict_classes):  
                if(self.dict_classes[predict[0]]==name_classs):
                    color=self.color_classes[cont]
                    break
              #predict = np.argmax(predict, axis=1)
              #print(predict)
              pred_list.append(self.dict_classes[predict[0]])
              #if(idx_segment<4):
                #plt.imshow(segment)
                #print(dict_classes[predict[0]])
                #plt.show()
                #break
              
              #painting  
              height, width, channels = img_paint.shape   
               
              mask_inv = cv2.bitwise_not(mask_segment)
               
              # Paint all pixels in original image with choosed color
              class_color = np.zeros((height,width,3), np.uint8)
              class_color[:, :] = color
              colored_image = cv2.addWeighted(img_paint, 0.0, class_color, 1.0, 0)
               
              colored_image = cv2.bitwise_and(colored_image, colored_image, mask=mask_segment)
              clear = False
              # Create a new image keeping the painting only in pixels inside of segments
              new_image =   img_paint
              new_image = cv2.bitwise_and(new_image, new_image, mask=mask_inv)
              mask_segment[:] = 255
              img_paint = cv2.bitwise_or(new_image, colored_image, mask=mask_segment)
               
             

            
            mpimg.imsave("result/"+model_name+"/paint/"+filename.split("image/")[1],img_paint)
            ##Calcula o IoU
            
            """im_mask=cv2.imread(self.location_drive+"sheep_mask/"+filename.split("sheep/")[1].split("_img")[0]+"_label.png")
            print(self.location_drive+"sheep_mask/"+filename.split("sheep/")[1].split("_img")[0]+"_label.png")
            #print(im_mask)
            im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)
            ret,im_mask = cv2.threshold(im_mask,25, 255, cv2.THRESH_BINARY)
            #im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            #plt.imshow(im_mask ,cmap=pylab.gray())
            ##target, predition
            img_p=cv2.cvtColor(img_paint, cv2.COLOR_BGR2GRAY)
            r,img_p = cv2.threshold(img_p, 25, 255, cv2.THRESH_BINARY)
            intersection = np.logical_and(np.array(im_mask), np.array(img_p))
            union = np.logical_or(im_mask,img_p)
            
            mpimg.imsave("result/"+model_name+"/inter/"+filename.split("sheep/")[1],intersection)
            
            mpimg.imsave("result/"+model_name+"/union/"+filename.split("sheep/")[1],union)
            iou_score = np.sum(intersection) / np.sum(union)
            print(iou_score)
            list_iou.append(iou_score)
            
            self.c.writerow([str(model_name),str(filename.split("sheep/")[1]),str(iou_score)])             
            """
     
       def process(self):       
              dict_preprocessing = {}
              dict_preprocessing[1] = applications.vgg16.preprocess_input, applications.vgg16.decode_predictions
              dict_preprocessing[2] = applications.inception_v3.preprocess_input, applications.inception_v3.decode_predictions
              dict_preprocessing[3] = applications.densenet.preprocess_input, applications.densenet.decode_predictions
              dict_preprocessing[4] = applications.resnet_v2.preprocess_input, applications.resnet_v2.decode_predictions
              dict_preprocessing[5] = applications.xception.preprocess_input, applications.xception.decode_predictions

              self.c.writerow(["Modelo","Arquivo","IOU"])
              #names of models
              list_model=["Xception"] #,"ResNet152V2", "InceptionV3","VGG16", "DenseNet201"]
              for model_name in tqdm(list_model, desc = 'tqdm() Progress Bar Models'):
                K.clear_session()
                self.model = load_model(self.location_drive+"model/"+model_name+"_transfer_learning_adagrad.h5")
                preprocess_input=None
                decode_predictions=None
                if model_name=="VGG16":
                   preprocess_input, decode_predictions = dict_preprocessing[1]
                if model_name=="InceptionV3":
                   preprocess_input, decode_predictions = dict_preprocessing[2]
                if model_name=="DenseNet201":
                   preprocess_input, decode_predictions = dict_preprocessing[3]
                if model_name=="ResNet152V2":
                   preprocess_input, decode_predictions = dict_preprocessing[4]
                if model_name=="Xception":
                   preprocess_input, decode_predictions = dict_preprocessing[5]   
                list_iou=[]
                print("Load Model H5:"+model_name+"_transfer_learning_adagrad.h5")
                #Process each image
                threads=[]
                for filename in tqdm(glob.iglob(self.location_drive+'image/*.JPG', recursive=True), desc = 'tqdm() Progress Bar Files'):
                   #self.job_process(model_name,filename,preprocess_input, decode_predictions,list_iou)
                   th = threading.Thread(target=self.job_process,args=(model_name,filename,preprocess_input, decode_predictions,list_iou ))
                   threads.append(th)
                
                # Output is in kb, here I convert it in Mb for readability
                RAM_stats = self.getRAMinfo()
                RAM_total = round(int(RAM_stats[0]) / 1000,1)
                RAM_used = round(int(RAM_stats[1]) / 1000,1)
                print("RAM Total : "+str(RAM_total))
                print("RAM Used : "+str(RAM_used))
                 
                print("Wait a moment, the threads are processing "+str(len(threads)) +" images, it may be delayed depending on the size or quantity of the images!")
                with tqdm(total=len(threads)) as pbar:
                  for  t in threads:
                        t.start()
                        #t.join()
                        if((RAM_total)<33000):#se menor que 10gb
                             RAM_stats = self.getRAMinfo()
                             RAM_used = round(int(RAM_stats[1]) / 1000,1)
                             if((RAM_total-RAM_used)<6000):
                               t.join()
                             pbar.update(1)
                  pbar.close()
                for  t in threads:
                   t.join()
                #print("Average IoU"+ str(np.mean(list_iou)))
                #print("Model_name:"+list_iou)

if __name__ == "__main__":
    process_file= Classfield_segmented()
    process_file.process()

