# Author: Ankush Gupta
# Date: 2015

'''
modified from gen.py for generating from on-desk images
'''
"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile
import glob
import cPickle as cp
import cv2 

## Define some configuration variables:
NUM_IMG = 100 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 5000 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
ImgPath = '/media/asd/01D076F80957F270/E_DataSets/bg_img/'
DB_FNAME = osp.join(DATA_PATH,'dset.h5')
# url of the data (google-drive public file):
OUT_FILE = 'results/SynthText.h5'
OUT_SET  = 'results/Out/'


def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in xrange(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']        
    db['data'][dname].attrs['txt'] = res[i]['txt']
    ######
    cv2.imwrite(OUT_SET+imgname, res[i]['img'][:,:,::-1])
    open(OUT_SET+imgname+'.txt', 'w').write(res[i]['txt'][0])



def main(viz=False):
  
  with open(ImgPath + 'imnames.cp', 'rb') as f:
    filtered_imnames = set(cp.load(f))

  # open the output h5 file:
  out_db = h5py.File(OUT_FILE,'w')
  out_db.create_group('/data')
  print colorize(Color.GREEN,'Storing the output in: '+OUT_FILE, bold=True)

  # get the names of the image files in the dataset:
  imnames = glob.glob(ImgPath+"*.jpg")
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  f_imnames = [ImgPath+x for x in list(filtered_imnames)]
  f_imnames = [x for x in f_imnames if osp.isfile(x)]
  np.random.shuffle(f_imnames)

  RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)
  for i in xrange(start_idx,end_idx):
    imname = f_imnames[i]

    try:

      img = cv2.imread(imname)[:,:,::-1]
      img = cv2.resize(img, dsize=(500, 600))

      #create fake planar depth and segmentation maps
      sz = np.shape(img)[:2]
      depth = np.ones(sz, dtype=np.float32)
      seg   = np.ones(sz, dtype=np.float32) 
      label = np.array([1])
      area  = np.array([ depth.size ])

      print colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True)
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
      if len(res) > 0:
        # non-empty : successful in placing text:
        add_res_to_db(os.path.split(imname)[1],res,out_db)
      # visualize the output:
      if viz:
        if 'q' in raw_input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print colorize(Color.GREEN,'>>>> CONTINUING....', bold=True)
      continue
  
  out_db.close()


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  args = parser.parse_args()
  main(args.viz)