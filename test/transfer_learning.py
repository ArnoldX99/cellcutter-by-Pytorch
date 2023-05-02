from os.path import join
import time
import tifffile
import tensorflow as tf
import numpy as np
import cellcutter
import cellcutter.utils
from skimage.color import label2rgb

def miou(label, pred, n = 2):
  v = 0
  for l in range(n):
    intersect = np.sum((pred == l) * (label == l), axis=(1,2))
    union = np.sum((pred==l), axis=(1,2)) + np.sum((label==l), axis=(1,2)) - intersect
    v += (intersect / (union + 0.01)).mean()
  return v / n

def train(train_dataset, epochs=50):
  start = time.time()
  model = cellcutter.UNet4(bn=True)
  cellcutter.train_self_supervised(train_dataset, model, n_epochs = epochs)
  print('Elapsed time: %f'%(time.time() - start))
  return model

DATADIR = join(PROJ, 'data')
np.set_printoptions(precision=4)

files = ['a1data.npz', 'a2data.npz', 'a3data.npz']
data = [np.load(join(DATADIR, f))['data'] for f in files]

print('Training three models with FL data')
dataset = [cellcutter.Dataset(d[...,0], d[...,3], mask_img = d[...,4]) for d in data]
models = [train(ds) for ds in dataset]
preds = [tf.sigmoid(m(ds.patches)).numpy().squeeze() > 0.5 for m,ds in zip(models, dataset)]

for i, m in enumerate(models):
  print('Model #%i :'%(i))
  for j,ds in enumerate(dataset):
     new_pred = m(ds.patches).numpy().squeeze() > 0.5
     v = miou(preds[j], new_pred)
     print('\tmIOU against dataset %i: %f'%(j, v))

labels = np.stack([np.stack([label2rgb(cellcutter.utils.draw_label(ds, m, np.zeros((1750,1750))), bg_label=0) for ds in dataset]) for m in models])
labels = (labels * 255).astype(np.uint8)
tifffile.imwrite('transfer_FL_label.tif', labels.reshape((9,1750,1750,3)))
border = np.stack([np.stack([cellcutter.utils.draw_border(ds, m, np.zeros((1750,1750))) for ds in dataset]) for m in models])
tifffile.imwrite('transfer_FL_border.tif', border)

print('Training three models with BF data')
dataset = [cellcutter.Dataset(d[...,1], d[...,3], mask_img = d[...,5]) for d in data]
models = [train(ds) for ds in dataset]
preds = [ tf.sigmoid(m(ds.patches)).numpy().squeeze() > 0.5 for m,ds in zip(models, dataset)]

for i, m in enumerate(models):
  print('Model #%i :'%(i))
  for j,ds in enumerate(dataset):
     new_pred = tf.sigmoid(m(ds.patches)).numpy().squeeze() > 0.5
     v = miou(preds[j], new_pred)
     print('\tmIOU against dataset %i: %f'%(j, v))

labels = np.stack([np.stack([label2rgb(cellcutter.utils.draw_label(ds, m, np.zeros((1750,1750))), bg_label=0) for ds in dataset]) for m in models])
labels = (labels * 255).astype(np.uint8)
tifffile.imwrite('transfer_BF_label.tif', labels)
border = np.stack([np.stack([cellcutter.utils.draw_border(ds, m, np.zeros((1750,1750))) for ds in dataset]) for m in models])
tifffile.imwrite('transfer_BF_border.tif', border)
