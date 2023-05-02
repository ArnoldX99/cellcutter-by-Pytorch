# Cellcutter notebook by GPU version
Demo is avaliable by https://colab.research.google.com/drive/1siXnvlopsYA4XpzXvzlH_no1-zCbcBDL?usp=sharing
This article comes from Din, N.U., Yu, J. Training a deep learning model for single-cell segmentation without manual annotation. Sci Rep 11, 23995 (2021). https://doi.org/10.1038/s41598-021-03299-4

Here is only a jupyternotebook to present the result from authors' work by using GPU.

First of all we need to download the specific packages/git clone we need in this task:

```python
!pip install pymaxflow
!git clone https://github.com/jiyuuchc/cellcutter.git
```

Then we need to load the cell image data(note: for this project's data you can also choose'a2data''a3data' for the test):
```python
# Load some image data

data = np.load('cellcutter/data/a1data.npz')
train_data = data['data']
input_img = train_data[...,0]
nucleus_img = train_data[..., 2]

# Check the images
fig, ax = plt.subplots(1,2)
ax[0].imshow(input_img[100:300,200:400])
ax[0].axis('off')
ax[1].imshow(nucleus_img[100:300,200:400])
ax[1].axis('off')
```

![image](https://user-images.githubusercontent.com/64125777/234941394-8fb41028-2713-4a91-abd8-77aa688a6110.png)

The next step is to define the area of anlysis by **graphcut**:

```python
## Defined the area of analysis
mask = cellcutter.utils.graph_cut(input_img, prior = 0.985, max_weight=10, sigma = 0.03)

#check results
fig, ax = plt.subplots(1,2)
ax[0].imshow(input_img[100:300,200:400])
ax[0].axis('off')
ax[1].imshow(mask[100:300,200:400])
ax[1].axis('off')
```
![image](https://user-images.githubusercontent.com/64125777/234942749-2825ecc6-dde0-47f5-ad34-02a5963b6cb6.png)

* The function `graph_cut` takes an input image and returns a binary mask that separates the pixels into two classes: foreground and background.

* The `prior` parameter is a coefficient that controls the a *priori probability of a pixel* being in the foreground. 

* The `max_weight` parameter controls the *maximum edge weight* between adjacent pixels in the graph. In graph cut algorithms, the image is represented as a graph, where nodes correspond to pixels and edges correspond to the similarity between adjacent pixels. The `max_weight` parameter sets an upper bound on the edge weights, which can help to prevent the algorithm from over-segmenting the image.

* The `sigma` parameter is a smoothing parameter that controls the *strength of the smoothing* applied to the input image before graph construction. It determines the size of the Gaussian blur kernel used to smooth the image. A larger `sigma` value will result in a smoother image, while a smaller value will preserve more of the image's detail.



Now we start to train our model: 

```py
# CNN segmentation with epochs of 30
dataset = cellcutter.Dataset(input_img, markers, mask_img = ~mask) # actually need the inverse of the mask

start = time.time()
model = cellcutter.UNet4(bn=True)
cellcutter.train_self_supervised(dataset, model, n_epochs = 30)

print('Elapsed time: %f'%(time.time() - start))

# Check the segmentation results
from skimage.color import label2rgb

label = cellcutter.utils.draw_label(dataset, model, np.zeros_like(input_img, dtype=int))
rgb = label2rgb(label, bg_label = 0)
border = cellcutter.utils.draw_border(dataset, model, np.zeros_like(input_img, dtype=int))
```

The training steps will be shown like below:

![image-20230502134452823](C:\Users\10306\AppData\Roaming\Typora\typora-user-images\image-20230502134452823.png)

We can see the loss function will be negative by definition.

Then we want to show our output figure: 

```python
# Check the segmentation results
from skimage.color import label2rgb

label = cellcutter.utils.draw_label(dataset, model, np.zeros_like(input_img, dtype=int))
rgb = label2rgb(label, bg_label = 0)
border = cellcutter.utils.draw_border(dataset, model, np.zeros_like(input_img, dtype=int))
fig, ax = plt.subplots(1,3,figsize = (10,7))
ax[0].imshow(input_img[100:300,200:400])
ax[0].axis('off')
ax[1].imshow(rgb[100:300,200:400])
ax[1].axis('off')
ax[2].imshow(border[100:300,200:400])
ax[2].axis('off')
plt.show()
```



Here we list results of epochs = 10 and epochs = 30 :

![image](https://user-images.githubusercontent.com/64125777/235638767-ebb666af-da26-479d-abc7-026ef236c579.png)

We can see the masks and segmentation result get clearer with the increase of epochs.
