# vision1
quantization A interpolation
#library
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
#function to show array 

def imshow(*args, figsize=10, to_rgb=True, title=None, fontsize=12):
    if isinstance(figsize, int):
        figsize = (figsize, figsize)
    images = args[0] if type(args[0]) is list else list(args)
    if to_rgb:
        images = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), images))
    if title is not None:
        assert len(title) == len(images), "Please provide a title for each image."
    plt.figure(figsize=figsize)
    for i in range(1, len(images)+1):
        plt.subplot(1, len(images), i)
        if title is not None:
            plt.title(title[i-1], fontsize=fontsize)
        plt.imshow(images[i-1])
        plt.axis('off')

#function for quantize
def quantize_simulation(image, n_bits):
    coeff = 2**8 // 2**n_bits
    return (image // coeff) * coeff
#get image and save in an array
img = cv2.imread('Elaine.bmp',cv2.IMREAD_GRAYSCALE)
plt.imshow(img)
plt.show()
#quantize image using function
img4bit = quantize_simulation(img, 2)
img8bit = quantize_simulation(img, 3)
img16bit = quantize_simulation(img, 4)
img32bit = quantize_simulation(img, 5)
img64bit = quantize_simulation(img, 6)
img128bit = quantize_simulation(img, 7)
images = [img4bit,img8bit,img16bit,img32bit,img64bit,img128bit]

x = 2
#show result
for image in images:
  x = x*2
  equ = cv2.equalizeHist(image)
  print('number of gray-level=',x)
  imshow(image, equ, title=['Original'+str(x)+'MSE='+str(np.square(np.subtract(image, img)).mean()), 'Equalized'+str(x)+'MSE='+str(np.square(np.subtract(equ, img)).mean())])

  # Plot histograms
  plt.figure(figsize=(10, 4))

  hist1, _ = np.histogram(image, 256, (0, 255))
  plt.subplot(1, 2, 1)
  plt.title('Original')
  plt.plot(list(range(256)), hist1)

  hist2, _ = np.histogram(equ, 256, (0, 255))
  plt.subplot(1, 2, 2)
  plt.title('Equalized')
  plt.plot(list(range(256)), hist2)

  plt.show()
 ![image](https://user-images.githubusercontent.com/45328431/219899333-84242850-40f5-4ce1-939b-30e84ae472df.png)

number of gray-level= 4
 ![image](https://user-images.githubusercontent.com/45328431/219899319-8f78d80c-d813-4981-8563-2b4a624c4940.png)

 ![image](https://user-images.githubusercontent.com/45328431/219899315-b06f6693-326b-43f9-a2e0-d0db4266532f.png)

number of gray-level= 8
 ![image](https://user-images.githubusercontent.com/45328431/219899308-1ed78d80-fae8-4b0b-9d95-0c7899d02641.png)

 ![image](https://user-images.githubusercontent.com/45328431/219899303-030061bf-1d6e-4076-b30c-b06f5f25b6a5.png)

number of gray-level= 16
 ![image](https://user-images.githubusercontent.com/45328431/219899298-27ead837-e2dd-4470-ab4e-29a575543a01.png)

 ![image](https://user-images.githubusercontent.com/45328431/219899292-30c0e8b6-f3e3-4d9c-9c55-87222f7b9224.png)

number of gray-level= 32
 ![image](https://user-images.githubusercontent.com/45328431/219899283-7976d3a4-b3e8-4e4a-b280-16b54077f998.png)

 ![image](https://user-images.githubusercontent.com/45328431/219899274-cf8c3890-f56c-4f94-9f30-35c8997eb4b0.png)

number of gray-level= 64
 ![image](https://user-images.githubusercontent.com/45328431/219899260-c79f9437-2f63-404d-9962-6b7a6cf34c7b.png)

![image](https://user-images.githubusercontent.com/45328431/219899264-fbf6d8fc-7f24-425d-83ac-d88e8aef33a7.png)

number of gray-level= 128
 ![image](https://user-images.githubusercontent.com/45328431/219899242-fc4a26ed-3ad0-4c48-b458-e1c5c729d87a.png)

	 ![image](https://user-images.githubusercontent.com/45328431/219899235-a0254adb-761a-4031-87ae-1daef7040b25.png)

1.1.2
def average(img):
  avg = np.zeros_like(img, dtype='uint8')
  #height and width of array 
  M = len(img)-1
  N = len(img[0])-1
  #max & min if for overflow
  for i in range(M+1):
      for j in range(N+1):
        #avg for 9 nearest
          avg[i][j]=np.round((img[max(i-1,0)][max(j-1,0)] +
                        img[max(i-1,0)][j] +
                        img[i][max(j-1,0)] + 
                        img[i][j] +
                        img[min(i+1,M)][min(j+1,N)] +
                        img[min(i+1,M)][j] +
                        img[i][min(j+1,N)] + 
                        img[max(i-1,0)][min(j+1,N)] + 
                        img[min(i+1,M)][max(j-1,0)]) / 9)
  return avg

def bilinear(x,y,z,w,i,j):
#  x = (i-1)*a + (j-1)*b + (i-1)(j-1)*c + d
#  y = (i-1)*a + (j+1)*b + (i-1)(j+1)*c + d
#  z = (i+1)*a + (j-1)*b + (i+1)(j-1)*c + d
#  w = (i+1)*a + (j+1)*b + (i+1)(j+1)*c + d
  A = np.array([[i-1,j-1,(i-1)*(j-1),1],[i-1,j+1,(i-1)*(j+1),1],[i+1,j-1,(i+1)*(j-1),1],[i+1,j+1,(i+1)*(j+1),1]])
  B = np.array([x,y,z,w])
  ans = np.linalg.solve(A,B)
  return ans[0]*i + ans[1]*j + ans[2]*i*j + ans[3]
img = cv2.imread('Goldhill.bmp',cv2.IMREAD_GRAYSCALE)
imshow(img)

#downsample with ignore odd cols and odd rows
downsample1 = img[0::2,0::2]
imshow(downsample1)

#average filter before down sample
avg = average(img)
print('MSE=' + str((np.square(avg - img)).mean()))
imshow(avg)

#get downsample from average of image
downsample2 = avg[0::2,0::2]
imshow(downsample2)
 ![image](https://user-images.githubusercontent.com/45328431/219899198-cf1254e8-181b-4831-a5b8-b7873be7dad6.png)
 MSE=111.26028060913086  

![image](https://user-images.githubusercontent.com/45328431/219899204-e58775f2-c765-43b2-b663-986c1d866391.png)
![image](https://user-images.githubusercontent.com/45328431/219899216-d6cd1f7b-98ed-4273-b716-cf4914553e78.png)
![image](https://user-images.githubusercontent.com/45328431/219899221-b6480e5c-ba8c-46d6-9cb9-7137a0cad41e.png)

 
#up sample of downsample without average filter
#pixel interpolation

upsample1_1 = np.zeros_like(img, dtype='uint8')
for i in range(len(downsample1)):
  for j in range(len(downsample1[0])):
    upsample1_1[i*2][j*2] = downsample1[i][j]
    upsample1_1[i*2+1][j*2+1] = downsample1[i][j]

#bilinear interpolation
upsample1_2 = np.zeros_like(img, dtype='uint8')
for i in range(len(downsample1)):
  for j in range(len(downsample1[0])):
    upsample1_2[i*2][j*2] = downsample1[i][j]

for i in range(len(upsample1_2)):
  for j in range(len(upsample1_2[0])):
      if(i%2 == 1 and j%2 == 1):
        upsample1_2[i][j] = bilinear(upsample1_2[i-1][j-1],upsample1_2[i-1][min(j+1,len(upsample1_2[0])-1)],upsample1_2[min(i+1,len(upsample1_2)-1)][j-1],upsample1_2[min(i+1,len(upsample1_2)-1)][min(j+1,len(upsample1_2[0])-1)],i,j)

#up sample of downsample with average filter
#pixel interpolation
upsample2_1 = np.zeros_like(img, dtype='uint8')
for i in range(len(downsample2)):
  for j in range(len(downsample2[0])):
    upsample2_1[i*2][j*2] = downsample2[i][j]
    upsample2_1[i*2+1][j*2+1] = downsample2[i][j]

#bilinear interpolation
upsample2_2 = np.zeros_like(img, dtype='uint8')
for i in range(len(downsample2)):
  for j in range(len(downsample2[0])):
    upsample2_2[i*2][j*2] = downsample2[i][j]

for i in range(len(upsample2_2)):
  for j in range(len(upsample2_2[0])):
      if(i%2 == 1 and j%2 == 1):
        upsample2_2[i][j] = bilinear(upsample2_2[i-1][j-1],upsample2_2[i-1][min(j+1,len(upsample2_2[0])-1)],upsample2_2[min(i+1,len(upsample1_2)-1)][j-1],upsample2_2[min(i+1,len(upsample2_2)-1)][min(j+1,len(upsample2_2[0])-1)],i,j)

imshow(upsample1_1,upsample1_2,title =['MSE=' + str((np.square(upsample1_1 - img)).mean()),'MSE=' + str((np.square(upsample1_2 - img)).mean())])
imshow(upsample2_1,upsample2_2,title =['MSE=' + str((np.square(upsample2_1 - img)).mean()),'MSE=' + str((np.square(upsample2_2 - img)).mean())])
 ![image](https://user-images.githubusercontent.com/45328431/219899174-0d25aaa0-da6a-4fec-b7bb-b0245e54d514.png)

 ![image](https://user-images.githubusercontent.com/45328431/219899164-49d0dae5-8002-4880-8a03-e435ca668ee2.png)


1.1.3
img = cv2.imread('Barbara.bmp',cv2.IMREAD_GRAYSCALE)
imshow(img)
![image](https://user-images.githubusercontent.com/45328431/219899144-e9b49805-4d7b-4285-870c-32de4af2d801.png)
