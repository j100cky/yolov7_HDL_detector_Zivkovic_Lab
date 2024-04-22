# yolov7-based high-density lipoprotein (HDL) detector

Arguably, the most accurate sizing method for HDL particles currently is transmission electron microscopy. However, conventional image analysis software can be challenged by various image quality issues and can be unpredictable. Here we implemented the yolov7 convolutional neural network for HDL imaging using TEM and particle measurement.

This implementation is based on [yolov7](https://github.com/WongKinYiu/yolov7/tree/u7).


## Instance segmentation

[source code](./seg)

## Tutorial

[Colab Example](https://colab.research.google.com/drive/1Fcne-E3Ykiq-qYkkyXADWhJaMJtPfBs1?usp=sharing)


1.	Mount Google Drive to Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

2.	Download YOLOv7 from the repository: 

```python
# Download YOLOv7 repository and install requirements
!git clone https://github.com/zivkovic-lab/yolov7_HDL_detector
!pip install -r requirements.txt
```

3.	Download the seg branch and python dependencies for the segmentation method

```python
# navigate to yolov7 directory and checkout u7 branch of YOLOv7
%cd /content/drive/yolov7_HDL_detector/
!git checkout 788fe3d

# navigate to seg directory and install python dependencies
%cd /content/drive/yolov7_HDL_detector/seg
!pip install --upgrade pip
!pip install -r requirements.txt
```

4.	Download the weights for HDL prediction

```python
%cd /content/drive/yolov7_HDL_detector
!wget https://github.com/zivkovic-lab/yolov7_HDL_detector/releases/download/v1/HDL_weight_best.pt
```

5.	Predict HDL particles from the image files. 

<pixel_per_nm_value>: a numerical value you obtain from your TEM or calculated from the scale bar of the image. Some common values are 2.33, 3.6.  

<your_image_size>: a numerical value of the size in pixels of your image to be analyzed. The image has to be square with equal length and width. Common values are 512 (optimal), 1024, 2048, or 4096.

```python
sourcePath = "<Address_of_your_image_folder>”
saveName = "<Name_your_result_folder>"
pixel_per_nm = <pixel_per_nm_value>
image_size = <your_image_size>

!python /content/drive/yolov7_HDL_detector/seg/segment/predict.py \
    --weights /content/drive/yolov7_HDL_detector/HDL_weight_best.pt \
    --conf 0.50 \
    --iou-thres 0.1 \
    --imgsz '{image_size}' \
    --line-thickness 1 \
    --source  '{sourcePath}' \
    --save-txt \
    --name {saveName} \
    --pixel-per-nm {pixel_per_nm}
```

The format of the output data is: 
Each row records the data of each unique particle detected by the model.  
Column 1: type of particle. “0” = HDL.  
Column 2: Normalized x-position of the particle’s center.   
Column 3: Normalized y-position of the particle’s center.   
Column 4: Normalized width of a box that contains the selected particle.   
Column 5: Normalized height of a box that contains the selected particle.  
Column 6: The index of the particle selected.   
Column 7: The diameter of the selected particle in pixels.   
Column 8: The diameter of the selected particle in nanometers.   


## Helpful tools: Image cropping

```python
base_path = "<Address_of_images_to_be_cropped>”
output_dir = "<Path_to_save_cropped_images>"
crop_size = <desired_image_size>

for file in glob.glob(base_path+"/*"):
      img = cv2.imread(file)
      h, w, _ = img.shape
      grid_y, grid_x = (h//crop_size, w//crop_size)
      for y in range(grid_y):
        for x in range(grid_x):
          img_cropped = img[(y*crop_size):((y+1)*crop_size), (x*crop_size):((x+1)*crop_size), :]
          output_filename = os.path.join(output_dir, f'{os.path.basename(file)[:-4]}_{y}{x}.jpg')
          cv2.imwrite(output_filename, img_cropped)
          print(f'Processed: {output_filename}')
```

