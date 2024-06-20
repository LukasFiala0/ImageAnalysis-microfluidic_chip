# ImageAnalysis-microfluidic_chip
## Python / OpenCV / Preprocessing / Segmentation / BFS

Computer Vision Project<br/>
Dissolution Kinetics of Alginate Beads in a Microfluidic Chip<br/> 

### Data
two sequences of images taken at given time (dissolution time) for two solvent volume flow rates (3, 5 ml/min)<br/>
2592 x 2048; 96 x 96 dpi; 8 bit depth; .tif <br/>
see data_example.tif<br/>

### Preprocessing
grayscale<br/>
median blur<br/>

### Methods
Segmentation – Thresholding and Region grow<br/>

### Output
again two sequences of images<br/>
gif (see output_threshold/region_grow.png)<br/>
time dependency graphs<br/>




