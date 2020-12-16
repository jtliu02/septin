# septin

## technologies
* fiji imagej ridge detection plugin (https://imagej.net/Ridge_Detection)
  * download fiji at https://fiji.sc
  * open fiji: help -> update
    * click "manage update sites"
    * check "biomedgroup"
    * close and "apply changes"
* anaconda jupyter notebook (python 3.8.6)
* python packages
  * gsd 2.4.0
  * matplotlib 3.3.3
  * numpy 1.19.4
  * pandas 1.1.5

## imagej instructions
* file -> open .tif microscopy image
* process -> subtract background (rolling ball radius 50.0 pixels)
  * save the new .tif image for analysis in jupyter notebook (put in same folder as .ipynb)
* image -> type -> 8-bit
* plugins -> ridge detection
* parameters
  * line width 0.6
  * high contrast 85
  * low contrast 0
  * check:
    * correct position
    * estimate width
    * extend line
    * display results
    * add to manager
  * method for overlap resolution: NONE
* save "results" (.csv file) in same folder as .tif and .ipynb

## analysis

### bundling
take the total fluorescence across each point along every detected filament. plot the histogram of cross-sectional fluorescence intensity.

### local orientation alignment
calculated by nematic correlation length (https://arxiv.org/pdf/2001.03804.pdf). pick random pairs of points and measure cos(2&Delta;&theta;) of each pair. <cos(2&Delta;&theta;)> vs r can be fitted with y = Ae^bx, and nematic correlation length is -1/b. longer nematic correlation length means more locally aligned.

### flexibility
calculated by persistence length. pick random filaments to measure cos(&Delta;&theta;) between every two points in the filament. <cos(&Delta;&theta;)> vs r can be fitted with y = e^(-x/p), where p is persistence length. longer persistence length means stiffer filament.
