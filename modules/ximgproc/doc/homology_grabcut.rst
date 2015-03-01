Homology Grabcut
===========

homology_grabcut
----------------
.. ocv:function::homology_grabcut ( InputArray img, InputArray filters, InputOutputArray mask, Rect _rect, int IterCount=1 )

The function performs the following operations on an image, in this order:
1) The original image is transformed into grayscale and expanded into 14 channels
2) The channels 1-13 have filters applied to them, each filter applied to a different copy
	of the original image. Channel 0 remains unchanged.
3) The image material is then shrunk 100 times, each 100 pixels of original image are used
	to create 1 shrunk pixels. Each pixel stores its mean and standard deviation, raising
	the amount of channels to a total of 28.
4) A standard grabcut algorythm is then ran on the 28 channel image. The answer is stored
	in temporary shrunk mask variable.
5) The answer is enlarged 100 times, each pixels classification expanded onto every single
	one it was made of in step 3). The final answer is loaded into given mask.

---------------
.. ocv:function::create_filters( OutputArray filters, int size=49 );

Function that creates a filter bank of 13 Schmid filters in accordance to:
	http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html
The filters are of given size, stored in 13-channel square matrix given by
parameter filters.