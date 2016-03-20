Stop Sign Detection
=====================
Prototype
-----------
Some quick prototype code to experiment with some old computer vision concepts and get some first time exposure to Open CV.  This was one my early python projects so it's probably not very Pythonic.

This code takes a "prototype" stop sign (stopPrototype.png) and creates a pyramid out of it by downsampling (read more here: https://en.wikipedia.org/wiki/Pyramid_(image_processing)) then it slides the pyramid one slice at a time over the target image, computing the mean square error. The slice with the mean square error is determined to be the stop sign. Actually works ok!

Run it like this: *python detectStopSigns.py -p stopPrototype.png -i Stop\ Sign\ Dataset/3.jpg*

![Screenshot of Stop Sign Detector](https://raw.githubusercontent.com/mbasilyan/Stop-Sign-Detection/master/Screenshot.png)

Dataset
--------
"Stop Sign Dataset" folder contains a data set of 100 images with stop signs and 100 images without stop signs (see labels.tsv). In the future -- If I ever get to it --  I am planning to use this dataset to evaluate the performance of a stop sign classifier.
