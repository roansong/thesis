70 tiffs are stored in the tiffset - 5 classes with 14 images each. This subset was used for the sake of computation time and demonstration.
tiffset/filenames.txt stores the names of the files, as well as their classification labels.
The only variables worth fiddling with are at the top of the script;
'location' is the name of the folder in which the tiff files are stored
'num_classes' is the number of different target classes in the set.
the labels are one-hot, i.e. [1 0 0 0 0] is class 1, [0 1 0 0 0] is class 2, etc.
The running time is not as fast as the MATLAB implementation was. I suspect this is due to my choice of padding all the images with zeroes to match the biggest images in the set (177x178), whereas in my MATLAB implementation I shrunk each image down to the size of whatever image was input. This can be amended fairly easily, and perhaps will be at a later stage.

You should get an accuracy of 58/70, i.e. 84.2% (resizing the images smaller instead of padding bigger gave 56/70)
