how to use: 

Run setup.py to extracts and remove best_model.zip 

user should install all dependencies themselves
the script requires:
-	pytorch
-	torchvision
-	scikitlearn

remove existing image files in the folder "test" in "for_pred" folder then put 1.5T brain slice MRI scans into the folder.
run prediction.py using a python terminal with the following template

!python DenseNetMRIPred/prediction.py --figure_size (input your figure size here) --font_size (input your font size here) 

	- this step could be done by making an .ipynb file inside densenetmripred, but when doing so remove the DenseNETMRIPred/ infront of prediction.py, 
	- other option is to make jupyter notebook just outside the DenseNetMRIPred folder / in the same folder as DenseNETMRIPred
	- go into terminal and cd to DenseNetMRIPred, then !python prediction.py --figure_size (input) --font_size (input) 


this will return an image with all images in for_pred loaded out and labeled with a prediction 
prediction includes 4 classes currently mildly demented, moderately demented, non demented, very mildly demented
