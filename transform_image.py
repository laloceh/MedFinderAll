#!/usr/local/bin/python2.7
#!/usr/bin/python

#http://www.pythonforbeginners.com/os/python-the-shutil-module

'''
For every image
read it
extract the class
resize it
convert it to .png 
'''
import cv2
import glob
import shutil
import time

def copy_image(origen, imageName, dest):
	shutil.move(origen, dest)
	print 'Copying from %s to %s' % (origen,dest)
	print '1 File copied'

def save_class_dictionary(className, imageName,dictionary):
	#print className, imageName
	if className in dictionary:
		#print dictionary[className]
		listValues = dictionary[className]
		#print listValues
		listValues.append(imageName)
		dictionary[className] = listValues
	else:
		#print 'NO'
		names = []
		names.append(imageName)
		dictionary[className] = names
		
def resize(imagePath, savePath, dictionary, size):

	#load the image
	img = cv2.imread(imagePath)

	# we need to keep in mind aspect ratio so the image does
	# not look skewed or distorted -- therefore, we calculate
	# the ratio of the new image to the old image
	# the new image will be 200px 
	r = float(size) / img.shape[1]
	dim = (size, int(img.shape[0] * r))
	 
	# perform the actual resizing of the image and show it
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

	# write the resized image to disk in PNG format
	cv2.imwrite(savePath, resized)
	
	#  save to dictionary
	imageName = savePath[savePath.rfind("/") + 1:]
	className = imageName[:imageName.rfind("-")]
	time.sleep(1)
	copy_image(savePath, imageName, dest='queries')
	save_class_dictionary(className, imageName, dictionary)

def imageNames(path, dictionary, size):
	total = 0
	data = []
	# use glob to grab the image paths and loop over them
	for imagePath in glob.glob(path + "*.jpg"):
		savePath = imagePath[:-4] + '.png'
		resize(imagePath, savePath, dictionary, size)
		total = total + 1
		#kpt,desc = surfdescriptors.descriptors()
	return total	
	

