# Names: Alex Sander Ribeiro da Silva, Danilo Marques Araujo dos Santos
# USP Number: 9779350, 8598670
# Course code: SCC0251
# Year/Semester: 2019-1

import argparse, imageio, os, cv2
import numpy as np

# Normalizes an image to the range [0, 255]
def normalize_image(img):
	# Disables division by zero warning
	np.seterr(divide='ignore', invalid='ignore')

	# Gets the minimum and maximum values of the image
	imax, imin = np.max(img), np.min(img)
	
	# Normalizes the image so that the minimum value is 0 and the maximum value is 1
	img_norm=(img-imin)/(imax-imin)
	
	# Multiplies each pixel of the image by 255 and converts it to integer
	img_norm=(img_norm*255).astype(np.uint8)
	
	return img_norm

def fxaa_conv():
	return np.array([[0.0625, 0.125, 0.0625], [0.0625, 0.125, 0.0625], [0.0625, 0.125, 0.0625]])

def fxaa(f, mask):
	# Pre-Init
	size=f.shape
	new_img=np.zeros((size[0], size[1], 3))
	conv=fxaa_conv()

	# Init
	n, m = conv.shape
	padx=n//2
	pady=m//2
	pimg=np.pad(f, ((padx, padx), (pady, pady), (0, 0)), 'constant')
	
	# For each pixel
	for x in range(size[0]):
		for y in range(size[1]):
			if(mask[x,y]>0):
				# Apply blur effect
				for k in range(3):
					new_img[x,y,k]=np.sum(np.matmul(pimg[x:x+n, y:y+m, k], conv))
	
	# Normalizes new edges
	new_img=normalize_image(new_img)

	# Now simply copies the non-edges to the new image
	for x in range(size[0]):
		for y in range(size[1]):
			if(mask[x,y]==0):
				new_img[x,y]=f[x,y]
	
	return new_img

def toLuma(img):
	# Initialize the output image
	new_img=np.zeros((img.shape[0], img.shape[1]))

	# Calculate the illuminance value for each pixel 
	new_img[:,:]=0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2]

	return new_img

def detectEdges(img):
	# Initialize the edge mask
	edge_mask=np.zeros(img.shape)
	
	# Initialize the area mask
	area_mask=np.zeros(img.shape)
	
	# Filter used to detect edges
	detector=np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
	
	# Detect edges
	edge_mask=cv2.filter2D(img, -1, detector)

	return edge_mask

def areaAroundEdges(img, edge_mask, threshold, areaSize):
	step=areaSize//2
	area_mask=np.zeros((img.shape[0], img.shape[1]))
	
	# Expand edges
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			if(edge_mask[x,y]>threshold):
				area_mask[x-step:x+step, y-step:y+step]=255

	return area_mask
	
# Main function
parser=argparse.ArgumentParser(description='Applies the edge detection algorithm to a colored image')
parser.add_argument('ii_name', help='name of input image (720p)')
parser.add_argument('--threshold', '-t', type=float, help='threshold')
args=parser.parse_args()

# Check if the input image exists
try:
	in_img=imageio.imread(args.ii_name)
except FileNotFoundError:
	exit('Error: file '+args.ii_name+' not found')

print('[!] File '+args.ii_name+' (%d, %d) successfully opened' % (in_img.shape[0], in_img.shape[1]))

# Convert the original image to luma and save the output to a new image
print('[+] Converting the original image to luma... ', end='', flush=True)
luma_img=toLuma(in_img)
print('OK')

# Apply the edge detection algorithm to the image
print('[+] Starting the edge detection algorithm... ', end='', flush=True)
e_mask=detectEdges(luma_img)
print('OK')

# Generate the area mask
print('[+] Generating the area mask (threshold=%.1f)... ' % (args.threshold),  end='', flush=True)
a_mask=areaAroundEdges(luma_img, e_mask, args.threshold, 3)
print('OK')

# Apply the FXAA algorithm
print('[+] Applying the FXAA algorithm... ', end='', flush=True)
out_img_norm=fxaa(in_img, a_mask)
print('OK')

# Creates the name of the output image
oi_name=args.ii_name.replace('.', '_fxaa'+str(args.threshold)+'.')

# Writes the output image to the disk
imageio.imwrite(oi_name, out_img_norm)

print('[!] File '+oi_name+' (%d, %d) successfully written to %s' % (out_img_norm.shape[0], out_img_norm.shape[1], os.getcwd()))