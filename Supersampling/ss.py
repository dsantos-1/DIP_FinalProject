# Names: Alex Sander Ribeiro da Silva, Danilo Marques Araujo dos Santos
# USP Number: 9779350, 8598670
# Course code: SCC0251
# Year/Semester: 2019-1

import argparse, imageio, os
import numpy as np

# Applies the supersampling algorithm to an image
def ss(f, n):
	fout=np.zeros((f.shape[0]//n, f.shape[1]//n, 3), dtype=np.uint8)
	
	for i in range(0, f.shape[0]-n//2, n):
		for j in range(0, f.shape[1]-n//2, n):
			for k in range(3):
				fout[i//n,j//n,k]=np.round(np.average(f[:,:,k][i:i+n, j:j+n]))

	return fout

parser=argparse.ArgumentParser(description='Applies the supersampling algorithm to a colored image')
parser.add_argument('ii_name', help='the name of the input image')
parser.add_argument('--n', type=int, help='how many pixels are to be merged (n x n square; default 2)')
args=parser.parse_args()

# Checks if the input image exists
try:
	in_img=imageio.imread(args.ii_name)
except FileNotFoundError:
	exit('Error: file '+args.ii_name+' not found')

print('[+] File '+args.ii_name+' (%d, %d) successfully opened' % (in_img.shape[0], in_img.shape[1]))
print('[+] Starting the supersampling algorithm with n = ', end='', flush=True)

# Applies the supersampling algorithm to the image
if(args.n!=None):
	print('%d... ' % args.n, end='', flush=True)
	out_img=ss(in_img, args.n)
else:
	print('%d... ' % 2, end='', flush=True)
	out_img=ss(in_img, 2)
	
print('OK')

# Creates the name of the output image
oi_name=args.ii_name.replace('.', '_ss.')

# Writes the output image to the disk
imageio.imwrite(oi_name, out_img)

print('[+] File '+oi_name+' (%d, %d) successfully written to %s' % (out_img.shape[0], out_img.shape[1], os.getcwd()))