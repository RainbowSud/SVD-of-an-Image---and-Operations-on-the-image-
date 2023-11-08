import numpy as np
from PIL import Image

# Import the image here and convert to greyscale
image = Image.open('waterfall.jpg').convert('L')

# Convert the image to an Array
imageAsArray = np.asarray(image)

# SVD on the array
U, S, V = np.linalg.svd(imageAsArray)

# Rank Approximations, shows rank 1, 5, 10, 50, 100, 250, 500 approximations
# Then it shows the actual image
R1approx = np.matrix(U[:,:1]) * np.diag(S[:1]) * np.matrix(V[:1,:])
R1approxImage = Image.fromarray(R1approx)
R1approxImage.show()

R5approx = np.matrix(U[:,:5]) * np.diag(S[:5]) * np.matrix(V[:5,:])
R5approxImage = Image.fromarray(R5approx)
R5approxImage.show()

R10approx = np.matrix(U[:,:10]) * np.diag(S[:10]) * np.matrix(V[:10,:])
R10approxImage = Image.fromarray(R10approx)
R10approxImage.show()

R50approx = np.matrix(U[:,:50]) * np.diag(S[:50]) * np.matrix(V[:50,:])
R50approxImage = Image.fromarray(R50approx)
R50approxImage.show()

R100approx = np.matrix(U[:,:100]) * np.diag(S[:100]) * np.matrix(V[:100,:])
R100approxImage = Image.fromarray(R100approx)
R100approxImage.show()

R250approx = np.matrix(U[:,:250]) * np.diag(S[:250]) * np.matrix(V[:250,:])
R250approxImage = Image.fromarray(R250approx)
R250approxImage.show()

R500approx = np.matrix(U[:,:500]) * np.diag(S[:500]) * np.matrix(V[:500,:])
R500approxImage = Image.fromarray(R500approx)
R500approxImage.show()

image.show()