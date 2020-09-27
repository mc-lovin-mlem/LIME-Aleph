import numpy as np

def quadratic(image, n_quads=8):
	output = np.zeros((image.shape[0], image.shape[1]), dtype=int)
	
	quad_size = image.shape[0] // n_quads
	#print(quad_size)
	
	sp_index = 0
	for offset_x in range(n_quads):
		for offset_y in range(n_quads):
			pixel_offset_x = offset_x * quad_size
			pixel_offset_y = offset_y * quad_size
			for x in range(quad_size):
				for y in range(quad_size):
					output[pixel_offset_x+x][pixel_offset_y+y] = sp_index
			sp_index += 1
	
	#print(output)
	return output
