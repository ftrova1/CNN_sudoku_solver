import cv2
import numpy as np
import operator


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def pre_process (original, skip_dilate=False):
    """Sfocatura gaussiana (GaussianBlur) + Adaptive Thresholding + Dilatazione per estrarre i contorni principali dell'immagine"""

    image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #Sfocatura gaussiana con kernel size = 11x11
    processed = cv2.GaussianBlur(image.copy(), (11,11), 0)

    #Adaptive Thresholding con 11 pixel di Nearest Neighbour
    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #Inverto i valori dei pixel in modo che i bordi neri non abbiamo valore = 0
    processed = cv2.bitwise_not(processed,processed)

    if not(skip_dilate):
        #Dilatazione dell'imagine in modo da rendere i bordi più spessi e riconoscibili
        kernel = np.array([[0.,1.,0.],[1.,1.,1.],[0.,1.,0.]], np.uint8)
        processed = cv2.dilate(processed, kernel)

    return processed

def get_corners(image):
    """Estrae le coordinate degli spigoli del poligono più grande che viene rilevato"""
    #Trova i contorni
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Ordina per area, il poligono più grande sarà all'indice 0 della lista
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    poly = contours[0]

    #operator.itemgetter ci permette di ottenere l'indice
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in poly]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in poly]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in poly]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in poly]), key=operator.itemgetter(1))

    return [poly[top_left][0], poly[top_right][0], poly[bottom_right][0], poly[bottom_left][0]]

def crop_and_warp(img,poly):
    top_left, top_right , bottom_right, bottom_left = poly[0], poly[1], poly[2], poly[3]
    src = np.array([top_left, top_right , bottom_right, bottom_left], dtype = 'float32')

    side = max( [np.linalg.norm(bottom_right-bottom_left),
                np.linalg.norm(top_right-top_left),
                np.linalg.norm(bottom_right-top_right),
                np.linalg.norm(bottom_left-top_left)])

    dst = np.array([[0,0], [side -1 , 0], [side -1, side - 1], [0, side - 1]], dtype = 'float32')

    m = cv2.getPerspectiveTransform(src,dst)

    return cv2.warpPerspective(img,m,(int(side),int(side)))

def traccia_sagoma_griglia(img):
    squares = []
    side = img.shape[:1]
    side = side[0] / 9
    for i in range(9):
        for j in range(9):
            p1 = (j * side, i * side)
            p2 = ((j+1) * side, (i+1) * side)
            squares.append((p1,p2))

    return squares

def cut_from_rect(img, rect):
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

def scale_and_centre(img, size, margin=0, background=0):
    h, w = img.shape[:2]

    def centre_pad(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1,side2

    def scale (r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size,size))

def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
	"""
	Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
	connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
	"""
	img = inp_img.copy()  # Copy the image, leaving the original untouched
	height, width = img.shape[:2]

	max_area = 0
	seed_point = (None, None)

	if scan_tl is None:
		scan_tl = [0, 0]

	if scan_br is None:
		scan_br = [width, height]

	# Loop through the image
	for x in range(scan_tl[0], scan_br[0]):
		for y in range(scan_tl[1], scan_br[1]):
			# Only operate on light or white squares
			if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
				area = cv2.floodFill(img, None, (x, y), 64)
				if area[0] > max_area:  # Gets the maximum bound area which should be the grid
					max_area = area[0]
					seed_point = (x, y)

	# Colour everything grey (compensates for features outside of our middle scanning range
	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 255 and x < width and y < height:
				cv2.floodFill(img, None, (x, y), 64)

	mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

	# Highlight the main feature
	if all([p is not None for p in seed_point]):
		cv2.floodFill(img, mask, seed_point, 255)

	top, bottom, left, right = height, 0, width, 0

	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 64:  # Hide anything that isn't the main feature
				cv2.floodFill(img, mask, (x, y), 0)

			# Find the bounding parameters
			if img.item(y, x) == 255:
				top = y if y < top else top
				bottom = y if y > bottom else bottom
				left = x if x < left else left
				right = x if x > right else right

	bbox = [[left, top], [right, bottom]]
	return img, np.array(bbox, dtype='float32'), seed_point


def extract_digit(img, rect, size):
    """Extracts a digit (if one exists) from a Sudoku square."""

    digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

    # Use fill feature finding to get the largest feature in middle of the box
    # Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)

    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
    	return scale_and_centre(digit, size, 4)
    else:
    	return np.zeros((size, size), np.uint8)

def get_digits(img, squares, size):
    """Extracts digits from their cells and builds an array"""
    digits = []
    img = pre_process(img.copy(), skip_dilate=True)
    for square in squares:
    	digits.append(extract_digit(img, square, size))
    return digits



img = cv2.imread("sudoku_2.jpg")
img = image_resize(img, height = 500)
cv2.imshow("Original Image", img)

processed = pre_process(img)
#cv2.imshow("Pre Processed Image", processed)

corners = get_corners(processed)
cropped = crop_and_warp(img,corners)
#cv2.imshow("Cropped", cropped)

squares = traccia_sagoma_griglia(cropped)

digits = get_digits(cropped, squares, 28)

def show_digits(digits, colour=255):
    """Shows list of 81 extracted digits in a grid format"""
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]

    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    cv2.imshow('digits',np.concatenate(rows))

show_digits(digits)
# for corner in corners:
#     cv2.circle(img,(int(corner[0]),int(corner[1])), 6, (0,0,255))

# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# max_area = 0
# c = 0
# for i in contours:
#         area = cv2.contourArea(i)
#         #if area > 1000:
#         if area > max_area:
#             max_area = area
#             best_cnt = i
#             provola = c
#         c+=1
#
# image = cv2.drawContours(image, contours, provola, (0, 255, 0), 3)
# print ("Il bordo esterno ha indice = " + str(c) + "\n")
#
# mask = np.zeros((gray.shape),np.uint8)
# cv2.drawContours(mask,[best_cnt],0,255,-1)
# cv2.drawContours(mask,[best_cnt],0,0,2)
# cv2.imshow("mask", mask)
# out = np.zeros_like(gray)
# out[mask == 255] = gray[mask == 255]
# cv2.imshow("New image", out)
#
# blur = cv2.GaussianBlur(out, (5,5), 0)
# cv2.imshow("blur1", blur)
#
# thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
# cv2.imshow("thresh1", thresh)
#
#     # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # hierarchy = hierarchy[0]
#     #
#     # c = 0
#     # for component in zip(contours, hierarchy):
#     #     currentContour = component[0]
#     #     currentHierarchy = component[1]
#     #     #hierarchy = Next - Previous - First Child - First Parent
#     #     area = cv2.contourArea(currentContour)
#     #     #if area > 1000/2:
#     #     if currentHierarchy[3] == 0:
#     #         print (currentHierarchy)
#     #         image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
#     #     c+=1
#
# cv2.imshow("Final image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
