# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import cv2

class MeasureApp(object):
	
	src_path          = 'imgs/prep/flor-1.jpg'
	src_width         = 12
	
	show_image        = False
	show_bounding_box = False
	
	hog_width         = 200
	hog_winstride     = (2,2)
	hog_padding       = (8,8)
	hog_scale         = 1.05
	hog_mean_shift    = False

	img_src           = None
	img_gray          = None
	img_edged         = None
	cnts              = None
	px_per_metric     = None


	def __init__(self):
		self.parseArgs()
		self.detectPerson()
		self.getContours()
		self.processContours()

	def parseArgs(self):
		ap = argparse.ArgumentParser()

		ap.add_argument(
			"-i", "--image", 
			required = False, help = "Input image path"
		)

		ap.add_argument(
			"-w", "--width", 
			type = float, 
			required = False, help = "Width of the left-most object in cm (marker)"
		)
	
		args = vars(ap.parse_args())

		self.src_path  = self.src_path if args["image"] == None else args["image"]
		self.src_width = self.src_width if args["width"] == None else args["width"]

	def detectPerson(self):
		hog = cv2.HOGDescriptor()
		hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		hogParams = {
			'winStride': self.hog_winstride, 
			'padding': self.hog_padding,
			'scale': self.hog_scale,
			'useMeanshiftGrouping': self.hog_mean_shift
		}

		img  = cv2.imread(self.src_path)
		img  = imutils.resize(img, width=min(self.hog_width, img.shape[1]))
		img_all = img.copy()
		img_nms = img.copy()

		(rects, weights) = hog.detectMultiScale(img, **hogParams)

		# draw all rects
		for (x, y, w, h) in rects:
			cv2.rectangle(img_all, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# draw rects left after applying non max suppresion
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick  = non_max_suppression(rects, probs=None, overlapThresh=0.65)
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(img_nms, (xA, yA), (xB, yB), (0, 255, 0), 2)

		cv2.imshow("Measure App 1", img_all)
		cv2.imshow("Measure App 2", img_nms)
		cv2.waitKey(0)

	def getContours(self):
		# load the image, convert it to grayscale, and blur it slightly
		self.img_src = cv2.imread(self.src_path)
		self.img_gray = cv2.cvtColor(self.img_src, cv2.COLOR_BGR2GRAY)
		self.img_gray = cv2.GaussianBlur(self.img_gray, (5, 5), 0)

		# perform edge detection, then perform a dilation + erosion to
		# close gaps in between object edges
		self.img_edged = self.img_gray.copy()
		self.img_edged = cv2.Canny(self.img_edged, 10, 70, self.img_edged, 3, True)
		self.img_edged = cv2.dilate(self.img_edged, None, iterations=1)
		self.img_edged = cv2.erode(self.img_edged, None, iterations=1)

		# find contours in the edge map
		cnts = cv2.findContours(self.img_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]

		# sort the contours from left-to-right
		(cnts, _) = contours.sort_contours(cnts)
		self.cnts = cnts

	def processContours(self):
		for c in self.cnts:
			# if the contour is not sufficiently large, ignore it
			if cv2.contourArea(c) < 100:
				continue
			
			# draw image and/or edges
			if self.show_image:
				img_out = cv2.addWeighted(
					self.addAlphaChannel(self.img_src), 1, 
					self.addAlphaChannel(self.img_edged, True), 1, 0
				)
			else:
				img_out = self.addAlphaChannel(self.img_edged, True)

			# draw contours
			self.drawContourPoints(c, img_out)

			if self.show_bounding_box:
				# draw bounding box
				self.drawContourBoundingBox(c, img_out)

			cv2.imshow("MeasureApp", img_out)
			cv2.setMouseCallback("MeasureApp", self.onClick)
			cv2.waitKey(0)

		cv2.destroyAllWindows

	def drawContourPoints(self, c, img_out):
		# draw contour points
		for p in c:
			px,py = p[0]
			cv2.circle(img_out, (int(px), int(py)), 1, (0, 255, 0), -1)

	def drawContourBoundingBox(self, c, img_out):
		# compute the rotated bounding box of the contour
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")

		# order the points in the contour such that they appear in top-left, 
		# top-right, bottom-right, and bottom-left order, then draw the 
		# outline of the rotated bounding box
		box = perspective.order_points(box)
		cv2.drawContours(img_out, [box.astype("int")], -1, (0, 0, 0), 1)

		# loop over the original points and draw them
		for (x, y) in box:
			cv2.circle(img_out, (int(x), int(y)), 2, (0, 0, 255), -1)

		# unpack the ordered bounding box, then compute the midpoint
		# between the top-left and top-right coordinates, followed by
		# the midpoint between bottom-left and bottom-right coordinates
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = self.getMidpoint(tl, tr)
		(blbrX, blbrY) = self.getMidpoint(bl, br)

		# compute the midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-righ and bottom-right
		(tlblX, tlblY) = self.getMidpoint(tl, bl)
		(trbrX, trbrY) = self.getMidpoint(tr, br)

		# draw the midpoints on the image
		cv2.circle(img_out, (int(tltrX), int(tltrY)), 2, (0, 0, 255   ), -1)
		cv2.circle(img_out, (int(blbrX), int(blbrY)), 2, (0, 0, 255), -1)
		cv2.circle(img_out, (int(tlblX), int(tlblY)), 2, (0, 0, 255), -1)
		cv2.circle(img_out, (int(trbrX), int(trbrY)), 2, (0, 0, 255), -1)

		# draw lines between the midpoints
		cv2.line(img_out, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (0, 0, 0), 1)
		cv2.line(img_out, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 0, 0), 1)

		# compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

		# if the pixels per metric has not been initialized, then
		# compute it as the ratio of pixels to supplied metric
		# (in this case, inches)
		if self.px_per_metric is None:
			self.px_per_metric = dB / self.src_width

		# compute the size of the object
		dimA = dA / self.px_per_metric
		dimB = dB / self.px_per_metric

		# draw the object sizes on the image
		cv2.putText(img_out, "{:.1f}cm".format(dimA),
			(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)
		cv2.putText(img_out, "{:.1f}cm".format(dimB),
			(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)

	''' EVENTS '''

	def onClick(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			print x, y

	''' HELPERS '''

	def getMidpoint(self, ptA, ptB):
		return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

	def addAlphaChannel(self, src, is_grayscale=False):
		dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) if is_grayscale else src
		tmp = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
		(_, alpha) = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
		(b, g, r) = cv2.split(dst)
		rgba = [b, g, r, alpha]
		dst = cv2.merge(rgba,4)
		return dst

app = MeasureApp()