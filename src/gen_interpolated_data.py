import csv
import os
import cv2
import time
import pickle

interpolated_data_file = "../data/toybox_rot_frames_interpolated.csv"
data_path = "/home/sanyald/Documents/AIVAS/Projects/Toybox_frames/Toybox_New_Frame6_Size1920x1080/"
transformations = ['rxminus', 'rxplus', 'ryminus', 'ryplus', 'rzminus', 'rzplus']
show = False
classes = ['airplane', 'ball', 'car', 'cat' , 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon', 'truck']


def crop_image(imagePath, image, l, t, w, h):
	if h < w:
		t_new = max(t - ((w - h) // 2), 0)
		h_new = min(image.shape[0], w)
		l_new = l
		w_new = w
		b_new = t_new + h_new
		if b_new > image.shape[0]:
			t_new = t_new - (b_new - image.shape[0])
	elif w < h:
		t_new = t
		h_new = h
		l_new = max(l - ((h - w) // 2), 0)
		w_new = min(image.shape[1], h)
		r_new = l_new + w_new
		if r_new > image.shape[1]:
			l_new = l_new - (r_new - image.shape[1])
	else:
		t_new = t
		h_new = h
		l_new = l
		w_new = h

	try:
		image_cropped = image[t_new:t_new + h_new, l_new:l_new + w_new]
	except ValueError:
		print(l, t, w, h)
		return None
	try:
		assert ((image_cropped.shape[1] == image_cropped.shape[0]) or w > image.shape[0])
	except AssertionError:
		print(imagePath, l, w, t, h, l_new, w_new, t_new, h_new, image_cropped.shape[1], image_cropped.shape[0])
	if show:
		cv2.imshow("image", image)
		cv2.imshow("image cropped", image_cropped)
		cv2.waitKey(0)
	return image_cropped


def generate_data(outPickleName, outCSVName):
	ff = open(interpolated_data_file, "r")
	data_csv = list(csv.DictReader(ff))
	num_frames = 0
	start_time = time.time()
	frames_list = []
	pickleFile = open(outPickleName, "wb")
	csvFile = open(outCSVName, "w")
	csvWriter = csv.writer(csvFile)
	csvWriter.writerow(["Index", "Class", "Class ID", "Object", "Transformation", "File Name"])
	for i in range(len(data_csv)):
		num_frames += 1
		row = data_csv[i]
		cl = row['ca']
		obj = row['no']
		tr = row['tr']
		fr = row['fr']
		fileName = cl + "_" + obj.zfill(2) + "//" + cl + "_" + obj.zfill(2) + "_pivothead_" + tr + \
				   ".mp4_" + fr.zfill(3) + ".jpeg"
		filePath = data_path + fileName
		try:
			assert os.path.isfile(filePath)
		except AssertionError:
			raise AssertionError(filePath + " not found.")
		if cl in classes:
			im = cv2.imread(filePath)
			left, top, width, height = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
			cropped_im = crop_image(fileName, im, l = left, t = top, w = width, h = height)
			resized_im = cv2.resize(cropped_im, (400, 400), interpolation = cv2.INTER_LINEAR)
			_, encoded_im = cv2.imencode(".jpeg", resized_im)
			frames_list.append(encoded_im)
			csvWriter.writerow([i, cl, classes.index(cl), obj, tr, fileName])
		if i % 1000 == 999:
			print(i, time.time() - start_time)
	print(num_frames)
	csvFile.close()
	pickle.dump(frames_list, pickleFile, pickle.DEFAULT_PROTOCOL)
	pickleFile.close()


def genVideo(outCSVName, pickleFileName, cl, obj, tr, outFilePath):
	csvFile = list(csv.DictReader(open(outCSVName, "r")))
	imgsFile = pickle.load(open(pickleFileName, "rb"))
	fourcc = cv2.VideoWriter_fourcc(*"XVID")
	outWriter = cv2.VideoWriter(outFilePath, fourcc, 10, (256, 256))
	for i in range(len(csvFile)):
		row = csvFile[i]
		if cl == row['Class'] and obj == int(row['Object']) and tr == row["Transformation"]:
			im = cv2.imdecode(imgsFile[i], 3)
			im_resized = cv2.resize(im, (256, 256), interpolation = cv2.INTER_LINEAR)
			outWriter.write(im_resized)
	outWriter.release()


def genAllVideos(outCSVName, pickleFileName, outDir):
	start_time = time.time()
	if not os.path.isdir(outDir):
		os.mkdir(outDir)
	for cl in classes:
		cl_path = outDir + str(cl)
		if not os.path.isdir(cl_path):
			os.mkdir(cl_path)
		for obj in range(1, 31):
			obj_path = cl_path + "/" + str(obj)
			if not os.path.isdir(obj_path):
				os.mkdir(obj_path)
			for tr in transformations:
				filePath = obj_path + "/" + tr + ".avi"
				genVideo(outCSVName = outCSVName, pickleFileName = pickleFileName, cl = cl, obj = obj, tr = tr,
						 outFilePath = filePath)
			print(cl, obj, time.time() - start_time)


if __name__ == "__main__":
	imFileName = "../supervised_data/toybox_interpolated_cropped_all.pickle"
	csvFileName = "../supervised_data/toybox_interpolated_cropped_all.csv"
	vid_dir = "../cropped_videos/"
	generate_data(outPickleName = imFileName, outCSVName = csvFileName)
	#genVideo(outCSVName = csvFileName, pickleFileName = imFileName, cl = "airplane", obj = 1, tr = "rxminus",
			# outFilePath = "out.avi")
	genAllVideos(outCSVName = csvFileName, pickleFileName = imFileName, outDir = vid_dir)
