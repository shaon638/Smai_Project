import os
import shutil

annotation_file_path = "/ssd_scratch/cvit/shaon/imageNet-tiny/tiny-imagenet-200/val/val_annotations.txt"
destination_file_path = "/ssd_scratch/cvit/shaon/imageNet-tiny/tiny-imagenet-200/new_val"
source_images_path = "/ssd_scratch/cvit/shaon/imageNet-tiny/tiny-imagenet-200/val/images"
with open (annotation_file_path , "r")as file:
	count = 0
	for lines in file:
		lines_split = lines.split("\t")
		image_folder_name = lines_split[1]
		saved_path = os.path.join(destination_file_path, image_folder_name)
		saved_inner_path = os.path.join(saved_path, "images")
		image_name = lines_split[0]
		
		if os.path.exists(saved_path) == False:
			os.mkdir(saved_path)
			os.mkdir(saved_inner_path)
			source_img_path = os.path.join(source_images_path, image_name)
			destination_img_path = saved_inner_path
			dest = shutil.copy(source_img_path, destination_img_path)
			new_name = image_folder_name + "_" + image_name
			rename_img_path = os.path.join(saved_inner_path, new_name)
			os.rename(dest, rename_img_path)
			count += 1
		else:
			source_img_path = os.path.join(source_images_path, image_name)
			destination_img_path = saved_inner_path
			dest = shutil.copy(source_img_path, destination_img_path)
			new_name = image_folder_name + "_" + image_name
			rename_img_path = os.path.join(saved_inner_path, new_name)
			os.rename(dest, rename_img_path)
			count += 1
		print(count)




		

		

