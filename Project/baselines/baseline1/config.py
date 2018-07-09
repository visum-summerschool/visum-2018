DATA_SET_SIZE = 154
n_images_in_train = 149  # number of images used for training

curr_image = ""   # changed in each iteration during the validation
debug_path = "debug/"
debug_segment = True
debug_probability = True
debug_final = True
debug_results = True

validate = "remaining"  # The method is validated on remaining images
#validate = "all"         # The method is validated on all images
#validate = False        # The method is not validated

image_size = 512    # All images will be normalized to have this width
seed = None     # For reproducible results substitute "None" by a number
