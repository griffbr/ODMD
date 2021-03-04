"""
Demonstration of how to generate new training data on ODMD.
"""

import sys, os, IPython, _pickle as pickle
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
sys.path.insert(0,"../")
import odmd

# Select configuration (more example settings from paper in config directory).
datagen_config = "../config/data_gen/standard_data.yaml" 
camera_config = "../config/camera/hsr_grasp_camera.yaml" 

# Other data generation settings.
n_examples = 20 # Configure for batch size if training.
save_examples = False
set_name = "example_data_gen"

# Initiate data generator.
odmd_data = odmd.data_gen.DataGenerator(datagen_config)
odmd_data.initialize_data_gen(camera_config)

# Generate examples for ODMD training (repeat for each training iteration).
bb_3D, bb = odmd_data.generate_object_examples(n_examples)
bb_3D, bb = odmd.data_gen.add_perturbations(bb_3D, bb, odmd_data)
bboxes, camera_movements, depths = odmd.data_gen.bb_to_inputs(bb_3D, bb, 
															odmd_data.num_pos)

"""
Use generated data to train your own network to predict depths given bboxes 
and camera_movements. See paper for ideas on possible initial configurations.
"""

# Save generated examples as a static dataset (optional).
if save_examples:
	result_dir = "../data/example_generated_data" 
	os.makedirs(result_dir, exist_ok=True)
	p_data = {"test_name": set_name, "bb_3D": bb_3D, "bb": bb}
	pickle.dump(p_data, open(os.path.join(result_dir, "%s.pk" % set_name), 'wb'))
