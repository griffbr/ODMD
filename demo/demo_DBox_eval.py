"""
Demonstration of how to evaluate DBox network on ODMD.
"""

import sys, os, IPython, torch, _pickle as pickle, numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
sys.path.insert(0,"../")
import odmd, dbox

net_name = "DBox_demo"
model_idx = -1 # Can cycle through indices to find best validation performance.

# Select dataset to evaluate.
dataset = "odmd" # or "odms_detection" for ODMS dataset converted to detection.
eval_set = "val" # or "test" once model training and development are complete.

# Select configuration (more example settings from paper in config directory).
datagen_config = "../config/data_gen/standard_data.yaml" 
camera_config = "../config/camera/hsr_grasp_camera.yaml" 
train_config = "../config/train/train_demo.yaml"
dbox_config = "../config/model/DBox.yaml"

# Initiate data generator, model, data loader, and load weights.
odmd_data = odmd.data_gen.DataGenerator(datagen_config)
odmd_data.initialize_data_gen(camera_config)
net, device, m_params = dbox.load_model(dbox_config, odmd_data.num_pos)
bb2net = dbox.BoundingBoxToNetwork(m_params)
model_dir = os.path.join("../results", "model", net_name)
model_list = sorted([pt for pt in os.listdir(model_dir) if pt.endswith(".pt")])
net = dbox.load_weights(net, os.path.join(model_dir, model_list[model_idx]))

# Initiate dataset information.
set_dir = os.path.join("../data", dataset, eval_set)
set_list = sorted([pk for pk in os.listdir(set_dir) if pk.endswith(".pk")])
percent_error=[]; abs_error=[]; predictions_all=[]

with torch.no_grad():
	for test in set_list:
		# Load data for specific set.
		bb_data = pickle.load(open(os.path.join(set_dir, test), "rb"))
		bb_3D, bb = bb_data["bb_3D"], bb_data["bb"]

		# Run DBox with correct post-processing for configuration.
		bb2net.set_batch(bb_3D["n_ex"])
		bb2net.bb_to_labels(bb_3D, bb)
		inputs = bb2net.inputs.to(device)
		predictions = net(inputs).cpu().numpy()
		if bb2net.prediction == "normalized":
			predictions[:,0] *= bb2net.norm
		depths = bb["bboxes"][-1][-1]

		percent_error.append(np.mean( abs(predictions[:,0] - depths) / depths))
		abs_error.append(np.mean(abs(predictions[:,0] - depths)))
		predictions_all.append(predictions)

	# Print out final results.
	print("\nResults summary for ODMD %s sets." % eval_set)
	for i, test_set in enumerate(set_list):
		print("\n%s-%s:" % (test_set, eval_set))
		print("Mean Percent  Error: %.4f" % percent_error[i]) 
		print("Mean Absolute Error: %.4f (m)" % abs_error[i]) 

# Generate final results file.
name = model_list[model_idx].split(".pt")[0]
data_name = "%s_%s" % (dataset, eval_set)
print("\nSaving %s results file for %s.\n" % (data_name, name))
result_data = {"Result Name": name, "Set List": set_list, 
				"Percent Error": percent_error, "Absolute Error": abs_error, 
				"Depth Estimates": predictions_all, "Dataset": data_name}
os.makedirs("../results/", exist_ok=True)
pickle.dump(result_data, open("../results/%s_%s.pk" % (name, data_name), "wb"))
