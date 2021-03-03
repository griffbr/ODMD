"""
Demonstration of how to evaluate a depth estimation model on ODMD.
"""

import sys, os, IPython, numpy as np, _pickle as pickle
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
sys.path.insert(0,"../")
import odmd

# Select dataset to evaluate.
dataset = "odmd" # or "odms_detection" for ODMS dataset converted to detection.
eval_set = "val" # or "test" once model training and development are complete.

# Misc. initialization.
n_observations = 10
set_dir = os.path.join("../data", dataset, eval_set)
set_list = sorted([pk for pk in os.listdir(set_dir) if pk.endswith(".pk")])
percent_error=[]; abs_error=[]; predictions_all=[]

for test in set_list:
	# Load data for specific set.
	bb_data = pickle.load(open(os.path.join(set_dir, test), "rb"))
	bb_3D, bb = bb_data["bb_3D"], bb_data["bb"]
	bboxes, camera_movements, depths = odmd.data_gen.bb_to_inputs(bb_3D, bb,
																n_observations)

	"""
	Use your own depth estimation model here (to replace Box_LS):
	"""
	predictions = odmd.closed_form.Box_LS(bboxes, camera_movements)

	percent_error.append(np.mean( abs(predictions - depths) / depths))
	abs_error.append(np.mean(abs(predictions - depths)))
	predictions_all.append(depths)

# Print out final results.
print("\nResults summary for ODMD %s sets." % eval_set)
for i, test_set in enumerate(set_list):
	print("\n%s-%s:" % (test_set, eval_set))
	print("Mean Percent  Error: %.4f" % percent_error[i]) 
	print("Mean Absolute Error: %.4f (m)" % abs_error[i]) 

# Generate final results file.
name = "Box_LS"
data_name = "%s_%s" % (dataset, eval_set)
result_data = {"Result Name": name, "Set List": set_list, 
				"Percent Error": percent_error, "Absolute Error": abs_error, 
				"Depth Estimates": predictions_all, "Dataset": data_name}
os.makedirs("../results/", exist_ok=True)
pickle.dump(result_data, open("../results/%s_%s.pk" % (name, data_name), "wb"))
