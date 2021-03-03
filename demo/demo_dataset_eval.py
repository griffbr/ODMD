"""
Demonstration of how to evaluate a depth estimation model on ODMS.
"""

import sys, os, cv2, IPython, numpy as np, _pickle as pickle
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
sys.path.insert(0,"../")
import odms

eval_set = "val" # or "test" once model training and development are complete.
set_list = ["robot", "driving", "normal", "perturb"]
display_iter=100; percent_error=[]; abs_error=[]; depths_all=[]

for test_set in set_list:

	# Get a list of parameters determining how test is going to go.
	test_data = odms.eval.initialize_test_data(eval_set, test_set)

	# Evaluate each test example.
	n = test_data["n_examples"]
	depths = np.zeros(n) 
	ground_truth = np.zeros(n)

	print("Processing results for %s (%s total examples)." % (test_set, n))
	for i, mask_list in enumerate(test_data["files"]):
		seg_masks = [cv2.imread(f,cv2.IMREAD_GRAYSCALE)/255 for f in mask_list]
		camera_movement = test_data["camera_movement"][i]

		"""
		Use your own depth estimation model here (to replace VOS-DE):
		"""
		depth_estimate = odms.vosde.estimate_depth(seg_masks, camera_movement)

		# Save results.
		depths[i] = depth_estimate
		ground_truth[i] = test_data["depths"][i][0]
		if i % display_iter == 0:
			print("%4d/%s" % (i, n))
	percent_error.append(np.mean( abs(depths - ground_truth) / ground_truth))
	abs_error.append(np.mean(abs(depths - ground_truth)))
	depths_all.append(depths)
	print("Mean Percent  Error: %.4f" % percent_error[-1]) 
	print("Mean Absolute Error: %.4f (m)\n" % abs_error[-1]) 

# Print out final results.
print("\nResults summary for ODMS %s sets." % eval_set)
for i, test_set in enumerate(set_list):
	print("\n%s-%s:" % (test_set, eval_set))
	print("Mean Percent  Error: %.4f" % percent_error[i]) 
	print("Mean Absolute Error: %.4f (m)" % abs_error[i]) 

# Generate final results file.
name = "VOS-DE"
result_data = {"Result Name": name, "Set List": set_list, "Eval": eval_set, 
				"Percent Error": percent_error, "Absolute Error": abs_error, 
				"Depth Estimates": depths_all}
pickle.dump(result_data, open("../results/%s_%s.pk" % (name, eval_set), "wb"))
