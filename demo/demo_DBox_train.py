"""
Demonstration of how to generate train DBox network on ODMD.
"""

import sys, os, IPython, torch
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
sys.path.insert(0,"../")
import odmd, dbox

net_name = "DBox_demo"

# Select configuration (more example settings from paper in config directory).
datagen_config = "../config/data_gen/standard_data.yaml" 
camera_config = "../config/camera/hsr_grasp_camera.yaml" 
train_config = "../config/train/train_demo.yaml"
dbox_config = "../config/model/DBox.yaml"

# Initiate data generator, model, training parameters, and data loader.
odmd_data = odmd.data_gen.DataGenerator(datagen_config)
odmd_data.initialize_data_gen(camera_config)
net, device, m_params = dbox.load_model(dbox_config, odmd_data.num_pos)
train = dbox.load_training_params(train_config)
bb2net = dbox.BoundingBoxToNetwork(m_params, train["batch_size"])

# Initiate training!
model_dir = os.path.join("../results", "model", net_name)
os.makedirs(model_dir, exist_ok=True)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0)

running_loss=0.0; ct=0
print("Starting training for %s." % net_name)
while ct < train["train_iter"]:

	# Generate examples for ODMD training (repeat for each training iteration).
	bb_3D, bb = odmd_data.generate_object_examples(bb2net.n_bat)
	bb_3D, bb = odmd.data_gen.add_perturbations(bb_3D, bb, odmd_data)

	# Network inputs and labels, forward pass, loss, and gradient.
	bb2net.bb_to_labels(bb_3D, bb)
	inputs, labels = bb2net.inputs.to(device), bb2net.labels.to(device)
	outputs = net(inputs).to(device)
	loss = criterion(outputs, labels)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	running_loss += loss.item()

	# Print progress details and save model at set interval.
	ct += 1
	if ct % train["display_iter"] == 0:
		cur_loss = running_loss / train["display_iter"]
		print("[%9d] loss: %.6f" % (ct, cur_loss))
		running_loss = 0.0
	if ct in train["save_iter"]:
		torch.save(net.state_dict(), "%s/%s_%09d.pt" % (model_dir,net_name,ct))
		print("[%9d] interval model saved." % ct)
