#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
########################################################
# Reinforcement based Active Learning Model
# Automation Project
# @author: Mirudhula Mukundan
# @date: May 5, 2023
########################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import csv
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer 
from keras_preprocessing.sequence import pad_sequences 
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from scipy.special import softmax

from imblearn.under_sampling import RandomUnderSampler

from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim

import pickle



##################
# MAKE CHANGES HERE
##################

##################
# Initializations & Hyperparameters
##################

output_file_loc = '/user_data/mirudhum/rein/'
data_loc = '/home/mirudhum/rein/'

# Hyperparameters

global simulation
global roundsNum
N_simulations = 10
split_ratio = 0.05
batch_size = 1

hidden_dim = 25

# exploration vs exploitation
explr_prob = 0.1



##################
# MAKE NO CHANGES FROM HERE
##################


state_dim = 3
action_dim = 2


if not os.path.isfile(output_file_loc+'vars.pkl'):
	##################
	# Extracting data
	##################

	data = []
	with open(data_loc+'data.csv', 'r') as csvfile: 
	    reader = csv.reader(csvfile, skipinitialspace=True)
	    data.append(tuple(next(reader)))
	    for ID, SMILE, B, RES, is_active in reader:
	        data.append((ID, SMILE, B, RES, is_active))

	# remove header
	data = data[1:]
	maxlen = len(data)
	print('Number of entries processed: ', maxlen)


	# Get only the SMILES data
	SMILES = []
	for i in data:
	    SMILES.append(i[1])


	##################
	# Molecular fingerprint encoding of SMILES
	##################

	# We will be using RDK fingerprint to vectorize all our SMILES structures
	x = []
	for i in range(len(SMILES)):
	    mol = Chem.MolFromSmiles(SMILES[i])
	    fingerprint_rdk = np.array(RDKFingerprint(mol))
	    x.append(fingerprint_rdk)
	x = np.array(x)


	print("Completed creating fingerprints")
	print("Number of features =",x.shape[1])



	##################
	# Get labels
	##################


	# Checking data for number of active compounds
	# idx - ID, SMILE, B, RES, is_active
	count = 0
	for i in range(len(data)):
	    if float(data[i][2]) <= -17.5:
	        if float(data[i][4]) == 1:
	            count += 1
	print("Number of active compounds =",count)
	# This follows what is mentioned in the paper.

	# Get the B-score values
	Bscores = []
	active = []
	is_active = []
	for i in range(len(data)):
	    Bscores.append(float(data[i][2]))
	    is_active.append(float(data[i][4]))
	    if float(data[i][4]) == 0:
	        active.append("Inactive")
	    else:
	         active.append("Active")   
	compound_idx = np.arange(len(Bscores))
	y = np.array(is_active)


	##################
	# Save Data
	##################

	with open(output_file_loc+'vars.pkl', 'wb') as f:
		pickle.dump([x, y], f)



print("Fetching data...")
with open(output_file_loc+'vars.pkl', 'rb') as f:
    x, y = pickle.load(f)

print("Data shape =", x.shape, y.shape)

##################
# General Helper functions
##################

# Initial train-test split
def train_test_split(X, y, split_ratio=0.8):
	# split data
	# get random permutations
	rand_idx = np.random.permutation(len(X))
	train_size = round(len(X)*split_ratio)
	trainX = X[rand_idx[0:train_size]]
	testX = X[rand_idx[train_size:len(X)]]
	trainY = y[rand_idx[0:train_size]]
	testY = y[rand_idx[train_size:len(X)]]

	return trainX, trainY, testX, testY


# Includes undersampling of major label to make classes balanced.
def Get_RUS(fingerprints, active, rng):
    rus = RandomUnderSampler(random_state=rng) # to make it reproducible -- random_state=rng
    X_resampled, y_resampled = rus.fit_resample(fingerprints, active)
    return X_resampled, y_resampled

# Includes undersampling of major label to make classes balanced.
def Get_RUS_no_rng(fingerprints, active):
    rus = RandomUnderSampler() # to make it reproducible -- random_state=rng
    X_resampled, y_resampled = rus.fit_resample(fingerprints, active)
    return X_resampled, y_resampled


##################
# Helper functions for active learning
##################

# Define a function that calculates the uncertainty of the unlabeled dataset
def uncertainty(unlabeled_data, model):
    # Predict the probability of each sample belonging to the positive class
	prob_pred = model.predict_proba(unlabeled_data)
	prob_log_pred = model.predict_log_proba(testX)
	entropy = -1*np.sum(prob_pred*prob_log_pred, axis=1)
	return entropy

# Define a function that calculates the similarity between samples in the labeled and unlabeled datasets
def similarity(unlabeled_data):
    # Calculate the pairwise distances between samples in the labeled and unlabeled datasets using Euclidean distance
    distances = cdist(unlabeled_data, unlabeled_data, metric='euclidean')
    # Calculate the mean distance of each sample in the unlabeled dataset to all the labeled samples
    mean_distance = np.mean(distances, axis=1)
    # Normalize the distances to be between 0 and 1
    normalized_distance = (mean_distance - np.min(mean_distance)) / (np.max(mean_distance) - np.min(mean_distance))
    return normalized_distance


def diversity(X_unlabeled):
    # Use KMeans clustering to group similar samples together
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_unlabeled)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    cluster_distances = kmeans.transform(X_unlabeled)
    cluster_probabilities = softmax(-cluster_distances, axis=1)
    return cluster_probabilities[:,1]


# def representativeness(X_unlabeled):
#     # Use kernel density estimation to calculate the density of the samples
#     kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_unlabeled)
#     log_density = kde.score_samples(X_unlabeled)
#     return log_density




##################
# Helper functions for reinforcement learning
##################


def get_state(X_unlabeled, model):
	# Predict probabilities for the unlabeled samples
	uncertainty_scores = uncertainty(X_unlabeled, model)
	print("uncertainty dim =", uncertainty_scores.shape)

	# Get the similarity scores between samples
	similarity_scores = similarity(X_unlabeled)
	print("similarity dim =", similarity_scores.shape)

	# # Get the representative density scores
	# representative_scores = representativeness(X_unlabeled)
	# print("representative dim =", representative_scores.shape)

	# Diversity scores
	diversity_scores = diversity(X_unlabeled)
	print("diversity dim =", diversity_scores.shape)

	# Concatenate the states into a single numpy array
	state_representation = np.column_stack((uncertainty_scores, similarity_scores, diversity_scores))
	print("state shape =",state_representation.shape)

	return state_representation


##################
# Defining Models
##################

class LogR():
    def __init__(self, x, y):
        self.trainX = x
        self.trainY = y.reshape(len(y),)
        self.model = LogisticRegression(random_state=0, max_iter=500)
    def fit(self):
        self.model.fit(self.trainX, self.trainY)
    def predict(self, x):
        return self.model.predict(x)
    def predict_proba(self, x):
        return self.model.predict_proba(x)
    def predict_log_proba(self, x):
        return self.model.predict_log_proba(x)
    def generate_acc(self, pred, true):
        return accuracy_score(true, pred)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights with Xavier initialization
        nn.init.uniform_(self.fc1.weight)
        nn.init.uniform_(self.fc2.weight)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x


class ActiveLearningAgent:
	def __init__(self, state_dim, action_dim, hidden_dim, lr=0.001, gamma=0.99):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		self.gamma = gamma

		self.policy_net = MLP(state_dim, hidden_dim, action_dim)
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

	def select_action(self, state, explr_prob):
		with torch.no_grad():
			state = torch.FloatTensor(state).unsqueeze(0)
			action_probs = self.policy_net(state)
			action_probs = action_probs.detach().squeeze()#.numpy().squeeze()
		print("Action_probs", action_probs.shape,action_probs[0:3,:])
		dist = []
		actions = []
		logprob = []
		for j in range(action_probs.shape[0]):
			# print(action_probs[j,:])
			dist.append(torch.distributions.Categorical(action_probs[j,:]))
			actions.append(dist[j].sample().item())
			logprob.append(dist[j].log_prob(dist[j].sample()))
		actions = np.array(actions)
		# print(logprob)

		# exploration vs exploitation
		random_val = np.random.rand()
		if random_val < explr_prob: # do exploration
			# Pick one randomly from the selections
			# Get indices of labeled ones
			ones_indices = np.where(actions == 1)[0]

			if len(ones_indices) != 0:
				# Randomly select an index from the labeled ones
				selected_index = np.random.choice(ones_indices)
				print("exploration")
			else:
				selected_index = np.random.choice(np.arange(actions.shape[0]))
				print("random")

		else: # do exploitation
			# Get index of highest probability of getting 1
			# most confident
			selected_index = np.argmax(action_probs[:, 1])
			print("exploitation")

		return selected_index.item(), logprob[selected_index]

	def update_policy(self, state, reward, logprobs):
		state = torch.FloatTensor(state).unsqueeze(0)
		action_probs = self.policy_net(state).squeeze().squeeze()
		print("new action prob =",action_probs)
		loss = 0
		# print(logprobs)
		# for i in range(len(logprobs)):
		# 	loss += - logprobs[i] 	* (-reward	)	
		# loss = loss / len(logprobs)

		loss = -reward * torch.log(action_probs[1])

		print("Loss =",loss)
		all_losses[simulation, roundNum] = loss.item()
		# loss.requires_grad_()

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()



#############
# Implementation
#############



print("***************************")
print("***************************")
print("Hyperparameters")
print("***************************")
print("***************************")

print("Hyperparameters:")
print("Number of simulations =",N_simulations)
print("Batch Size =",batch_size)
print("State Dimension =",state_dim)
print("Action Dimension =",action_dim)
print("Hidden Dimension =",hidden_dim)
print("Exploration probability =",explr_prob)


#initially 

X_resampled, y_resampled = Get_RUS(x, y, rng=10)
trainX, trainY, testX, testY = train_test_split(X_resampled, y_resampled, split_ratio=split_ratio)
print("ttt",len(testX))
len_test = len(testX)
N_rounds = int(len_test / batch_size) - 10
init_train_size = round(len(trainX)*(split_ratio))

print("Number of rounds =",N_rounds)
print("Initial train size =",init_train_size)

X_resampled, y_resampled = Get_RUS(x, y, rng=10)

global all_losses 
all_losses = np.zeros((N_simulations, N_rounds))



print("***************************")
print("***************************")
print("Starting training and evaluation")
print("***************************")
print("***************************")
##################
# Implementation
##################

performance_mat = np.zeros((N_simulations, N_rounds))
prec_mat = np.zeros((N_simulations, N_rounds))
recall_mat = np.zeros((N_simulations, N_rounds))
f1_mat = np.zeros((N_simulations, N_rounds))

for simulation in range(N_simulations):
	np.random.seed(simulation+10)
	
	# X_resampled, y_resampled = Get_RUS_no_rng(x, y)
	trainX, trainY, testX, testY = train_test_split(X_resampled, y_resampled, split_ratio=split_ratio)

	rewardcounter = 1
	reward = 0
	# initialize the active learning agent
	agent = ActiveLearningAgent(state_dim, action_dim, hidden_dim)
	# loop over every round
	for roundNum in range(N_rounds):
		print("***************************")
		print("New Round",roundNum)
		print("***************************")
		# Get performance metrics on the logistic regression model
		logreg = LogR(trainX, trainY)
		logreg.fit()
		pred_y = logreg.predict(testX)
		prev_acc = logreg.generate_acc(pred_y, testY)
		performance_mat[simulation, roundNum] = prev_acc
		print("Prev Performance =", prev_acc)

		# calculate precision, recall, and F1 score
		precision = precision_score(testY, pred_y)
		prec_mat[simulation, roundNum] = precision
		recall = recall_score(testY, pred_y)
		recall_mat[simulation, roundNum] = recall
		f1 = f1_score(testY, pred_y)
		f1_mat[simulation, roundNum] = f1


		if len(testX) >= batch_size:
			# training 
			# select the next batch of samples
			# print(trainX, trainY)
			state = get_state(testX, logreg)
			# print(state)
			selected_samples = []
			logprobs = []
			for j in range(batch_size):
				print("Iteration",simulation, roundNum, j, len(testX))
				# select_action will take a bunch of samples, calculate teh action for each sample
				# and proceeds to give the best one with consideration to 
				# exploration and exploitation
				if roundNum < 80:
					explr_prob = 0.8
				else:
					explr_prob = 0.1
				selected_index, logprob = agent.select_action(state, explr_prob)
				print("Selected index =", selected_index)
				selected_samples.append(selected_index)
				logprobs.append(logprob)
				# state = np.delete(state, selected_index, axis=0)

			# update the training set
			# Add row from testX to trainX
			trainX = np.concatenate((trainX, testX[selected_samples]), axis=0)
			trainY = np.concatenate((trainY, testY[selected_samples]), axis=0)

			# Delete row from testX
			testX = np.delete(testX, selected_samples, axis=0)
			testY = np.delete(testY, selected_samples, axis=0)

			print("Data shapes",trainX.shape, trainY.shape, testX.shape, testY.shape, state.shape)

			# train the logistic regression model on the updated training set
			new_logreg = LogR(trainX, trainY)
			new_logreg.fit()
			pred_y = new_logreg.predict(testX)
			new_acc = new_logreg.generate_acc(pred_y, testY)
			print("New Performance =", new_acc)

			# compute the reward based on the change in accuracy
			diff = new_acc - prev_acc
			if diff > 0:
				rewardcounter += 1
			else:
				rewardcounter = 1
			reward = diff**(1/rewardcounter)
			# if new is greater than prev, it will be rewarding
			# otherwise it is punishment
			print("Reward =", reward)
			# update the policy network
			agent.update_policy(state[selected_samples,:], reward*100, logprobs)




# Average over all simulations
avg_acc = np.mean(performance_mat, axis=0)
avg_std = np.std(performance_mat, axis=0)
avg_loss = np.mean(all_losses, axis=0)
std_loss = np.std(all_losses, axis=0)

avg_prec = np.mean(prec_mat, axis=0)
std_prec = np.std(prec_mat, axis=0)
avg_recall = np.mean(recall_mat, axis=0)
std_recall = np.std(recall_mat, axis=0)
avg_f1 = np.mean(f1_mat, axis=0)
std_f1 = np.std(f1_mat, axis=0)

# Save the array to a file
np.save(output_file_loc+'rein_performance'+str(N_simulations)+'.npy', avg_acc)
np.save(output_file_loc+'rein_std'+str(N_simulations)+'.npy', avg_std)
np.save(output_file_loc+'rein_losses'+str(N_simulations)+'.npy', all_losses)

np.save(output_file_loc+'rein/rein_prec'+str(N_simulations)+'.npy', avg_prec)
np.save(output_file_loc+'rein/rein_recall'+str(N_simulations)+'.npy', avg_recall)
np.save(output_file_loc+'rein/rein_f1'+str(N_simulations)+'.npy', avg_f1)

np.save(output_file_loc+'rein/rein_prec_std'+str(N_simulations)+'.npy', std_prec)
np.save(output_file_loc+'rein/rein_recall_std'+str(N_simulations)+'.npy', std_recall)
np.save(output_file_loc+'rein/rein_f1_std'+str(N_simulations)+'.npy', std_f1)



plt.figure()
# Create x axis values
xaxis = np.arange(N_rounds)
print(avg_acc.shape, xaxis.shape)
# Create plot
sns.set_style("whitegrid")
# Plot the data with shaded error bars
# Plot the data with error bars
# Create a mask to show error bars only every 50 data points
# Subset arrays to show error bars every 50 on the x-axis
step_size = 50
x_subset = xaxis[::step_size]
y_subset = avg_acc[::step_size]
std_subset = avg_std[::step_size]

# Create line plot with error bars
sns.lineplot(x=xaxis, y=avg_acc, errorbar='ci', err_style='bars')

# Overlay error bars for the subset of data
plt.errorbar(x=x_subset, y=y_subset, yerr=std_subset, fmt='none', color='black', capsize=5, label='Reinforcement Active Learning')# plt.plot(xaxis, avg_acc)
# plt.errorbar(xaxis, avg_acc, yerr = avg_std, color='c', label='Random Learning')
plt.ylim([0.4, 1])
plt.ylabel("Accuracy")
plt.xlabel("No. of samples in training set")
plt.savefig(output_file_loc+"rein_performance_plot"+str(N_simulations)+".png")


plt.figure()
# Create x axis values
xaxis = np.arange(N_rounds)
print(avg_loss.shape, xaxis.shape)
# Create plot
sns.set_style("whitegrid")
# Plot the data with shaded error bars
# Plot the data with error bars
# Create a mask to show error bars only every 50 data points
# Subset arrays to show error bars every 50 on the x-axis
step_size = 50
x_subset = xaxis[::step_size]
y_subset = avg_loss[::step_size]
std_subset = std_loss[::step_size]

# Create line plot with error bars
sns.lineplot(x=xaxis, y=avg_loss, errorbar='ci', err_style='bars')

# Overlay error bars for the subset of data
plt.errorbar(x=x_subset, y=y_subset, yerr=std_subset, fmt='none', color='black', capsize=5, label='Loss')# plt.plot(xaxis, avg_acc)
# plt.errorbar(xaxis, avg_acc, yerr = avg_std, color='c', label='Random Learning')
# plt.ylim([0.4, 1])
plt.ylabel("Loss")
plt.xlabel("No. of samples in training set")
plt.savefig(output_file_loc+"rein_loss_plot"+str(N_simulations)+".png")












