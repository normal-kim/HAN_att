from model_han import * 
from data_loader_han import *
import numpy as np
from sklearn.metrics import roc_auc_score
from itertools import chain
import copy

######################### Classifier  #########################

class createClassifier(nn.Module):
    def __init__(self, encoder, params_dict, 
                 drop1 = 0.3, drop2 = 0.2, drop3 = 0.2, 
                fc1 = 100, fc2 = 20):
        super(createClassifier, self).__init__()
        self.encoder = encoder
        self.params_dict = params_dict
        if (params_dict['fc'] == None):
            self.classifier = nn.Sequential(
                nn.Dropout(drop1), 
                nn.Linear( params_dict['total_dim'], fc1), 
                nn.Dropout(drop2), 
                nn.Linear( fc1, fc2), 
                nn.Dropout(drop3),
                nn.Linear( fc2, params_dict['output_size']), 
                nn.Sigmoid()
            )
        else:
            self.classifier = params_dict['fc']
    
    def forward(self, x_s):
        docs, doc_lengths, sent_lengths = x_s
        v, a_it, a_i = self.encoder(docs, doc_lengths, sent_lengths)
        sig_out = self.classifier(v)
        return sig_out, a_it, a_i


def initialize_encoder(params_dict):
	han_encoder = ReviewHAN(params_dict).to('cuda:0')
	classifier = createClassifier( han_encoder, params_dict).to('cuda:0')
	optimizer = torch.optim.Adam(classifier.parameters(), lr = params_dict['lr'])
	return classifier, optimizer



def create_fc( fc1, fc2, dropval, drop2 = 0.2, drop3 = 0.2):
    fc = nn.Sequential(
                nn.Dropout(dropval), 
                nn.Linear( 200, fc1), 
                nn.Dropout(drop2), 
                nn.Linear( fc1, fc2), 
                nn.Dropout(drop3),
                nn.Linear( fc2, 1), 
                nn.Sigmoid()
            )
    return fc
    

######################### Train LOOP  #########################



def do_training(df_train, params_dict, criterion, clip_flag = False):
	print('Start Training =======>')
	#print(params_dict)
	print('='*80)

	classifier, optimizer = initialize_encoder(params_dict)
	train_loader, valid_loader = get_trainloader(df_train, params_dict)
	print_flag = False
	train_losses = []
	train_corrects = []
	steps = 0
	best_val = 10
	best_hm = 0

	for e in range( params_dict['epochs']):
		tr_labels = []
		tr_outputs = []
		for idx, sample in enumerate(train_loader):
			steps += 1
			# classifer
			classifier.train()
			classifier.zero_grad()

			# data
			label = sample[1].to('cuda:0')

			x_s = get_x_s(sample)
			output, _, _ = classifier(x_s)

			train_loss = criterion(output, label.unsqueeze(1) )
			train_loss.backward()

			train_losses.append(train_loss.item())
			tr_outputs.append(output.flatten())
			tr_labels.append(label.flatten())

			if clip_flag:
				nn.utils.clip_grad_norm_(classifier.parameters(), params_dict['clip_val'])

			optimizer.step()

			if (steps) % params_dict['print_every'] == 0:
				val_losses = []
				val_auc = []
				val_correct = 0
				labels = []
				outputs = []
				classifier.eval()
				for sample in valid_loader:
					label = sample[1].to('cuda:0')

					x_s = get_x_s(sample)
					output, _, _ = classifier(x_s)
					val_loss = criterion( output, label.unsqueeze(1) )
					val_losses.append(val_loss.item())
					outputs.append(output.flatten() )
					labels.append( label.flatten() )
					val_correct += torch.sum(output.flatten().round() == label).item()

				lab_tot = list(chain(*[ labels[i].detach().cpu().tolist() for i in range(len(labels))]))
				out_tot = list(chain(*[ outputs[i].detach().cpu().tolist() for i in range(len(outputs))]))   
				roc_val = roc_auc_score( lab_tot, out_tot)

				val_acc = val_correct / valid_loader.n_samples
				harmonic = 2 / (1 / val_acc + 1 / roc_val)                
				if print_flag == True:
					print("Epoch:{}/{}".format(e+1, params_dict['epochs']), 
					     "step:{}".format(steps), 
					      "[Train] Loss:{:.3f}".format(np.mean(train_losses)), 
					      "[Val] Loss:{:.3f}".format(np.mean(val_losses) ), 
					      "ACC:{:.3f}".format(val_acc), "AUC:{:.3f}".format(roc_val), 
					      "HM:{:.3f}".format(harmonic)
					     )

				if (np.mean(val_losses) < best_val) and (harmonic > best_hm):
					best_val = np.mean(val_losses)
					best_hm = harmonic
					print(f' ==> New best value..loss:{best_val} hm:{best_hm}')
					best_model_wts = copy.deepcopy( classifier.state_dict() )
	
	classifier.load_state_dict(best_model_wts)                    
	return best_hm, classifier