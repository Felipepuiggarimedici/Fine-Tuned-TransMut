from modelAndPerformances import *

pep_max_len = 9 # peptide; enc_input max sequence length
hla_max_len = 34 # hla; dec_input(=dec_output) max sequence length
tgt_len = pep_max_len + hla_max_len
pep_max_len, hla_max_len

vocab = np.load('tokenizer/Transformer_vocab_dict.npy', allow_pickle = True).item()
vocab_size = len(vocab)


# In[36]:


# Transformer Parameters
d_model = 64  # Embedding Size
d_ff = 512 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 2  # number of Encoder of Decoder Layer before Fourier Transform

batch_size = 1024
epochs = 50
threshold = 0.5

folder_path = 'data/'
files = os.listdir(folder_path)
folds = sum(1 for f in files if os.path.isfile(os.path.join(folder_path, f)) and 'train' in f and 'fold' in f)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

independent_data, independent_pep_inputs, independent_hla_inputs, independent_labels, independent_loader = data_with_loader(type_ = 'independent',fold = None,  batch_size = batch_size)
external_data, external_pep_inputs, external_hla_inputs, external_labels, external_loader = data_with_loader(type_ = 'external',fold = None,  batch_size = batch_size)


for n_heads in [1, 2, 3]:
    
    ys_train_fold_dict, ys_val_fold_dict = {}, {}
    train_fold_metrics_list, val_fold_metrics_list = [], []
    independent_fold_metrics_list, external_fold_metrics_list, ys_independent_fold_dict, ys_external_fold_dict = [], [], {}, {}
    attns_train_fold_dict, attns_val_fold_dict, attns_independent_fold_dict, attns_external_fold_dict = {}, {}, {}, {}
    loss_train_fold_dict, loss_val_fold_dict, loss_independent_fold_dict, loss_external_fold_dict = {}, {}, {}, {}

    for fold in range(folds):
        print('=====Fold-{}====='.format(fold))
        print('-----Generate data loader-----')
        train_data, train_pep_inputs, train_hla_inputs, train_labels, train_loader = data_with_loader(type_ = 'train', fold = fold,  batch_size = batch_size)
        val_data, val_pep_inputs, val_hla_inputs, val_labels, val_loader = data_with_loader(type_ = 'val', fold = fold,  batch_size = batch_size)
        print('Fold-{} Label info: Train = {} | Val = {}'.format(fold, Counter(train_data.label), Counter(val_data.label)))

        print('-----Compile model-----')
        model = FourierTransformer(d_model, d_k, n_layers, n_heads, d_ff).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = 1e-3)#, momentum = 0.99)

        print('-----Train-----')
        dir_saver = './model/pHLAIformerFourier/'
        path_saver = './model/pHLAIformerFourier/model_layer{}_multihead{}_fold{}.pkl'.format(n_layers, n_heads, fold)
        print('dir_saver: ', dir_saver)
        print('path_saver: ', path_saver)

        metric_best, ep_best = 0, -1
        time_train = 0
        for epoch in range(1, epochs + 1):

            ys_train, loss_train_list, metrics_train, time_train_ep = train_step(model, train_loader, fold, epoch, epochs, criterion, optimizer, threshold=threshold, use_cuda = use_cuda) # , dec_attns_train
            ys_val, loss_val_list, metrics_val = eval_step(model, val_loader, fold, epoch, epochs,criterion, threshold = threshold, use_cuda = use_cuda) #, dec_attns_val

            metrics_ep_avg = sum(metrics_val[:4])/4
            if metrics_ep_avg > metric_best: 
                metric_best, ep_best = metrics_ep_avg, epoch
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                print('****Saving model: Best epoch = {} | 5metrics_Best_avg = {:.4f}'.format(ep_best, metric_best))
                print('*****Path saver: ', path_saver)
                torch.save(model.eval().state_dict(), path_saver)

            time_train += time_train_ep

        print('-----Optimization Finished!-----')
        print('-----Evaluate Results-----')
        if ep_best >= 0:
            print('*****Path saver: ', path_saver)
            model.load_state_dict(torch.load(path_saver))
            model_eval = model.eval()

            ys_res_train, loss_res_train_list, metrics_res_train = eval_step(model_eval, train_loader, fold, ep_best, epochs,criterion, threshold, use_cuda) # , train_res_attns
            ys_res_val, loss_res_val_list, metrics_res_val = eval_step(model_eval, val_loader, fold, ep_best, epochs,criterion, threshold, use_cuda) # , val_res_attns
            ys_res_independent, loss_res_independent_list, metrics_res_independent = eval_step(model_eval, independent_loader, fold, ep_best, epochs,criterion, threshold, use_cuda) # , independent_res_attns
            ys_res_external, loss_res_external_list, metrics_res_external = eval_step(model_eval, external_loader, fold, ep_best, epochs, criterion, threshold, use_cuda) # , external_res_attns

            train_fold_metrics_list.append(metrics_res_train)
            val_fold_metrics_list.append(metrics_res_val)
            independent_fold_metrics_list.append(metrics_res_independent)
            external_fold_metrics_list.append(metrics_res_external)

            ys_train_fold_dict[fold], ys_val_fold_dict[fold], ys_independent_fold_dict[fold], ys_external_fold_dict[fold] = ys_res_train, ys_res_val, ys_res_independent, ys_res_external    
#             attns_train_fold_dict[fold], attns_val_fold_dict[fold], attns_independent_fold_dict[fold], attns_external_fold_dict[fold] = train_res_attns, val_res_attns, independent_res_attns, external_res_attns   
            loss_train_fold_dict[fold], loss_val_fold_dict[fold], loss_independent_fold_dict[fold], loss_external_fold_dict[fold] = loss_res_train_list, loss_res_val_list, loss_res_independent_list, loss_res_external_list  

        print("Total training time: {:6.2f} sec".format(time_train))

print('****Independent set:')
print(performances_to_pd(independent_fold_metrics_list))
print('****External set:')
print(performances_to_pd(external_fold_metrics_list))
print('****Train set:')
print(performances_to_pd(train_fold_metrics_list))
print('****Val set:')
print(performances_to_pd(val_fold_metrics_list))