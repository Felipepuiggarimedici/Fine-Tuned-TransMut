from modelAndPerformances import *
from hyperparametricSelection import hyperparameterSelection
#needed for iterating over three-tuple
from itertools import product

pep_max_len = 9 # peptide; enc_input max sequence length
hla_max_len = 34 # hla; dec_input(=dec_output) max sequence length
tgt_len = pep_max_len + hla_max_len
pep_max_len, hla_max_len

vocab = np.load('tokenizer/Transformer_vocab_dict.npy', allow_pickle = True).item()
vocab_size = len(vocab)

# Transformer Parameters
d_model = 64  # Embedding Size
d_ff = 512 # FeedForward dimension
patienceFull = 10 #patience for whole model training

n_heads, n_layers, d_k = hyperparameterSelection()
batch_size = 1024
epochs = 200
threshold = 0.5

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
results = []

data = pd.read_csv('data/trainDataFull/fullTrainingData.csv')
pep_inputs, hla_inputs, labels = make_data(data)
dataset = MyDataSet(pep_inputs, hla_inputs, labels)

loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
train_data, train_pep_inputs, train_hla_inputs, train_labels, train_loader = data, pep_inputs, hla_inputs, labels, loader

model = Transformer(d_model, d_k, n_layers, n_heads, d_ff).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)#, momentum = 0.99)

best_loss = float('inf')
patience_counter = 0
avgLosses = np.zeros(epochs)

print("Beginning training")

for epoch in range(1, epochs + 1):
    print("Epoch: ", epoch)
    _, loss_train_list, metrics_train, time_train_ep = train_step(
        model, train_loader, -1, epoch, epochs, criterion, optimizer, threshold, use_cuda
    )

    avg_loss = torch.mean(torch.stack(loss_train_list)).item()
    avgLosses[epoch - 1] = avg_loss
    print(f"Epoch {epoch}, Training Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
        print("Model improved. Saved.")
    else:
        patience_counter += 1
        print(f"No improvement. Patience counter: {patience_counter}/{patienceFull}")

    if patience_counter >= patienceFull:
        print("Early stopping triggered.")
        break
np.save('results/avg_losses.npy', avgLosses)
print("Training finished")