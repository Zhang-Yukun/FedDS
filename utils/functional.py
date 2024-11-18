import random
import numpy as np
import torch
import tqdm


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_embeddings(model, data):
    model.cuda()
    model.eval()
    last_hidden_states_avg_all = []
    with torch.no_grad():
        for datapoint in tqdm.tqdm(data["train_dataset"], total=len(data["train_dataset"])):
            input_ids = datapoint["input_ids"].unsqueeze(0).to(model.device)
            labels = datapoint["labels"].unsqueeze(0).to(model.device)
            result = model(input_ids=input_ids, labels=labels, return_dict=True, output_hidden_states=True)
            hidden_states = result["hidden_states"]
            last_layer_hidden_states = hidden_states[-1] # (batch_size=1, seq_len, hidden_dim)
            # avg_pooling the sequence-hidden-states
            last_hidden_states_avg = torch.mean(last_layer_hidden_states.squeeze(0), dim=0)  # -> (hidden_dim)
            last_hidden_states_avg_all.append(last_hidden_states_avg)  # keep track
    del model
    return last_hidden_states_avg_all # list[Tensor .shape=(hidden_dim,)], .len=n_samples


