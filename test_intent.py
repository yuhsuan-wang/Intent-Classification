import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
import csv

import torch

from dataset import SeqClsDataset
from model_intent import SeqClassifier
from utils import Vocab
import itertools


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset

    x_test_text = [item['text'].split() for item in dataset.data]
    x_test = np.array(vocab.encode_batch(x_test_text, to_len = 28))

    testset = Dataset(x_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = False, num_workers=0)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )

    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    # load weights into model
    y_pred_list = list()
    with torch.no_grad():
        for i, x_test in enumerate(testloader):
            x_test = x_test.to(device).to(torch.int64)
            model = model.to(device)
            y_test_pred = model(x_test)
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_list = list(itertools.chain(*y_pred_list))
    result = [dataset.idx2label(index) for index in y_pred_list]
    test_id = [item['id'] for item in dataset.data]
    output = zip(test_id, result)
    output = dict(output)
    # TODO: predict dataset

    # # TODO: write prediction to file (args.pred_file)
    # df = pd.DataFrame(list(zip(test_id, result)), columns =['id', 'intent']) 
    # df.to_csv(args.pred_file, index=False)

    csvfile = open(args.pred_file, 'w')
    header = ['id', 'intent']
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for key, value in output.items():
        writer.writerow([key, value])
    csvfile.close



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/state_dict_model.pt"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
