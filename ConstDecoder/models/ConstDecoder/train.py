"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
This software is licensed under the BSD 3-Clause License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
PURPOSE.
"""

import argparse
import os
import shutil

import numpy as np
from data_reader import *
from jiwer import wer
from model import *
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EarlyStopping:
    def __init__(self, tolerance=3, min_delta=0):
        """
        Args:
            tolerance (int):   How long to wait after last time validation loss improved.
                               Default: 3
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
        """
        self.tolerance = tolerance
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_delta = min_delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.tolerance}")
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_epoch(model, train_data_loader, optimizer):
    loss_list = []
    gen_losses = []
    tag_losses = []

    model.train()
    for i, batch in enumerate(train_data_loader):
        optimizer.zero_grad()
        tag_loss, gen_loss, total_loss = model(batch)
        gen_losses.append(gen_loss)
        tag_losses.append(tag_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_list.append(total_loss)
        if i % 20 == 0:  # monitoring
            print(
                f"train step: {i}, tag loss is {tag_loss.item()}, gen loss is {gen_loss.item()}, total loss is {total_loss.item()}"
            )

    avg_train_loss = np.average(
        torch.stack([loss.cpu() for loss in loss_list]).detach().numpy()
    )
    avg_gen_loss = np.average(
        torch.stack([gen_loss.cpu() for gen_loss in gen_losses]).detach().numpy()
    )
    avg_tag_loss = np.average(
        torch.stack([tag_loss.cpu() for tag_loss in tag_losses]).detach().numpy()
    )
    return avg_train_loss, avg_gen_loss, avg_tag_loss


def valid_epoch(model, valid_data_loader):
    model.eval()
    preds = []
    golds = []
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            raws = batch["RAWS"]
            asrs = batch["ASRS"]
            for asr, raw in zip(asrs, raws):
                pred = model.generate(asr)
                preds.append(pred.strip())
                golds.append(raw)

    return wer(golds, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # model
    parser.add_argument("--base_model", type=str, default="", help="")
    parser.add_argument(
        "--strip_accents",
        type=str_to_bool,
        default=str_to_bool(os.getenv("STRIP_ACCENTS", "False")),
        help="If accents should be removed",
    )
    parser.add_argument("--tag_pdrop", type=float, default=0.2, help="")
    parser.add_argument("--decoder_proj_pdrop", type=float, default=0.2, help="")
    parser.add_argument("--tag_hidden_size", type=int, default=768, help="")
    parser.add_argument("--tag_size", type=int, default=3, help="")
    parser.add_argument("--vocab_size", type=int, default=30522, help="")
    parser.add_argument("--pad_token_id", type=int, default=0, help="")
    parser.add_argument("--alpha", type=float, default=3.0, help="")
    parser.add_argument("--change_weight", type=float, default=1.5, help="")

    # data
    parser.add_argument("--train_data_file", type=str, default="", help="")
    parser.add_argument("--eval_data_file", type=str, default="", help="")
    parser.add_argument("--max_src_len", type=int, default=256, help="")
    parser.add_argument("--max_add_len", type=int, default=10, help="")
    parser.add_argument("--tokenizer_name", type=str, default="", help="")

    # train
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--lr", type=float, default=5e-5, help="")
    parser.add_argument("--max_num_epochs", type=int, default=10, help="")
    parser.add_argument("--save_dir", type=str, default="", help="")
    parser.add_argument("--device", type=str, default="", help="")

    args = parser.parse_args()
    args.device = "cuda:" + args.device

    # load data
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        do_lower_case=True,
        do_basic_tokenize=False,
        add_special_tokens=False,
        strip_accents=args.strip_accents,
        max_length=args.max_src_len,
    )
    train_examples = get_examples(
        examples_path=args.train_data_file,
        tokenizer=tokenizer,
        max_src_len=args.max_src_len,
        max_add_len=args.max_add_len,
    )
    eval_examples = get_examples(
        examples_path=args.eval_data_file,
        tokenizer=tokenizer,
        max_src_len=args.max_src_len,
        max_add_len=args.max_add_len,
    )

    train_dataset = ExampleDataset(train_examples)
    valid_dataset = ExampleDataset(eval_examples)
    train_data_loader = DataLoader(
        train_dataset,
        collate_fn=collate_batch,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        collate_fn=collate_batch,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # define model, loss_fn, optimizer
    model = TagDecoder(args)
    model = model.to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_epochs = args.max_num_epochs
    lr_opt_scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    eval_loss_list = []

    all_avg_total_losses = []
    all_avg_gen_losses = []
    all_avg_tag_losses = []
    all_val_wers = []
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.02)

    for epoch in range(1, args.max_num_epochs + 1):
        print(f"=========train at epoch={epoch}=========")

        avg_train_loss, avg_gen_loss, avg_tag_loss = train_epoch(
            model, train_data_loader, optimizer
        )
        all_avg_total_losses.append(avg_train_loss)
        all_avg_gen_losses.append(avg_gen_loss)
        all_avg_tag_losses.append(avg_tag_loss)

        print(
            f"train {epoch} average loss is {avg_train_loss}, avg gen_loss: {avg_gen_loss} avg_tag_loss: {avg_tag_loss}"
        )

        print(f"=========eval at epoch={epoch}=========")
        avg_val_loss = valid_epoch(model, valid_data_loader)
        all_val_wers.append(avg_val_loss)
        early_stopping(avg_val_loss)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        lr_opt_scheduler.step(avg_val_loss)

        torch.save(model.state_dict(), args.save_dir + f"/{epoch}.pt")
        print(f"eval {epoch} wer is {avg_val_loss}")
        eval_loss_list.append((epoch, avg_val_loss))

    eval_loss_list.sort(key=lambda x: x[-1])
    print(eval_loss_list)
    best_epoch_path = os.path.join(args.save_dir, str(eval_loss_list[0][0]) + ".pt")
    print(f"best epoch path is {best_epoch_path}")
    shutil.copyfile(best_epoch_path, os.path.join(args.save_dir, f"best.pt"))
