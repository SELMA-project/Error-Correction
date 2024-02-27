import argparse
from dataclasses import dataclass

import torch
from model import TagDecoder


@dataclass
class Config:
    base_model: str
    strip_accents: bool
    device: str
    tag_pdrop: float
    decoder_proj_pdrop: float
    tag_hidden_size: int
    tag_size: int
    change_weight: float
    vocab_size: float
    pad_token_id: int
    alpha: float
    tokenizer_name: str
    max_add_len: int
    max_src_len: int
    model_path: str


def load_model() -> TagDecoder:
    config = Config(
        base_model=args.pretrained_bert,
        strip_accents=args.strip_accents,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tag_pdrop=0.2,
        decoder_proj_pdrop=0.2,
        tag_hidden_size=args.hidden_size,
        tag_size=3,
        change_weight=3.0,
        vocab_size=args.vocab,
        pad_token_id=0,
        alpha=3.0,
        tokenizer_name=args.pretrained_bert,
        max_add_len=10,
        max_src_len=args.max_src_len,
        model_path=args.trained_model,
    )

    model = TagDecoder(config)
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()

    return model


def get_correction(text_input: str) -> str:
    model = load_model()
    const_decoder_output = model.generate(text_input)
    return const_decoder_output


def parse_args() -> argparse.Namespace:
    """Get command line options.

    Returns:
        argparse.Namespace: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sentence",
        type=str,
        help="sentence to be corrected",
    )
    parser.add_argument(
        "-trained_model",
        type=str,
        help="path the the trained model, e.g. ./scripts/models.YOUR_DATASET/best.pt",
    )
    parser.add_argument(
        "-pretrained_bert",
        type=str,
        default="dbmdz/bert-base-german-uncased",
        help="name of pre-trained BERT model / tokenizer",
    )
    parser.add_argument(
        "-vocab",
        type=int,
        default=31_102,
        help="vocabulary size of BERT model",
    )
    parser.add_argument(
        "-max_src_len",
        type=int,
        default=512,
        help="max number of tokens",
    )
    parser.add_argument(
        "-hidden_size",
        type=int,
        default=768,
        help="hidden units",
    )
    parser.add_argument(
        "-strip_accents",
        type=bool,
        default=False,
        help="indicator if accents should be removed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    correction = get_correction(args.sentence)
    print(
        f"Sentence before correction: \t{args.sentence}\nSentence after correction: \t{correction}"
    )
