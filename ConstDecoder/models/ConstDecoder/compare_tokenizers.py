"""
Helper script to check which tokenizer config fits best.
"""

from pathlib import Path

from transformers import AutoTokenizer

example_sentences = {
    "am": "መማር እፈልጋለሁ የሚል ነበር።",
    "de": "Das ist ein Beispielsatz mit Umlauten å, ä, ö, ü.",
    "en": "The jérôme-lejeune foundation takes part in the bioethics debate",
    "es": "la cúpula del techo se apoya encima de una cúpula invertida o tazón",
    "fr": "une contrefaçon fut publiée en belgique",
    "ps": "هوښيار نه ختا وزي ، چې ختا وزې غاښونه يې غبرگ وزي",
    "pt": "a saída não será imediata haverá um período de transição de algumas semanas",
    "uk": "так ми поговоримо про давальний відмінок іменників чоловічого роду",
    "ur": "اندھا دھند آنکھیں بند کر بلا گھمانے والا بلے باز یا پھر وہاب ریاض بن جائے گا۔",  # noqa: RUF001
}
tokenizer_names = [
    "bert-base-multilingual-uncased",
    "dbmdz/bert-base-german-uncased",
    "bert-large-uncased",
]
accents = [True, False]

STORE_PATH = Path("./tokenizer_comparison_new.txt")

with Path.open(STORE_PATH, "a") as f:
    for language, sentence in example_sentences.items():
        print(f"\n{language=}, {sentence=}", file=f)  # noqa: T201

        for tokenizer_name in tokenizer_names:
            print(f"\t{tokenizer_name=}", file=f)  # noqa: T201
            for strip_acc in accents:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    do_lower_case=True,
                    do_basic_tokenize=False,
                    strip_accents=strip_acc,
                )

                tokens = tokenizer.tokenize(sentence)

                useful = ""
                if "[UNK]" in tokens:
                    useful = " not"

                print(  # noqa: T201
                    f"\t\t{useful} useful, \t {strip_acc=}, {tokens=}",
                    file=f,
                )
