import torch
import psutil
from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import time
import csv
import os

from main import transformer_start
from transformer.pipeline.inference.translator import Translator

device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando GPU? {torch.cuda.is_available()}")
print(f"Dispositivo: {device}")

dataset = load_dataset("tatoeba", lang1="en", lang2="pt", trust_remote_code=True)["train"]
dataset = dataset.shuffle(seed=42).select(range(1000))

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

model_names = [
    "Helsinki-NLP/opus-mt-tc-big-en-pt",  # Ainda usado apenas para o tokenizer
]

output_file = "../resultados_traducao.csv"
if not os.path.exists(output_file):
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Modelo",
            "Tempo Total",
            "Tempo por Sentença (s)",
            "Palavras por Segundo",
            "Uso de Memória (RAM)",
            "Total de Sentenças",
            "Total de Palavras",
            "BLEU",
            "chr-F"
        ])

def compute_metrics(dataset, translator, tokenizer, num_samples=1000, batch_size=4):
    # agora recebe um Translator, não um torch.Module
    references, hypotheses = [], []

    for i in tqdm(range(0, min(num_samples, len(dataset)), batch_size), desc="Calculando métricas"):
        batch = dataset[i:i+batch_size]
        inputs_text  = [item['en'] for item in batch['translation']]
        targets_text = [item['pt'] for item in batch['translation']]

        # para cada sentença, usa o Translator
        for src in inputs_text:
            hyp = translator.translate(src, method="beam", beam_size=4, max_len=64)
            hypotheses.append(hyp)

        references.extend([[ref] for ref in targets_text])

    bleu_score = bleu_metric.compute(predictions=hypotheses, references=references)["bleu"] * 100
    chrf_score = chrf_metric.compute(predictions=hypotheses, references=references)["score"]
    return bleu_score, chrf_score

def print_table(data):
    print("\n" + "-" * 100)
    for metric, value in zip(data["Metric"], data["Valor"]):
        print(f"| {metric.ljust(45)} | {str(value).ljust(45)} |")
    print("-" * 100)

for model_name in model_names:
    print(f"\nAvaliando o modelo: {model_name}")
    process = psutil.Process()
    memory_before = process.memory_info().rss

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    # instancia o Transformer manual com os hiper‑parâmetros do paper
    model = transformer_start(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_len=128
    )

    # cria o Translator que envolve encode/decode + beam search
    stoi = tokenizer.get_vocab()
    itos = {v:k for k,v in stoi.items()}
    translator = Translator(
        model=model,
        stoi=stoi,
        itos=itos,
        pad_idx=tokenizer.pad_token_id,
        sos_idx=tokenizer.cls_token_id,
        eos_idx=tokenizer.sep_token_id
    )

    # (opcional) dataparallel se tiver >1 GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    start_time = time.time()
    bleu_score, chrf_score = compute_metrics(dataset, translator, tokenizer)  # passa o Translator
    end_time = time.time()

    elapsed_time = end_time - start_time
    num_sentences = len(dataset)
    num_words = sum(len(ex["translation"]["pt"].split()) for ex in dataset)
    memory_after = process.memory_info().rss

    ram_mb = (memory_after - memory_before) / (1024 ** 2)

    data = {
        "Metric": [
            "Nome do Modelo",
            "Tempo Total",
            "Tempo por Sentença (s)",
            "Palavras por Segundo",
            "Uso de Memória (RAM) (MB)",
            "Total de Sentenças",
            "Total de Palavras",
            "BLEU Score",
            "chr-F Score",
        ],
        "Valor": [
            model_name,
            f"{elapsed_time:.2f}s",
            f"{elapsed_time/num_sentences:.4f}",
            f"{num_words/elapsed_time:.2f}",
            f"{ram_mb:.2f}",
            num_sentences,
            num_words,
            f"{bleu_score:.2f}",
            f"{chrf_score:.5f}",
        ]
    }

    print_table(data)

    with open(output_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            data["Valor"][0],
            data["Valor"][1],
            data["Valor"][2],
            data["Valor"][3],
            data["Valor"][4],
            data["Valor"][5],
            data["Valor"][6],
            data["Valor"][7],
            data["Valor"][8],
        ])
