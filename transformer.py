from transformers import AutoModelForSeq2SeqLM

# TODO ELENCAR ETAPAS DO TRANSFORMER E DESENVOLVER AQUI

def transformer_start(model_name, device):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    print(f"Modelo real carregado: {type(model)}")
    return model

