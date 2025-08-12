from ..common import xp
from ..softmax import softmax

def beam_search(model, src_ids, src_mask, max_len, sos_id, eos_id, beam_size=4, alpha=0.6):
    """
    Executa a geração de sequência usando Beam Search otimizado.

    Args:
        model: A instância do modelo Transformer (com métodos encode/decode/project).
        src_ids (xp.ndarray): IDs dos tokens da sentença de entrada. Shape: (1, src_len).
        src_mask (xp.ndarray): Máscara de padding para a entrada.
        max_len (int): Comprimento máximo da sequência a ser gerada.
        sos_id (int): ID do token de início de sentença (<sos>).
        eos_id (int): ID do token de fim de sentença (<eos>).
        beam_size (int): Tamanho do feixe (número de hipóteses).
        alpha (float): Fator de penalidade de comprimento.

    Returns:
        list[int]: A lista de IDs de token da melhor hipótese encontrada.
    """
    # 1. ENCODER: Executado apenas uma vez.
    memory = model.encode(src_ids, src_mask, training=False)

    # 2. INICIALIZAÇÃO:
    # 'sequences' guarda as hipóteses ativas.
    # Formato: (tensor da sequência, score de log-probabilidade acumulado)
    sequences = [(xp.array([[sos_id]], dtype=xp.int32), 0.0)]
    # 'completed_sequences' guarda as hipóteses que já encontraram <eos>.
    completed_sequences = []

    # 3. LOOP DE DECODIFICAÇÃO
    for step in range(max_len - 1):
        if not sequences:
            break  # Para se não houver mais hipóteses ativas.

        all_candidates = []
        
        # 4. EXPANSÃO: Expande cada hipótese ativa
        for seq_tensor, score in sequences:
            # Cria a máscara causal para a sequência atual
            tgt_len = seq_tensor.shape[1]
            tgt_mask = xp.triu(xp.ones((1, 1, tgt_len, tgt_len), dtype=xp.bool_), k=1)

            # Executa o decoder e a projeção para obter os logits do próximo token
            decoder_output = model.decode(seq_tensor, memory, src_mask, tgt_mask, training=False)
            logits = model.project(decoder_output[:, -1, :])  # Apenas o último token
            
            # Calcula as log-probabilidades
            log_probs = xp.log(softmax(logits[0]))

            # Obtém os 'beam_size' melhores próximos tokens
            top_k_scores = xp.sort(log_probs)[-beam_size:][::-1]
            top_k_ids = xp.argsort(log_probs)[-beam_size:][::-1]
            
            # Cria novas hipóteses a partir dos melhores tokens
            for i in range(beam_size):
                next_id = int(top_k_ids[i])
                next_score = float(top_k_scores[i])
                
                new_seq_tensor = xp.concatenate([seq_tensor, xp.array([[next_id]], dtype=xp.int32)], axis=1)
                new_score = score + next_score
                all_candidates.append((new_seq_tensor, new_score))

        # 5. PRUNING: Seleciona as melhores hipóteses entre todas as candidatas
        ordered_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        
        sequences = [] # Reseta as hipóteses ativas para a próxima iteração
        for cand_seq, cand_score in ordered_candidates:
            # Se a hipótese termina com <eos>, move para a lista de "completas"
            if cand_seq[0, -1] == eos_id:
                completed_sequences.append((cand_seq, cand_score))
            # Senão, a mantém como ativa para a próxima iteração
            else:
                sequences.append((cand_seq, cand_score))
            
            # Mantém apenas 'beam_size' hipóteses ativas
            if len(sequences) >= beam_size:
                break
    
    # Se não houver sequências completas, usa as que estão ativas
    if not completed_sequences:
        completed_sequences = sequences

    # 6. SELEÇÃO FINAL: Aplica a penalidade de comprimento e retorna a melhor
    final_scores = []
    for seq_tensor, score in completed_sequences:
        length = seq_tensor.shape[1]
        penalty = ((5 + length) / 6)**alpha  # Variação comum da penalidade de comprimento
        final_scores.append(score / penalty)
    
    best_idx = int(xp.argmax(xp.array(final_scores)))
    best_sequence_tensor = completed_sequences[best_idx][0]
    
    # Retorna a lista de tokens, removendo o <sos> inicial
    return [int(i) for i in best_sequence_tensor[0, 1:]]