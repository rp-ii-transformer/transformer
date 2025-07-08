import torch
import torch.nn as nn
from scaled_dot_product import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Implementação da camada de Multi-Head Self-Attention.
    """

    def __init__(self, embed_dim, num_heads):
        """
        Inicializador da classe.

        Args:
            embed_dim (int): A dimensionalidade dos embeddings de entrada. (d_model)
            num_heads (int): O número de cabeças de atenção a serem usadas.
        """
        super(MultiHeadAttention, self).__init__()

        # Garante que a dimensão do embedding pode ser dividida igualmente entre as cabeças
        assert embed_dim % num_heads == 0, "A dimensão do embedding deve ser divisível pelo número de cabeças."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Camadas lineares para criar os vetores Query, Key e Value a partir da entrada
        # No self-attention, todos vêm da mesma fonte, mas as projeções são diferentes.
        self.W_q = nn.Linear(embed_dim, embed_dim)  # Projeção para Query
        self.W_k = nn.Linear(embed_dim, embed_dim)  # Projeção para Key
        self.W_v = nn.Linear(embed_dim, embed_dim)  # Projeção para Value

        # Camada linear final para a saída, após concatenar as cabeças
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x):
        """
        Divide a última dimensão em (num_heads, head_dim) e transpõe para o cálculo de atenção.

        Entrada x: (batch_size, seq_length, embed_dim)
        Saída: (batch_size, num_heads, seq_length, head_dim)
        """
        batch_size, seq_length, _ = x.shape
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        """
        Passo forward da camada.
        Para self-attention, query, key e value serão o mesmo tensor de entrada.

        Args:
            query (torch.Tensor): Tensor de entrada para a Query. Shape: (batch_size, seq_length, embed_dim)
            key (torch.Tensor): Tensor de entrada para a Key. Shape: (batch_size, seq_length, embed_dim)
            value (torch.Tensor): Tensor de entrada para a Value. Shape: (batch_size, seq_length, embed_dim)
            mask (torch.Tensor, opcional): Máscara para ignorar certas posições (ex: padding).

        Returns:
            torch.Tensor: O tensor de saída após a atenção. Shape: (batch_size, seq_length, embed_dim)
        """
        # 1. Projeção Linear: Passa as entradas pelas camadas lineares W_q, W_k, W_v
        # Shape: (batch_size, seq_length, embed_dim)
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # 2. Dividir em múltiplas cabeças
        # Shape resultante para q, k, v: (batch_size, num_heads, seq_length, head_dim)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 3. Cálculo do Scaled Dot-Product Attention

        context_vector = scaled_dot_product_attention(q, k, v, mask)

        # 4. Concatenar as cabeças e passar pela camada de saída

        # Transpõe para juntar as cabeças novamente
        # -> (batch_size, seq_length, num_heads, head_dim)
        context_vector = context_vector.transpose(1, 2).contiguous()

        # Concatena as cabeças, resultando na dimensão original do embedding
        # -> (batch_size, seq_length, embed_dim)
        batch_size, seq_length, _, _ = context_vector.shape
        concatenated_output = context_vector.view(batch_size, seq_length, self.embed_dim)

        # 5. Passar pela projeção linear final W_o
        # -> (batch_size, seq_length, embed_dim)
        output = self.W_o(concatenated_output)

        return output


### Exemplo de Uso
if __name__ == '__main__':
    # --- Parâmetros de Exemplo ---
    batch_size = 4  # 4 sentenças em um lote
    seq_length = 10  # Cada sentença tem 10 tokens
    embed_dim = 512  # Cada token é representado por um vetor de 512 dimensões (d_model)
    num_heads = 8  # Usaremos 8 cabeças de atenção

    # --- Simulação da Entrada ---
    # Na prática, este 'x' seria a saída de uma camada de embedding.
    # É o seu "array de tokens" representado numericamente.
    # Shape: (lote, tamanho_da_sequencia, dimensao_embedding)
    x = torch.rand(batch_size, seq_length, embed_dim)

    print(f"Shape da entrada (simulando um array de tokens embedados): {x.shape}\n")

    # --- Instanciando e Usando a Camada ---

    # Criar a instância da nossa camada
    multi_head_attention_layer = MultiHeadAttention(embed_dim, num_heads)

    # Realizar a chamada forward.
    # Para self-attention, Query, Key e Value são o mesmo tensor 'x'.
    output = multi_head_attention_layer(query=x, key=x, value=x, mask=None)

    # --- Saída ---

    print(f"Shape da saída (o novo array contextualizado): {output.shape}\n")
    print("O shape da entrada e da saída são os mesmos, como esperado.")
    print("A saída agora contém as informações de contexto para cada token, calculadas pelo mecanismo de atenção.")
