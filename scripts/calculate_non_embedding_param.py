import math
def calculate_non_embedding_param(n_layer=6, n_embd=512,):
    """
    Assuming Llama architecture, MHA, attention dim == n_embd. FF intermediate size is calculated as (n_embd * 8 / 3) and rounded up to the multiple of 256. 
    attn_param = 4 * n_embd * n_layer * n_attn
    ff_param = n_embd * n_ff * 3
    """
    n_ff = math.ceil(n_embd * 8 / 3 / 256) * 256
    n_attn = n_embd
    attn_param = 4 * n_embd * n_layer * n_attn
    ff_param = n_embd * n_ff * 3  * n_layer
    total = attn_param + ff_param
    print(f"{total / 1e6} million")
    
if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(calculate_non_embedding_param)