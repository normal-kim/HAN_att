

def set_params_dict(batch_size, epochs, embedding_dim, n_layers, 
                   output_size, lr, clip_val, print_every = 20, total_dim = 200, fc = None):
    return { 'batch_size': batch_size, 'epochs': epochs, 
           'embedding_dim': embedding_dim, 'n_layers': n_layers, 
           'output_size': output_size, 'lr': lr, 'print_every': print_every,
           'clip_val' : clip_val, 'total_dim': total_dim, 'fc': fc}



