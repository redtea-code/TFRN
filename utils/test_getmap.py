import time
from utils.metrics import *


def test_getmap(test_model, decode_mode, test_loader, device, ):
    """
    date_index will be used if only_metric == False
    """
    t1 = time.time()

    test_model.eval()
    with torch.no_grad():
        for x_seq, y_seq_past, y_seq_future, _, static_fea in test_loader:
            x_seq, y_seq_past, y_seq_future = x_seq.to(device), y_seq_past.to(device), y_seq_future.to(device)
            static_fea = static_fea.to(device)
            batch_size = y_seq_past.shape[0]
            tgt_len = y_seq_past.shape[1] + y_seq_future.shape[1]
            tgt_size = y_seq_future.shape[2]
            pred_len = y_seq_future.shape[1]

            y_hat, attn = test_model(x_seq, y_seq_past, static_fea)
            break
    t2 = time.time()
    print(f"Testing used time:{t2 - t1}")

    return attn