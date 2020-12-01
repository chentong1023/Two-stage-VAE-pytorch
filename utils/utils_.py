import os

def save_logfile(log_loss, save_path):
    with open(save_path, "wt") as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += " %.3f" % digit
            f.write(w_line + "\n")