import os
import pandas as pd
import matplotlib.pyplot as plt

window_lens = [30, 100, 200, 500, 1000, 2000, 3317]

splits = ['Train', 'Val', 'Test']
topk_keys = ['Top_1_Accuracy', 'Top_2_Accuracy', 'Top_3_Accuracy']

acc_data = {
    split: {k: [] for k in topk_keys} for split in splits
}

for win_len in window_lens:
    csv_path = f"evaluations/evaluation_results_known_control_{win_len}/known_control_classification_results.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for split in splits:
        row = df[df['Dataset'] == split]
        if row.empty:
            for k in topk_keys:
                acc_data[split][k].append(None)
        else:
            for k in topk_keys:
                acc_data[split][k].append(row.iloc[0][k])

for split in splits:
    plt.figure()
    for k in topk_keys:
        plt.plot(window_lens, acc_data[split][k], marker='o', label=k.replace('_', ' '))

    plt.title(f'{split} Accuracy vs Window Length')
    plt.xlabel('Window Length')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(window_lens)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{split.lower()}_accuracy_vs_window_len.png')
    plt.show()
