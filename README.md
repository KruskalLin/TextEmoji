# TextEmoji
Data Mining on emoji text.

*Metrics*:

| Model                | MNB  | LinearSVC | NN   | LR   | Voting | Bagging(MNB) | GBDT | XGBoost | LightGBM    |
| -------------------- | ---- | --------- | ---- | ---- | ------ | ------------ | ---- | ------- | ----------- |
| **Average F1-Score** | 0.16 | 0.11      | 0.09 | 0.12 | 0.15   | 0.15         | 0.13 | 0.15    | 0.15(0.151) |

|                      | **TextCNN** | **MultiLayerCNN** | **LSTM**   | **LSTM-Attention** | **BiLSTM** |
| -------------------- | ----------- | ----------------- | ---------- | ------------------ | ---------- |
| **Accuracy**         | 0.18        | 0.18              | 0.16       | 0.17               | 0.188      |
| **Average F1-Score** | 0.167       | 0.181             | 无上榜检验 | 无上榜检验         | **0.1845** |