### Early Stopping の実装

## 使い方

    from early_stopping import *

    es = early_stopping()

    es( valid_loss,model, model_path= "model_path")
    if es.early_stop:
      break
