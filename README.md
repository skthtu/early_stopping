## Early Stopping の実装

### 使い方

    from early_stopping import *

    es = early_stopping( 引数についてはpythonファイル内を参照 ) #インスタンスの作成

    es( valid_loss, model, model_path= "model_path") #esを呼び出しEarlyStoppingするか確認
    if es.early_stop: #もし停止フラグが立っているなら停止
      break


### References
[Feed back Price 2021 1st Solution](https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook )<br>
[GitHubで公開されている簡易的なEarly stopping](https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py)<br>
[PyTorchでEarlyStoppingを実装する by Qiita@ku_a_i](https://qiita.com/ku_a_i/items/ba33c9ce3449da23b503 )<br>
[PyTorchでearly stoppingを簡単に実装 ワクワクネズミの技術ブログより](https://wakunezu.tech/post-8)<br>
