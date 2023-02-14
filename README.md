# LabNLPCompetitions
研究室の新人研修のコンペの記録

コンペに利用したデータは念のため削除しています

# Twitter感情極性分析

- タスク
    - 日本語のTwitterテキストについての感情極性分類
    - 書き手の感情極性を5クラス分類（-2, -1, 0, 1, 2）
    - 評価指標は Quadratic Weighted Kappa
- データセット
    - 訓練用が3万件、検証用が1,250件、評価用が1,250件、提出用が2,500件
    - Train/Dev/Testの分割変更は禁止
- リーダーボードのスコアで評価

# コード共有
## 第1回　※ニューラルネットワークを使用しない
最終提出で利用した実行ファイル

- [StepEstimator_v.0.0.1_3class_LogisticR.ipynb](https://github.com/suzuhiki/lab-competition-01/blob/main/01/StepEstimator_v.0.0.1_3class_LogisticR.ipynb)
- [StepEstimator_v.0.0.1_negative_LightGBM.ipynb](https://github.com/suzuhiki/lab-competition-01/blob/main/01/StepEstimator_v.0.0.1_negative_LightGBM.ipynb)
- [StepEstimator_v.0.0.1_positive_LightGBM.ipynb](https://github.com/suzuhiki/lab-competition-01/blob/main/01/StepEstimator_v.0.0.1_positive_LightGBM.ipynb)

stopword辞書の変更で少しスコアが下がっていますが、この実行ファイルでも同じようなスコアがでます。
```StepEstimator```が一番スコアが出たので簡単に概要を書いておきます。

## StepEstimator
StepEstimatorというのは勝手に命名したモデルを複数使って推論する方法のことです。
以下の図のような構造をしています。

![B3コンペ_5クラス分類図_whiteback](https://user-images.githubusercontent.com/82814541/209739575-1eae98e8-daa5-45e6-bd80-23b21a8829db.png)

3クラス分類器でpositive、neutral、negativeの3クラスに分類した後、positiveとnegativeはそれぞれ2クラス分類したあと結果を出力します。

3クラス分類にLogisticRegression、2クラス分類にLightGBMを使ったときにBestスコアが出ました。その時devデータについての結果は以下の2枚のようになりました。

![1000](https://user-images.githubusercontent.com/82814541/209740318-3b168d98-c01a-4ca1-b7c2-2a49fc8ac4dd.png)

![1000](https://user-images.githubusercontent.com/82814541/209740326-e2704fce-29bc-4440-8b96-bb9648624867.png)

## 第2回 ※外部データを使用しない（事前学習も不可）
[v.0.0.1_selfattention.ipynb](02/script/v.0.0.1_selfattention.ipynb)を実装しました。
参考にしたページ：[PyTorchでSelf Attentionによる文章分類を実装してみた](https://qiita.com/m__k/items/98ff5fb4a4cb4e1eba26)
