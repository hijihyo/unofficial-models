# Sequence-to-Sequence Learning with Attention

## Summary

SMT (Statistical Machine Translation) 시스템이 지배적이던 기계 번역 (Machine Translation) 분야에서 점차 신경망 기계 번역 (Neural Machine Translation; NMT) 시스템이 떠오르기 시작했다. 여러 개의 하위 요소로 구성된 SMT 시스템과는 다르게, NMT 시스템은 문장을 읽고 이를 번역하는 하나의 큰 신경망만으로 이루어져 있다.

당시 대부분의 NMT 시스템은 인코더-디코더 (encoder-decoder) 형식을 보이고 있었는데, 이는 입력 시퀀스를 고정 길이의 벡터로 인코딩한 뒤 해당 벡터를 출력 시퀀스로 변환하는 방법이었다. 그러나 이와 같은 방식에서는 입력 시퀀스의 길이에 상관 없이 고정 길이의 벡터로 인코딩해야 하기 때문에 입력 시퀀스가 길어질수록 모델의 성능이 떨어진다는 문제점이 있었다.

해당 논문에서는 모델에 어텐션 메커니즘 (attention mechanism) 을 적용하여 문제점을 해결하고 NMT 시스템의 성능을 극적으로 향상하였다. 논문에서 제안한 모델의 디코더는 고정 길이의 벡터가 아닌, 벡터 리스트 중에서 매 동작마다 가장 관련 있는 입력 시퀀스 부분을 부드럽게 탐색하여 (soft-search) 요약 벡터를 생성한다.

이와 같은 모델은 고정 길이의 벡터에 정보를 압축할 필요가 없기 때문에 입력 시퀀스의 길이에 크게 제한받지 않는다. 또한 입력 시퀀스와 출력 시퀀스 사이의 배열 관계를 명시적으로 학습하기 때문에 시각화하여 확인할 수 있다.

해당 모델을 WMT’14 English to French Machine Translation 데이터셋으로 학습시켰을 때 당시 SMT 시스템의 최고점에 훨씬 가까워진 BLUE 점수를 받을 수 있었다. (28.45)

## References
- Neural Machine Translation by Jointly Learning to Align and Translate [[link]](https://arxiv.org/abs/1409.0473)
- PyTorch Seq2Seq [[link]](https://github.com/bentrevett/pytorch-seq2seq)
