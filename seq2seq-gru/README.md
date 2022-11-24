# Sequence-to-Sequence Learning with GRU

## Summary

다양한 분야에서 Deep neural networks (DNNs) 를 도입하려는 움직임이 나타나면서, 구 단위 SMT 시스템 (phrase-based SMT system; Statistical Machine Translation) 에도 순방향 신경망 (feedfoward neural networks) 을 도입하려는 연구가 활발하다. 해당 논문에서는 전통적인 구 단위 SMT 시스템의 일부분으로 도입할 수 있는 신경망 구조를 제안하였다.

해당 논문에서는 이 구조를 RNN 인코더-디코더 (RNN Encoder-Decoder) 라고 지칭하며, 각각 인코더와 디코더로 동작하는 두 개의 RNN으로 이루어져 있다고 설명한다. 인코더 역할의 RNN은 가변 길이의 입력 시퀀스를 고정 길이의 벡터로 매핑하는 연산을 수행하고, 디코더 역할의 RNN은 이 벡터를 다시 가변 길이의 출력 시퀀스로 매핑한다. 두 RNN은 입력 시퀀스가 주어졌을 때 출력 시퀀스의 조건부 확률을 최대화하도록 "함께" 훈련된다.

또한 해당 논문에서는 추후에 GRU (Gated Recurrent Unit) 라고 불리게 되눈 새로운 RNN 구조를 제시한다. 이 구조는 LSTM에서 영감을 받아 간소화한 것으로, 총 두 개의 게이트로 이루어져 있다. 그 중 하나인 리셋 게이트 (reset gate) 는 현재 입력으로 새로운 은닉 스테이트 (hidden state) 를 계산할 때 이전 은닉 스테이트를 얼마나 반영할 것인지 조절한다. 다른 하나인 업데이트 게이트 (update gate) 는 다음 은닉 스테이트를 계산하는 데에 있어서 이전 은닉 스테이트와 새로운 은닉 스테이트 간의 비중을 조절한다.

제안한 방법을 WMT'14 English to French Machine Translation 작업에 적용하였다. SMT 시스템으로 기본 세팅을 이용한 Moses를 사용하였으며, 이와 함께 논문에서 제안한 구조를 사용했을 때 테스트 세트에서 33.87에 달하는 점수를 얻었다. (기존 시스템만 사용했을 때보다 0.57만큼 높은 점수이다.)

(단, 해당 논문에서는 영어 문장을 프랑스어 문장으로 번역하는 확률을 학습시킨 것이 아니라, 영어 구를 프랑스어 구로 번역하는 확률을 학습시켰다.)

## References
- Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation [[link]](https://arxiv.org/abs/1406.1078)
- PyTorch Seq2Seq [[link]](https://github.com/bentrevett/pytorch-seq2seq)
