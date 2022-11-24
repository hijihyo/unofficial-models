# Sequence-to-Sequence Learning with LSTM

## Summary

Deep Neural Networks (DNNs) 는 입력과 출력 데이터를 고정 길이의 벡터로 인코딩하는 방식으로 음성 인식과 객체 탐지 등의 분야에서 우수한 성과를 내고 있었다. 그러나 이러한 방법론은 길이가 제각각인 시퀀스 처리에는 도입하기 어렵다는 한계가 있었다. 해당 논문에서는 일반적인 시퀀스 처리에 도입할 수 있도록 두 개의 Recurrent Neural Networks (RNNs) 을 사용한 네트워크 구조를 제안하였다.

이 네트워크는 두 개의 RNNs로 이루어져 있다. 하나는 한 번에 한 단위씩 (논문에서는 단어 기준) 읽는 방식으로 여러 번에 걸쳐 하나의 시퀀스를 입력받아 고정 길이의 벡터로 인코딩한다. 다른 하나는 인코딩된 벡터를 읽고 한 번에 한 단위씩 출력하는 방식으로 하나의 시퀀스를 만들어낸다.

해당 논문에서는 Long Short-Term Memory (LSTM) 이 긴 시간 동안의 의존성이 필요한 문제에서 유용하다는 점에 착안하여 네트워크의 RNN 구조로 LSTM을 선택하였다. (그리고 [2]에서 제안한 LSTM을 사용하였다.)

그리고 제안한 방법을 WMT'14 English to French Machine Translation 작업에 적용하였다. 그 결과 전체 테스트 세트에서 34.8에 달하는 BLEU 점수를 얻었다.

## References
- Sequence to Sequence Learning with Neural Networks [[link]](https://arxiv.org/abs/1409.3215)
- PyTorch Seq2Seq [[link]](https://github.com/bentrevett/pytorch-seq2seq)
