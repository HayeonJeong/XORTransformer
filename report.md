## 1. 예상

| 선택사항                         | 예상    | 근거                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. $D_k$의 최소값?               | 1     | XOR 문제의 입력값($x_1$ or $x_2$) 종류는 -1 또는 1 뿐이므로, 하나의 값으로 구분할 수 있을 것이라고 생각한다.                                                                                                                                                                                                                                                                                           |
| 2. weight matrix 학습 필요성?     | 필요함   | Query, key, value의 projection 가중치를 모두 고정($[1,1,...,1]^T$)한 경우, attention weight가 입력에 따라 충분히 차별화되지 못하며, 결과적으로 value 조합도 의미 있는 방향으로 조절되지 못할 것이라고 생각한다.                                                                                                                                                                                                                |
| 3. bias의 필요성?                | 필요 없음 | 입력에 따른 차별화를 위해서는 bias보다 weight matrix가 더 큰 역할을 할 것이라고 생각한다. 학습해야 할 파라미터 수도 대체로 weight matrix가 많기 때문에, 더 중요한 역할을 할 것 같다.                                                                                                                                                                                                                                             |
| 4. positional encoding의 필요성? | 필요 없음 | XOR 문제가 요구하는 토큰 수가 2개 뿐이고, 순서와 관련 없는 결과를 내기 때문에 필요하지 않다고 생각한다.                                                                                                                                                                                                                                                                                                      |
| 5. softmax의 필요성?             | 필요함   | Softmax는 입력값들을 지수 함수로 변형한 후 정규화하는 방식이라, 결과적으로 linear operation만으로는 만들 수 없는 non-linear operation 출력을 만든다. Self-attention에서는 이 softmax가 $q$와 $k$의 내적 결과를 non-linear weight로 바꿔주고, 이 weight로 $v$들을 섞어 새로운 벡터(ex. $y_1$)를 만들어 낸다.<br>그렇기 때문에 XOR이라는 비선형 문제는 softmax를 거치면서 해결될 수 있다고 생각한다.                                                                               |
| 6. layer normalization의 필요성? | 필요 없음 | LayerNorm은 입력 벡터(하나의 토큰 토큰) $x \in \mathbb{R}^{D_k}$에 대해 다음과 같이 평균과 분산을 계산한다:<br>$\mu = \frac{1}{D_k} \sum_{i=1}^{D_k} x_i, \quad \sigma^2 = \frac{1}{D_k} \sum_{i=1}^{D_k} (x_i - \mu)^2$<br>그런데, $D_k$가 작으면 “너무 작은 표본으로 평균 내는 것”이라서 정규화 효과가 불안정하거나 의미가 없다. 특히 $D_k = 1$이면 평균이 자기 자신이 되고, 분산은 0이기 때문에 정규화를 해도 항상 같은 값이 될 것이다. 결국 LayerNorm이 사실상 아무 역할도 하지 않을 것 같다. |
## 2. 구현, 실험, 분석

@`xor_transformer.py`
@`train.py`
@`results`

### 1, 3) 실험해 본 구조들의 결과를 보이고 분석하라. 왜 그 구조가 최소가 되는가? 그보다 작아지는 것이 왜 불가능한가?
- *success seed*: 0부터 10까지 순차적으로 seed를 바꿔가며 실험하면서 처음으로 XOR을 정확히 해결한 seed 번호
- *prob*: 각 입력 샘플에 대해 모델이 출력한 예측 확률. 예를 들어, 입력 [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]에 대해 출력된 확률을 순서대로 나열한 것.

| case num | *D_k* | *weights* | *bias*  | *pe*     | *softmax* | *layernorm* | prob                                    | success seed | time(s) |
| -------- | ----- | --------- | ------- | -------- | --------- | ----------- | --------------------------------------- | ------------ | ------- |
| 1        | 4     | true      | true    | true     | true      | true        | 0.0002,<br>0.9997,<br>0.9997,<br>0.0002 | 0            | 5.14    |
| 2-1      | 3     | true      | true    | true     | true      | true        | 0.0000,<br>0.9999,<br>0.9999<br>0.0000  | 0            | 6.73    |
| 2-2      | 2     | true      | true    | true     | true      | true        | 0.0007,<br>0.9993,<br>0.9993,<br>0.0007 | 0            | 42.37   |
| 2-3      | 1     | true      | true    | true     | true      | true        | 0.0005,<br>0.0005,<br>0.0005,<br>0.0005 | fail         | 2.92    |
| 3        | 2     | true      | true    | *false* | true      | true        | 0.0004,<br>0.9998,<br>0.9998,<br>0.0004 | 0            | 43.04   |
| 4        | 2     | true      | true    | *false*  | true      | *false*     | 0.0005,<br>0.9992,<br>0.9992,<br>0.0005 | 0            | 2.64    |
| 5        | 2     | true      | *false* | *false*  | true      | *false*     | 0.0005,<br>0.0005,<br>0.0005,<br>0.0005 | fail         | 2.32    |
| 6        | 2     | *false*   | true    | *false*  | true      | *false*     | 0.0102,<br>0.9898,<br>0.9898,<br>0.0102 | 0            | 2.08    |
| 7        | 2     | *false*   | true    | *false*  | *false*   | *false*     | 0.0004,<br>0.9996,<br>0.9996,<br>0.0004 | 0            | 1.94    |

#### case #1. baseline
```json
{
  "D_k": 4,
  "weights": true,
  "bias": true,
  "positional_encoding": true,
  "softmax": true,
  "layer_norm": true
}
```
모든 기능이 활성화된 구성($D_k=4$, 학습 가능한 weight와 bias, positional encoding, softmax, layer normalization 포함)은 가장 먼저 XOR 문제를 정확히 해결한 구조였다. 충분한 차원 수와 학습 가능한 파라미터를 통해 입력 간의 복잡한 관계를 유연하게 표현할 수 있었고, 각 구성 요소는 안정적인 학습을 가능하게 했다. 이후 실험에서는 이 구조를 기준으로 개별 요소를 제거하며 최소 조건을 탐색했다. 다만 baseline 실험에서는 계산량이 상대적으로 많아 학습 시간은 다소 길었다.
#### case #2. $D_k$ 줄이기
$D_k$ 값을 줄여가며 실험한 결과, $D_k = 2$까지는 안정적으로 학습이 되었지만, $D_k = 1$에서는 학습이 실패했다. 이는 차원이 너무 작으면 입력의 미묘한 차이를 구분하기 어려워지고, 그에 따라 attention score나 업데이트 결과가 모두 비슷해져버리기 때문으로 보인다. 다시 말해, 표현 공간이 너무 좁으면 서로 다른 입력도 비슷하게 처리되어 XOR 문제를 풀 수 없게 된다.
#### case #3. positional encoding 제거
XOR 문제는 입력의 순서가 결과에 영향을 주지 않기 때문에, positional encoding이 굳이 필요하지 않을 것이라 예상했었다. 실제로 positional encoding을 제거해도 성능에는 큰 차이가 없었고, 오히려 예측 확률이 더 또렷하게 나오는 결과를 보였다. 따라서 이 요소는 XOR 문제 해결에 필수적이지 않다고 판단할 수 있다.
#### case #4. layer_norm 제거
LayerNorm은 일반적으로 학습의 안정성을 높이는 데 기여하지만, $D_k = 2$처럼 표현 차원이 매우 작은 경우에는 그 효과가 제한적이거나 오히려 불안정할 수 있다고 예상했었다. 실제 실험에서도 LayerNorm 없이도 학습이 문제없이 진행되었고, 성능 저하 없이 XOR 문제를 해결할 수 있었다.
본 과제의 목적은 최소 모델을 찾는 데 있지만, LayerNorm을 제거했을 때 학습 속도는 눈에 띄게 향상되었다. 이는 LayerNorm이 내부적으로 평균과 분산을 계산하고, 정규화를 위한 연산을 수행하면서 추가적인 연산 비용이 발생하기 때문으로 보인다.
#### case #5. bias 제거
```python
Epoch 0, Loss: 0.7133
Epoch 100, Loss: 0.6932
Epoch 200, Loss: 0.6932
Epoch 300, Loss: 0.6931
Epoch 400, Loss: 0.6931
Epoch 500, Loss: 0.6931
Epoch 600, Loss: 0.6931
Epoch 700, Loss: 0.6931
Epoch 800, Loss: 0.6931
Epoch 900, Loss: 0.6931

Test:
Input: [-1.0, -1.0], Prob: 0.5000, Output: 1
Input: [-1.0, 1.0], Prob: 0.5000, Output: 1
Input: [1.0, -1.0], Prob: 0.5000, Output: 0
Input: [1.0, 1.0], Prob: 0.5000, Output: 0

seed 0: FAILED
Execution time for seed 0: 2.32 seconds
```

Bias를 제거한 경우, 학습이 초반부터 loss가 약 0.6931 수준에서 정체되었고 이후 전혀 감소하지 않는 현상이 관찰되었다. 출력 확률 역시 네 가지 입력 모두에 대해 0.5000으로 고정되었으며, 이는 모델이 입력 간의 차이를 전혀 구분하지 못하고 있는 상태로 해석할 수 있다. sigmoid 출력이 항상 0.5라는 것은 cross-entropy loss가 최대 엔트로피 상태에 머물고 있다는 뜻이고, 이로 인해 gradient가 일정하고 작아져 학습이 더 이상 진전되지 않는 양상이 나타났다고 볼 수 있다.

이러한 현상은 bias가 없는 상태에서 attention score 및 그로부터 계산되는 representation들이 입력에 따라 충분히 달라지지 않았기 때문일 가능성이 높다. 특히 XOR 문제의 입력은 $[-1, -1]$, $[-1, 1]$, $[1, -1]$, $[1, 1]$처럼 평균이 0인 대칭 구조를 가지고 있다. 여기에 사용된 $W_q$, $W_k$는 일반적으로 Kaiming(He) 초기화를 통해 평균이 0이고 분산이 작은 값들로 설정되기 때문에, 서로 다른 입력을 넣어도 $q_1$, $q_2$ (또는 $k_1$, $k_2$)가 매우 유사한 벡터로 계산될 수 있다. 이로 인해 attention score $q^\top k / \sqrt{D_k}$가 입력 간 차이를 잘 반영하지 못하고, softmax는 거의 균등한 분포를 출력하게 되는 것으로 보인다.

결과적으로 attention output이 모든 입력에 대해 유사하게 형성되면, 이후 linear + sigmoid 층을 거친 출력도 동일한 값에 수렴하게 된다. 이는 모델이 입력에 관계없이 항상 같은 출력을 내며, 그에 따라 손실 함수의 변화가 거의 없는 상태로 정체되는 현상을 유발한다.

이러한 관찰을 바탕으로 보면, bias는 attention 연산에서 입력 간의 차이를 효과적으로 반영하고 softmax 분포를 비대칭적으로 조정하는 데 중요한 역할을 할 수 있다. 본 실험에서는 bias가 제거된 경우 학습이 실패하는 양상을 확인할 수 있었고, 이를 통해 bias가 XOR 문제 해결을 위해 유의미한 기여를 한다고 판단할 수 있다.
#### case #6. weight 학습 제거
처음에는 linear projection 없이 attention을 구성하면 표현력이 부족할 것이라 생각했다. 특히 XOR처럼 입력 값이 대칭적인 구조를 가질 경우, softmax가 모든 경우에 비슷한 값을 출력하면서 입력 간 차이를 제대로 반영하지 못할 것 같았다. 하지만 실제로 실험해보니, weight가 고정되어 있어도 bias를 통해 충분한 비대칭성이 만들어졌고, softmax와 sigmoid의 비선형성이 더해지면서 학습이 문제없이 이루어졌다.

예를 들어, 입력 $x_1$, $x_2$가 서로 다른 스칼라 값을 가질 때, 고정된 $W_q = [1, 1]^T$를 사용하더라도 $q = W_q x + b$ 형태에서 bias가 입력 차이를 만들어낼 수 있다. 이 차이는 softmax를 거치며 비선형적인 attention 분포로 바뀌고, 결과적으로 attention output $y_1 = \alpha_1 v_1 + \alpha_2 v_2$도 입력마다 달라진다.

결론적으로, 학습 가능한 weight 없이도 bias 하나만으로 입력을 구분하는 데 필요한 차이를 만들 수 있었고, XOR처럼 단순한 문제에서는 이런 최소한의 구조만으로도 충분히 학습이 가능하다는 걸 확인할 수 있었다.
#### case #7. softmax 제거
처음에는 softmax가 attention 구조에서 핵심적인 비선형 요소이기 때문에, 이를 제거하면 입력 간 차이를 효과적으로 반영하지 못하고 학습이 어려워질 것이라 예상했다. 그러나 실제 실험에서는 softmax 없이도 XOR 문제를 해결하는 데 성공한 사례가 나타났고, 이 현상은 몇 가지 요인으로 설명될 수 있다.

우선, bias에 의해 query와 key가 입력마다 다르게 형성되면서, softmax 없이도 attention score—즉, dot product 결과—가 입력 쌍에 따라 충분히 구분되는 값으로 계산되었을 가능성이 있다. 이 경우, softmax 없이도 attention 결과에 차이를 부여할 수 있으며, 모델이 입력 간 구분을 학습하는 데 도움이 되었을 수 있다.

또한 softmax를 생략하면 attention score는 정규화 없이 직접 가중치로 사용되기 때문에, score 간의 상대적인 차이가 클 경우 오히려 더 직접적인 방식으로 입력 간 차이를 반영할 여지도 생긴다. 이후에 적용되는 linear layer와 sigmoid 함수는 강한 비선형성을 제공하므로, 전체 구조 내에서 충분한 표현력과 결정 능력이 확보될 수 있었던 것으로 보인다.

결과적으로, softmax가 항상 필수적인 것은 아니며, bias와 sigmoid 등의 요소가 함께 작동할 경우, 최소한의 구성으로도 XOR 문제를 해결할 수 있는 경우가 존재한다.
### 2) 최소구조의 구성 요소와 학습 대상 파라미터의 수는 얼마인가?
```json
{
  "D_k": 2,
  "weights": false,
  "bias": true,
  "positional_encoding": false,
  "softmax": false,
  "layer_norm": false
}
```

1. `"weights": false` → $W_q$, $W_k$, $W_v$는 고정 (requires_grad = False)
2. `"bias": true`:
	- $b_q$: `shape = (2,)` → 2개
	- $b_k$: `shape = (2,)` → 2개
	- $b_v$: `shape = (2,)` → 2개
3. **Linear layer (Linear($D_k$, 1) with bias)**:
	- weight: `shape = (1, 2)` → 2개
	- bias: `shape = (1,)` → 1개

따라서 학습 대상 파라미터 수는, **2 + 2 + 2 + 2 + 1 = 9개**

## 3. 수학적 분석

**전제 조건**

- 입력 $x_l$, $x_r \in \mathbb{R}$ (예: $x_l = -1$, $x_r = 1$)
- 차원 수: $D_k = 2$
- 고정 weight:
$$W_q = W_k = W_v = \begin{bmatrix} 1 \ 1 \end{bmatrix} \in \mathbb{R}^{2 \times 1}$$
- 학습 가능한 bias: $b_q, b_k, b_v \in \mathbb{R}^2$

---
**1. Query, Key, Value 계산***

각 입력 $x_i$에 대해:
- Query: $q_i = W_q x_i + b_q = \begin{bmatrix}1 \\ 1\end{bmatrix} x_i + b_q = \begin{bmatrix}x_i \\ x_i\end{bmatrix} + b_q$
- Key: $k_i = W_k x_i + b_k = \begin{bmatrix}1 \\ 1\end{bmatrix} x_i + b_k = \begin{bmatrix}x_i \\ x_i\end{bmatrix} + b_k$
- Value: $v_i = W_v x_i + b_v = \begin{bmatrix}1 \\ 1\end{bmatrix} x_i + b_v = \begin{bmatrix}x_i \\ x_i\end{bmatrix} + b_v$

---
**2. Attention Score 계산**

Attention score는 softmax 없이 raw dot-product로 사용되므로:
$$

\text{score}_{ij} = \frac{q_i^T k_j}{\sqrt{D_k}} = \frac{(W_q x_i + b_q)^T (W_k x_j + b_k)}{\sqrt{2}}

$$
예를 들어 $x_l = -1$, $x_r = 1$일 때:
$q_l = \begin{bmatrix} -1 \\ -1 \end{bmatrix} + b_q,\quad$
$k_r = \begin{bmatrix} 1 \\ 1 \end{bmatrix} + b_k$
$\Rightarrow \text{score}_{lr} = \frac{( -1 + b_{q1})(1 + b_{k1}) + (-1 + b_{q2})(1 + b_{k2}) }{\sqrt{2}}$

---
**3. Attention Output 계산**

Softmax가 없으므로 가중치는 그대로 사용:
$y_i = \alpha_{i1} v_1 + \alpha_{i2} v_2 \quad \text{where} \quad \alpha_{ij} = \text{score}_{ij},\quad v_j = W_v x_j + b_v$

---
**4. 최종 출력**

최종 Linear layer를 통과한 후 sigmoid 함수 적용:
$\hat{y}_i = \sigma(w^T y_i + b) \quad\text{where}\quad w \in \mathbb{R}^{2},\ b \in \mathbb{R}$