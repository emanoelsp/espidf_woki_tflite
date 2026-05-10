# Hello World com TensorFlow Lite Micro

**Disciplina:** IA Embarcada  
**Data:** 10/05/2026  
**Aluno:** Emanoel Spanhol

---

## O que foi feito

Este projeto reproduz o exemplo clássico **Hello World** do TensorFlow Lite Micro. O objetivo é rodar uma rede neural diretamente em um microcontrolador — sem sistema operacional, sem sistema de arquivos, com pouquíssima memória disponível.

A tarefa da rede é simples: dado um valor de entrada `x`, prever `y = sin(x)`. Ao longo do tempo, o microcontrolador percorre os valores de 0 até 2π repetidamente, gerando uma onda senoidal ponto a ponto.

O projeto foi executado no simulador **Wokwi**, que emula o hardware sem precisar de uma placa física.

---

## O modelo

A rede neural foi treinada para aproximar a função seno. Ela é composta por três camadas:

- **Camada 1:** 16 neurônios com ativação ReLU  
- **Camada 2:** 16 neurônios com ativação ReLU  
- **Camada 3:** 1 neurônio de saída (valor contínuo)

Após o treinamento, o modelo foi convertido para o formato **TFLite** e quantizado em **INT8** — uma técnica que reduz os pesos de 32 bits para 8 bits, tornando o modelo muito mais leve. O resultado final ocupa apenas **3,4 KB**, o suficiente para caber na memória flash de um microcontrolador.

O modelo é armazenado como um array de bytes diretamente no código C, sendo compilado junto com o firmware.

---

## Análise da saída

No monitor serial do Wokwi, a saída do programa aparece no seguinte formato:

```
x_value: 0.000000, y_value: 0.000000
x_value: 0.314159, y_value: 0.309017
x_value: 0.628318, y_value: 0.587785
...
```

A cada iteração, o programa:
1. Calcula o próximo valor de `x` dentro do intervalo 0 a 2π
2. Passa esse valor pelo modelo e obtém a predição de `y`
3. Imprime o par `(x, y)` no serial

Os valores de `y` seguem a curva esperada do seno — próximos de zero no início, crescendo até ~1, voltando a zero, descendo até ~-1 e retornando. Isso confirma que o modelo aprendeu a aproximar corretamente a função seno, mesmo rodando de forma quantizada em um ambiente com memória extremamente limitada.

---

## Print da simulação

![Wokwi rodando Hello World](Captura%20de%20Tela%202026-05-10%20às%2020.21.04.png)
