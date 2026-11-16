# 🛡️ Detecção de Fraudes em Cartões de Crédito com Machine Learning

Este projeto apresenta um pipeline completo de Machine Learning desenvolvido para a comparação de algoritmos na detecção de fraudes em cartões de crédito. O principal desafio abordado é o **desbalanceamento extremo** da base de dados, onde aproximadamente **99,83%** das transações são legítimas e apenas **0,17%** correspondem a fraudes.

---

## 📊 O Desafio do Desbalanceamento Extremo

Trabalhar com classes altamente desbalanceadas altera fundamentalmente a forma como avaliamos os modelos, pois as métricas tradicionais passam a mascarar a real capacidade de aprendizado do algoritmo.

### 1. A Acurácia torna-se inútil (O Paradoxo da Acurácia)
Se criarmos um modelo "preguiçoso" que nunca aprende padrões e simplesmente classifica **100%** das transações como `0` (não é fraude), ele atingirá uma acurácia de **99,83%**.
* Em um problema equilibrado ($50\% / 50\%$), $99,8\%$ de acurácia seria um resultado excepcional.
* Na detecção de fraudes, $99,83\%$ de acurácia representa um **fracasso total**, pois o modelo deixou passar $100\%$ dos golpes. Por isso, a acurácia global deve ser ignorada neste cenário.

### 2. O Peso Assimétrico dos Erros (Custo de Negócio)
Na prevenção a fraudes, os dois tipos de erro possuem impactos financeiros e operacionais completamente distintos:
* **Falso Negativo (Deixar uma fraude passar):** O banco ou o cliente perde o valor da transação diretamente.
* **Falso Positivo (Bloquear uma compra legítima):** Gera atrito, causa insatisfação no cliente honesto (risco de *churn*/abandono do cartão) e sobrecarrega a equipe de atendimento.

Como a classe positiva (`1`) é raríssima, isolamos a avaliação exclusivamente nela utilizando:
* **Precision:** Controla a taxa de bloqueio indevido de clientes honestos.
* **Recall:** Mede a proporção de fraudes reais que foram efetivamente barradas.
* **F1-Score:** Média harmônica que resume o equilíbrio entre *Precision* e *Recall*.

### 3. Médias Simples Mascaram o Desempenho Real
Ao analisar o `classification_report`, nos deparamos com duas médias globais: `macro avg` e `weighted avg`.
* **`weighted avg` (Média Ponderada):** Atribui peso proporcional ao volume de cada classe. Como $99,83\%$ dos registros pertencem à classe `0`, essa métrica será puxada para perto de $1.00$, ocultando um eventual desempenho ruim na detecção de fraudes.
* **Onde focar:** Devemos observar diretamente as métricas da **classe minoritária (`1`)** ou o **`macro avg`**, que atribui peso igual ($50\% / 50\%$) para ambas as classes na média final.

---

## ⚙️ Preparação e Engenharia de Dados

Na etapa de pré-processamento, os dados passaram pelas seguintes transformações:
1. **Divisão Estratificada:** Separação entre treino e teste utilizando `stratify`, garantindo que a proporção original de $0,17\%$ de fraudes fosse preservada em ambos os conjuntos.
2. **Padronização:** Aplicação do `StandardScaler` (`scikit-learn`) para colocar as variáveis na mesma escala.
3. **Transformação Logarítmica na variável `Amount`:** Utilização da função `np.log1p()` (`NumPy`).

### Por que aplicar `np.log1p(x)` na coluna `Amount`?
Em bases financeiras, o valor das transações (`Amount`) possui uma distribuição altamente assimétrica (*skewed*): a grande maioria das compras tem valores baixos (café, transporte), enquanto pouquíssimas transações atingem valores altíssimos (eletrônicos, passagens aéreas). 

* **Tratamento de Outliers e Zeros:** Valores extremos podem distorcer o aprendizado do algoritmo. A função `np.log1p(x)` calcula $\ln(1 + x)$, o que lida perfeitamente com transações de valor igual a zero ($\ln(1 + 0) = \ln(1) = 0$), evitando erros matemáticos ou divisões por zero.
* **Efeito Prático:** A transformação comprime a escala dos valores altos e expande a visualização dos valores baixos. Uma transação de $\$10$ e uma de $\$10.000$ tornam-se muito mais próximas em ordem de grandeza, aproximando a distribuição de uma curva normal (gaussiana) e facilitando a convergência de algoritmos baseados em distância (como Regressão Logística, SVM e KNN).

---

## 🏆 Comparação de Modelos e Resultados

Abaixo estão os resultados obtidos para a **classe de fraude (`1`)** no conjunto de teste:

| Modelo | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.87 | 0.66 | 0.75 |
| **Random Forest** | 0.84 | 0.76 | 0.79 |
| **XGBoost** | **0.94** | **0.78** | **0.85** |

### Análise Comparativa
A avaliação evidencia a superioridade progressiva dos modelos baseados em conjuntos de árvores (*ensembles*) sobre a abordagem linear na captura de padrões complexos de fraude:

* 🥇 **XGBoost (Melhor Desempenho Geral):** Destacou-se como o modelo mais equilibrado e eficiente, atingindo o maior **F1-Score ($0.85$)**. Sua **Precision de $0.94$** garante que praticamente todos os alertas gerados correspondem a fraudes reais (minimizando o atrito com clientes legítimos), enquanto o **Recall de $0.78$** assegura a captura da maior parcela de transações ilícitas entre todos os algoritmos avaliados.
* 🥈 **Random Forest (Intermediário Sólido):** Apresentou uma evolução expressiva em relação ao modelo linear, elevando o **Recall para $0.76$** (capturando $10$ pontos percentuais a mais de fraudes que a Regressão Logística) e mantendo uma boa **Precision ($0.84$)**, resultando em um **F1-Score de $0.79$**.
* 🥉 **Logistic Regression (Baseline Conservador):** Embora tenha entregue uma alta **Precision ($0.87$)** — indicando baixo índice de falsos alarmes quando sinaliza uma fraude —, sua natureza linear limitou a captura de relações não lineares, resultando no menor **Recall ($0.66$)** e deixando passar cerca de um terço das fraudes reais (**F1-Score de $0.75$**).

### Limiar de Decisão Escolhido (*Threshold*)
Utilizou-se o limiar padrão de **$0.50$** (probabilidades acima de $50\%$ são classificadas como fraude). No **XGBoost**, esse corte mostrou-se altamente calibrado, mantendo a *Precision* em $0.94$ sem sacrificar a detecção real (*Recall* de $0.78$).

---

## 🔍 Interpretabilidade do Modelo (SHAP Values)

A análise de explicabilidade com **SHAP** sobre o modelo campeão (**XGBoost**) revelou como as decisões são tomadas:

* **Principais Fatores de Decisão:** As componentes **`V4` ($+0.71$)** e **`V14` ($+0.44$)** são de longe as variáveis mais determinantes para distinguir transações legítimas de fraudes, seguidas por **`V26` ($+0.33$)**, **`Time` ($+0.28$)** e **`V16` ($+0.25$)**.
* **Relevância Temporal:** A presença da variável `Time` entre as quatro mais influentes confirma que o momento e a frequência das transações exercem papel crítico na identificação de comportamentos suspeitos.
* **Decisão Distribuída:** Embora `V4` e `V14` liderem o impacto individual, o agrupamento *"Sum of 23 other features"* ($+2.77$) demonstra que o XGBoost não depende de uma regra isolada, mas sim da combinação de dezenas de pequenos sinais das demais variáveis para formar um veredito robusto.

---

## 🧪 Experimentos Adicionais e Lições Aprendidas

Durante o desenvolvimento do pipeline, diferentes hipóteses foram testadas para validar o comportamento dos dados:

### 1. Engenharia de Atributos (*Feature Engineering*)
Foram criadas novas variáveis comportamentais a partir das colunas originais `Time` e `Amount`, incluindo:
* Intervalo de tempo entre transações consecutivas (sequência temporal).
* Média móvel e desvio do valor em relação às transações recentes.

> **Resultado:** A inclusão desses atributos não gerou ganho significativo nas métricas. Esse comportamento é esperado neste dataset específico, pois a ausência de um identificador único de cliente/cartão impede o rastreamento individualizado (as janelas temporais refletem apenas o fluxo global do sistema), além de as componentes principais (`V1` a `V28`, oriundas de PCA) já concentrarem a maior parte da variância discriminante.

### 2. Estratégias de Balanceamento de Classes
Foram avaliadas técnicas de reamostragem manual dos dados de treino:
* **Undersampling:** Descartou um volume excessivo de transações legítimas, causando perda de informação relevante.
* **Oversampling (SMOTE):** Introduziu ruído sintético, tornando os classificadores excessivamente sensíveis e gerando um alto índice de falsos alarmes (queda acentuada na *Precision*).

> **Conclusão:** A melhor abordagem comprovada foi **manter a distribuição original dos dados** e tratar o desbalanceamento diretamente nos hiperparâmetros nativos dos algoritmos — utilizando `scale_pos_weight` no **XGBoost** e `class_weight='balanced'` no **Random Forest**.