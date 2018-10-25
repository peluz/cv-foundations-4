# cv-foundations-4

## Conteúdo
 1. [Requisitos](#requisitos)
 2. [Estrutura](#estrutura)
 3. [Uso](#uso)

## Requisitos 
1.  Python 3.5.2	
2.  OpenCV 3.3.0
3.  Keras 2.2.4
4.  Matplotlib 2
5.  Scikit-learn 0.20
6.  Tensorflow 1.11

## Estrutura
- Pasta relatorio com código fonte do relatório
- Arquivo Araujo_Pedro__Ramos_Raphael.pdf com o relatório
- Pasta src contendo o código principal do projeto: pd4.py.

## Uso
- A partir do diretório raiz rodar com configurações padrão:
	```bash
	python ./src/pd3.py --r[número do requisito]
	```
-  [número do requisito] corresponde a 1 ou 2 dependendo da técnica a ser testada.
- [Repositório do github](https://github.com/peluz/cv-foundations-4)
- Requisito 1:
	- PREENCHER
- Requisito 2:
	- Flags de uso:
		- --imageSize tamanho das subimagens que o modelo obterá da imagem de treinamento original, pode ser 100, 200 ou 250.
		- --batchSize tamanho do batch de treinamento e avaliação.
		- --freeze não treinar camadas do extrator de características
		- --randomInit não usar pesos pretreinados
		- --train treinar o modelo, não apenas avaliar
		- --model Nome do modelo a ser salvo/carregado
	- Caso use a flag de treino, o modelo especificado será treinado
	- Após, será exibido o loss, a acurácia e o jaccard da imagem de teste
	- Por fim, será exibida a segmentação da imagem de teste, que será salva no diretório raiz.