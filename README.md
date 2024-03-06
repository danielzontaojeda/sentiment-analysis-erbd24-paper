# Utilizando a quantificação na análise de sentimentos em *reviews* de produtos

Material suplementar.

## Descrição

Este repositóroio contém o código utilizado para obter os resultados do artigo submetido para o [ERBD 2024](https://web.farroupilha.ifrs.edu.br/erbd24/chamada.php). 


## Instalação

```
git clone https://github.com/danielzontaojeda/sentiment-analysis-erbd24-paper
cd sentiment-analysis-erbd24-paper
pip install -R requirements.txt
```

## Execução
No geral:

```
python run [dataset_name] [output_name]
```

Os datasets devem estar na pasta `dataset`, e os resultados serão gerados na pasta `output`.

```
python run games.csv games.csv
python run iphone.csv iphone.csv
python run lg.csv lg.csv
python run samsung.csv samsung.csv
python run shoes.csv shoes.csv
python run xiaomi.csv xiaomi.csv
```

Para que não seja necessária a execução dos quantificadores, os resultados já foram gerados. É possível ver a análise realizada no arquivo `analysis.ipynb`.