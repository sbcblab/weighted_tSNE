# Weighted t-SNE

Welcome!

## How to use

To configure all the hyperparameters of Weighted t-SNE, you only need to create a ```config.py``` file. An example can be downloaded [here](config.py). It also contains the necessary documentation. To set the weights of each features you should use a .csv file as in this [example](weights.csv).

You will need Python 3 to run this code. Check if the needed libraries are installed with:

```
python3 check_dep.py
```
And for the weighted t-SNE visualization, run:
```
python3 wtsne.py config.py
```

## Datasets

You can download the datasets used in the experiments [here](DATA.zip) and the scores from the feature scorers [here](selections.zip).

The complete CuMiDa can be found here: [https://sbcb.inf.ufrgs.br/cumida](https://sbcb.inf.ufrgs.br/cumida)

## Experiments

You can download the complete configuration of the experiments [here](configurations).

Interactive plots of the results can be seen here: [https://sbcblab.github.io/wtsne](https://sbcblab.github.io/wtsne)

## Libraries

This implementation of relevance aggregation uses the following [Python 3.7](https://www.python.org/) libraries:

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://www.scipy.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Plotly](https://plotly.com/python/)
- [openTSNE](https://opentsne.readthedocs.io/en/latest/)
- [Relevance Aggregation](https://github.com/sbcblab/RelAgg.git)

## How to cite

If you use our code, methods, or results in your research, please consider citing the main publication of weithed t-SNE:

To be published.

Bibtex entry:
```
@article{grisci2021relevance,
  title={},
  author={},
  journal={},
  year={},
  doi = {},
  publisher={}
}
```

## Contact information

- [Bruno I. Grisci](https://orcid.org/0000-0003-4083-5881) - PhD candidate ([Institute of Informatics](https://www.inf.ufrgs.br/site/en) - [UFRGS](http://www.ufrgs.br/english/home))

    - bigrisci@inf.ufrgs.br

- [Prof. Dr. Mario Inostroza-Ponta](https://orcid.org/0000-0003-1295-8972) - Associate Professor ([Departamento de Ingeniería Informática](https://informatica.usach.cl/) - [USACH](https://www.usach.cl/))

    - mario.inostroza@usach.cl

- [Prof. Dr. Marcio Dorn](https://orcid.org/0000-0001-8534-3480) - Associate Professor ([Institute of Informatics](https://www.inf.ufrgs.br/site/en) - [UFRGS](http://www.ufrgs.br/english/home))

    - mdorn@inf.ufrgs.br

- http://sbcb.inf.ufrgs.br/