# Weighted t-SNE

Welcome!

Weighted t-SNE is a method for...

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

## Data sets

You can download the datasets used in the experiments [here](DATA/README.md).

## Results

If you are looking for the trained Keras models and resulting table heatmaps from the main paper, you can find them [here](RESULTS).

## Libraries

This implementation of relevance aggregation uses the following [Python 3.7](https://www.python.org/) libraries:

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://www.scipy.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Plotly](https://plotly.com/python/)
- [openTSNE](https://opentsne.readthedocs.io/en/latest/)

## How to cite

If you use our code, methods, or results in your research, please consider citing the main publication of relevance aggregation:

- Bruno Iochins Grisci, Mathias J. Krause, Marcio Dorn. _Relevance aggregation for neural networks interpretability and knowledge discovery on tabular data_, Information Sciences, Volume 559, June **2021**, Pages 111-129, DOI: [10.1016/j.ins.2021.01.052](https://doi.org/10.1016/j.ins.2021.01.052)

Bibtex entry:
```
@article{grisci2021relevance,
  title={Relevance aggregation for neural networks interpretability and knowledge discovery on tabular data},
  author={Grisci, Bruno Iochins and Krause, Mathias J and Dorn, Marcio},
  journal={Information Sciences},
  year={2021},
  doi = {10.1016/j.ins.2021.01.052},
  publisher={Elsevier}
}
```

## Contact information

- [Bruno I. Grisci](https://orcid.org/0000-0003-4083-5881) - PhD student ([Institute of Informatics](https://www.inf.ufrgs.br/site/en) - [UFRGS](http://www.ufrgs.br/english/home))

    - bigrisci@inf.ufrgs.br

- [Dr. Marcio Dorn](https://orcid.org/0000-0001-8534-3480) - Associate Professor ([Institute of Informatics](https://www.inf.ufrgs.br/site/en) - [UFRGS](http://www.ufrgs.br/english/home))

    - mdorn@inf.ufrgs.br

- http://sbcb.inf.ufrgs.br/