# Weighted t-SNE

Welcome!

Relevance aggregation is a method for computing relevance scores for multilayer feed-forward neural networks trained on tabular data. It can be used to inspect the behavior of the trained models, identify biases, and for knowledge discovery. Besides the numerical values, the relevance scores can be visualized with table heatmaps and weighted t-SNE.

## How to use

To configure all the hyperparameters of relevance aggregation, you only need to create a ```config.py``` file. An example can be downloaded [here](config.py). It also contains the necessary documentation.

You will need Python 3 to run this code. Check if the needed libraries are installed with:

```
python3 check_dep.py
```
And for the weighted t-SNE visualization, run:
```
python3 visualize.py config.py
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