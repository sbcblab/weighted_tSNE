# Weighted t-SNE

Welcome!

While a vast literature on high-dimensional data visualization is available, there are not many works regarding the visualization of the effects of feature scorers on the original data. These algorithms can attribute numerical importance scores to each feature of multi-dimensional datasets and range from statistical filters to embedded machine learning models. These importance scores can be used in several applications, such as feature selection, knowledge discovery, and machine learning interpretability. However, there are several distinct feature scorers to choose from, and it is often the case that there is no single metric or ground truth available to guarantee the quality of their results. In this scenario, visualization can become a valuable tool to inform the decision of which method to choose and how good are its results. With this goal in mind, this work expands the popular t-SNE algorithm presenting the "weighted t-SNE." It modifies the relationship between data points in the embedded 2D space of the visualization to reflect the importance of each dimension of the original datasets as assessed by a feature scorer. The results show that each feature scorer produces unique visualizations and that weighted t-SNE can be an inspection tool to compare and choose the one that better suits a given dataset and task. Weighted t-SNE can visually display the importance of features as learned by machine learning models, which could better represent their internal patterns.

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

You can download the datasets used in the experiments [here](IEEEVIS_data/DATA.zip) and the scores from the feature scorers [here](IEEEVIS_data/selections.zip).

The complete CuMiDa can be found here: [https://sbcb.inf.ufrgs.br/cumida](https://sbcb.inf.ufrgs.br/cumida)

## Results

You can download the complete results of the experiments below:

- [XOR](IEEEVIS_data/RESULTS/xor.zip)
- [Synth](IEEEVIS_data/RESULTS/synth.zip)
- [Liver](IEEEVIS_data/RESULTS/liver.zip)
- [Prostate](IEEEVIS_data/RESULTS/prostate.zip)
- [Regression](IEEEVIS_data/RESULTS/regression.zip)

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