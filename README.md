# Algoritma Machine Learning for Product Recommendation

This project implements two machine learning algorithms for product recommendation: User-Based Collaborative Filtering and Content-Based Filtering. These algorithms are integrated into a Laravel project for recommending products.

## Description

The project consists of the following files:

- `connection.py`: This file contains functions to connect to the Laravel API and retrieve data.
- `preprocessing.py`: This file preprocesses text data and computes user vectors for User-Based Collaborative Filtering. It also preprocesses product data and calculates TF-IDF vectors for Content-Based Filtering.
- `main.py`: This file contains Flask APIs for getting recommendations based on User-Based and Content-Based Filtering.

## Installation

To run this project, you need to install the following dependencies:

- Python 3.x
- Flask
- Requests
- Sastrawi
- NLTK
- scikit-learn
- pandas

You can install these dependencies using pip:

```bash
pip install Flask requests Sastrawi nltk scikit-learn pandas
