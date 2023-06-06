# README

In the given assignment, I have implemented frequency-based modelling approaches, such as Singular Value Decomposition (SVD), and comparing it with the embeddings obtained using one of the variants of Word2vec, such as CBOW implementation with Negative Sampling.

I have divided the asignment into these two parts and thus have two corresponding files. The first part is in the file `assgn3_svd.py` and the second part is in the file `assgn3_cbow.py`. The first part is the implementation of SVD and the second part is the implementation of CBOW with Negative Sampling.

To run the files:
```
python3 assgn3_svd.py
python3 assgn3_cbow.py
```

Be careful about the path where you are saving the model while loading it. For now, I have saved the cbow model as cbow.pth and have saved the svd embeddings in the file svd_1.npy.

There are quite a few overlapping functions in both the files so please free to edit and analyse them as per your requirements (most notably the functions to find the nearest words).

While loading the source file, please rename it to 'reviews.json' and place it in the same directory as the python files. The file is too large to be uploaded on github/moodle.