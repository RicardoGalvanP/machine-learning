# Movie Recommender Systems: SVD, Hybrid Model and LLM Agent

## Version

- English: [`movie_recommender_systems_en.ipynb`](movie_recommender_systems_en.ipynb) (outputs cleared; re-run it in Colab)
- Spanish: [`movie_recommender_systems_es.ipynb`](movie_recommender_systems_es.ipynb) (with the outputs from the real run)

## Problem Statement

Given a user's favorite movies, recommend others they are likely to enjoy and explain why. The project compares three ideas on the same footing: matrix factorization (truncated SVD and ALS), a hybrid model that adds content features (genre, decade), and an LLM agent that writes the justifications.

The question is not "which model is fanciest" but "which one actually beats the simple alternatives". Popularity and a user + movie bias baseline are strong competitors in recommendation, so every model is measured against them on held-out data.

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) (GroupLens): 1,000,209 ratings from 6,040 users on 3,706 movies. Only 4.47% of the user-movie matrix has a rating (95.53% sparse). Popularity is very skewed: the median movie has 124 ratings, the most rated has 3,428.

## Tech Stack

Python, NumPy, pandas, SciPy (sparse matrices), scikit-learn (TruncatedSVD), matplotlib, Gemini API (`google-genai`) for the agent.

## Methodology

- Random split of the ratings: 70% train, 10% validation, 20% test. Every fitted component (means, biases, SVD, ALS, similarities) uses train only. Validation chooses k, regularization and weights; test is used once per block.
- Rating prediction (RMSE): global, per-user and per-movie means; a regularized user + movie bias baseline; truncated SVD on three versions of the matrix (zeros for missing, user-centered, bias-centered); and an ALS written from scratch on the bias residual that optimizes observed ratings only.
- Top-10 ranking: seeds are each user's train ratings of 4 or more; relevant items are held-out ratings of 4 or more; 1,500 evaluation users. Metrics: precision, recall, NDCG@10, hit rate, catalog coverage and average popularity of the recommended movies. Baselines: random and popularity. Bootstrap 95% confidence intervals over users.
- Hybrid model: weighted concatenation of unit-normalized blocks (SVD, genre, decade, and optionally an audience block built from gender and age).
- Five-favorites profile: item-item cosine similarity aggregated by `mean`, `max`, or `coverage` (retrieve per favorite, round-robin).
- LLM agent: receives the candidates and writes a justification per movie. A code validator checks format, rating, closest favorite and the genres it cites, and the coverage of the favorites is computed in code, not by the model.

## Key Results

Rating prediction, test RMSE (lower is better):

| Model | RMSE | vs. bias baseline |
|---|---|---|
| Global mean | 1.116 | +22.6% |
| User + movie bias baseline | 0.910 | reference |
| SVD, zeros for missing (k=20) | 2.518 | +176.8% |
| SVD, user-centered (k=20) | 0.968 | +6.4% |
| SVD, bias-centered (k=20) | 0.887 | -2.5% |
| ALS on the bias residual (k=50) | **0.868** | **-4.6%** |

Top-10 ranking, test set (1,500 users):

| Model | NDCG@10 | Hit rate | Catalog coverage |
|---|---|---|---|
| Random | 0.005 | 0.05 | 98.3% |
| Popularity | **0.178** | 0.61 | 1.7% |
| SVD + cosine (mean of seeds) | 0.142 | 0.61 | 31.2% |
| Hybrid without audience (mean) | 0.133 | 0.56 | 28.4% |
| ALS (predicted rating) | 0.072 | 0.38 | 13.6% |
| Hybrid with gender and age | 0.067 | 0.32 | 26.9% |

- Zero-filled SVD is useless for prediction, and choosing k by explained variance (k = 559 for 80%) gives a validation RMSE 5.4% worse than the k = 20 chosen on held-out error. Variance explained measures how well a matrix of mostly fake zeros is rebuilt, not how well ratings are predicted.
- Popularity wins on NDCG@10 (SVD is 0.035 lower, bootstrap 95% CI of the difference [-0.046, -0.025]). SVD matches its hit rate (0.61) while recommending 19 times more of the catalog and less popular movies.
- Predicting ratings well does not mean ranking well: ALS has the best RMSE and a much lower NDCG than SVD with cosine similarity.
- Adding genre and decade did not improve plain SVD, and the gender-and-age block made the ranking clearly worse (validation NDCG@10 from 0.055 to 0.039 in the clean comparison). There is no performance reason to use sensitive attributes here.
- With five favorites, averaging similarities leaves 1.8 of the 5 unrepresented on average. Retrieving per favorite represents all five, at a cost of 16-18% of NDCG@10 (SVD: 0.103 to 0.084).
- LLM agent: 4 runs (2 profiles x 2 strategies), all valid on the first attempt according to the validator. With `mean`, one profile left Braveheart unrepresented and the agent said so in its assumptions.

## Limitations

- Offline evaluation: unrated movies count as non-relevant, the split is random (not temporal) and a single seed is used.
- The agent was tested on only 4 runs with one model, and the validator does not check the similarity values the agent cites. A third profile was dropped because of the API free-tier daily quota.
- The agent only picks among the candidates it receives: it improves the explanation, not the selection.
- The agent is not deterministic: the same profile and candidates can yield a different order and different wording across runs. This makes bias (favoring popular movies or dominant genres) and hallucinations (cited facts not present in the tables) a risk, and the validator does not check the cited similarities or the order of the list. Repeating each configuration several times, and verifying both in code, is the next step.
- Movies with very few ratings (for example, one with 5 ratings) can rank high because their latent vector is unreliable; a minimum-ratings rule or shrinkage is the next step.

## What I Learned

- A strong baseline is the real test: a bias model and popularity are hard to beat, and several "sophisticated" variants did not.
- Choose k and hyperparameters on held-out error, never on explained variance, and keep a separate validation split so the test set is used once.
- Rating-prediction accuracy and ranking quality are different problems; a model can be good at one and poor at the other. Accuracy alone also hides popularity bias, so coverage and the popularity of recommendations should be reported too.
- Sensitive attributes should earn their place with evidence; here they only hurt.
- LLM output needs verification in code (format, facts, coverage), and the verification itself has limits that must be stated.

## About

Ricardo Galván Páez, Mechatronics Engineer and Software Support Engineer (WMS/OMS, SQL), studying the M.Sc. in Applied Artificial Intelligence at Tecnológico de Monterrey. This project was built as part of that program.

[GitHub](https://github.com/RicardoGalvanP) · [LinkedIn](https://www.linkedin.com/in/ricardo-galvan-paez)
