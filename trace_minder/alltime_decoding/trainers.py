from sklearn.model_selection import cross_val_score


class Trainer:
    def __init__(self, scoring, cv):
        self.scoring = scoring
        self.cv = cv

    def __call__(self, model, X, y) -> float:
        scores = cross_val_score(model, X, y, scoring=self.scoring, cv=self.cv)
        return scores.mean()
