from catboost import CatBoostClassifier

train_data = [[0,3],
              [4,1],
              [8,1],
              [9,1]]
train_labels = [0,0,1,1]

model = CatBoostClassifier(iterations=1000, task_type = "GPU")
model.fit(train_data, train_labels, verbose = False)