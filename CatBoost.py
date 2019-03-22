from catboost import CatBoostClassifier
from catboost import Pool
import pandas as pd

df_train = pd.read_csv("csv/bank.csv", sep=',').to_numpy()
df_test = pd.read_csv("csv/bank_test.csv", sep=',').to_numpy()

X_train, Y_train = df_train[:, 0:13], df_train[:, 13]

cat_features = [1, 2, 3, 5, 6, 7, 12]

model = CatBoostClassifier(iterations=1000,
                           learning_rate=0.03,
                           depth=6,
                           gpu_ram_part=0.95,
                           task_type='CPU',
                           loss_function='Logloss'
                           )

model.fit(X_train, Y_train, cat_features,
          # use_best_model=True, eval_set=(df_test[:, 0:13], df_test[:, 13])
          )

X, Y = df_test[:, 0:13], df_test[:, 13]
answer = model.predict(X)
loss = 0
for x in range(len(answer)):
    if (Y[x] == 1 and answer[x] == 0):
        loss += 1
    if (Y[x] == 0 and answer[x] == 1):
        loss += 1
print("loss = ", loss)

X, Y = df_test[:, 0:13], df_test[:, 13]
answer = model.predict(X)

model.save_model("model.cbm", format="cbm")