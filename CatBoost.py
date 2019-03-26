from catboost import CatBoostClassifier, Pool
import pandas as pd
import shap
import numpy as np

df_train = pd.read_csv("csv/my_bank.csv", sep=',').to_numpy()
df_test = pd.read_csv("csv/bank_test.csv", sep=',').to_numpy()

X_train, Y_train = df_train[:, 0:13], df_train[:, 13]

cat_features = [1, 2, 3, 5, 6, 7, 12]
features = ["age","marital","education","default","balance","housing","loan","contact","duration","campaign","pdays","previous","poutcome"]

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

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Pool(X_train, Y_train, cat_features=cat_features))

#shap.summary_plot(shap_values, X_train)

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

importances = model.feature_importances_
np.set_printoptions(precision=3, suppress=True)
print('Feature importances: ', np.array(importances))
shap.summary_plot(shap_values, X_train, feature_names=features)
shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=features)
model.save_model("model.cbm", format="cbm")