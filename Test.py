from catboost import CatBoostClassifier
import time

start_time = time.time()

train_data = [[0,3],
              [4,1],
              [8,1],
              [9,1]]
train_labels = [0,0,1,1]

model = CatBoostClassifier(iterations=1000, task_type = "CPU")
model.fit(train_data, train_labels, verbose = False)

print(time.time()-start_time)