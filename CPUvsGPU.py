import timeit

from catboost import CatBoostClassifier
from catboost.datasets import epsilon

train, test = epsilon()

X_train, y_train = train.iloc[:,1:], train[0]
X_test, y_test = train.iloc[:,1:], train[0]

print(len(train))

def train_on_cpu():
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.03
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=10
    );

cpu_time = timeit.timeit('train_on_cpu()',
                         setup="from __main__ import train_on_cpu",
                         number=1)

print('Time to fit model on CPU: {} sec'.format(int(cpu_time)))


def train_on_gpu():
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.03,
        task_type='GPU'
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=10
    );


gpu_time = timeit.timeit('train_on_gpu()',
                         setup="from __main__ import train_on_gpu",
                         number=1)

print('Time to fit model on GPU: {} sec'.format(int(gpu_time)))
print('GPU speedup over CPU: ' + '%.2f' % (cpu_time / gpu_time) + 'x')