import json

import pandas as pd
import numpy as np



data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, np.nan],  # np.nan表示NA
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
data1 ={'predict':[1,3,4,6,6]}

data_frame =pd.DataFrame(data,
          # index=['a','b','c','d','e'],
          # index = range(1,6)
         )  # 默认生成整数索引, 字典的键作列,值作行
data1_frame = pd.DataFrame(data1)

print(data_frame)
print(data_frame['state'])
a = pd.concat([data_frame,data1_frame],axis=1)
print(a)
print("====")
#创建列名称为“perdiction”的空列
df = pd.DataFrame(columns=['prediction'])
results = [[(124, 77, 144, 98), '1'], [(281, 34, 357, 124), '0']]
results1 = [[(12, 75, 44, 8), '1'], [(21, 4, 57, 24), '0']]
# locations = [[(124, 77, 144, 98), '1']]
df = df.append({'prediction':str(results)},ignore_index=True)
df = df.append({'prediction':str(results1)},ignore_index=True)
print(df)
print("json =",json.dumps(df,ensure_ascii=False))
print("===========")

for i in range(5):
    print(data_frame['state'][i])

d = {'a': {'tp': 26, 'fp': 112},
     'b': {'tp': 26, 'fp': 91},
     'c': {'tp': 23, 'fp': 74}}

df_index = pd.DataFrame.from_dict(d, orient='index')
print(df_index)
print(df_index['tp'])

df_columns = pd.DataFrame.from_dict(d,orient='columns')
print(df_columns)

print("-------------------------------")


# def load_model(weight_path):
#     weight_path+=2
#     return weight_path
#
# def predict(model):
#     model +=2
#     return model
#
# class ABC():
#     def __init__(self,weight_path):
#         self.weight_path = weight_path
#
#     def predict(self):
#         model = self.model
#         print("model =",model)
#         out = predict(model)
#         return out
#
#     def prepare(self):
#         self.model = load_model(self.weight_path)
#
#
# if __name__ == '__main__':
#     abc = ABC(1)
#     a1 =abc.prepare()
#     a2 =abc.predict()
#     print(a1)
#     print(a2)


    